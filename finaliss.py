import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tkinter as tk
from gekko import GEKKO

root = tk.Tk()
#root.attributes('-fullscreen', True)
root.title('Pendulum Simulation')
#root.resizable(0, 0)
b1 = tk.Scale(root, from_=10, to=99, length=450, tickinterval=5,
              orient='horizontal')  #end
b1.set(80)
b1.grid(column=1, row=1, sticky='w')

b2 = tk.Scale(root, from_=5, to=50, length=450, tickinterval=5,
              orient='horizontal')  # m1
b2.set(10)
b2.grid(column=1, row=2, sticky='w')

b3 = tk.Scale(root, from_=1, to=10, length=450,
              tickinterval=1, orient='horizontal')  # m2
b3.set(1)
b3.grid(column=1, row=3, sticky='w')

b4 = tk.Scale(root, from_=0, to=20, length=450, tickinterval=2,
              orient='horizontal')  # u force
b4.set(1)
b4.grid(column=1, row=4, sticky='w')

b5 = tk.Scale(root, from_=0, to=10, length=450, tickinterval=1,
              orient='horizontal')  # v
b5.set(1)
b5.grid(column=1, row=5, sticky='w')

b6 = tk.Scale(root, from_=-20, to=20, length=450, tickinterval=4,
              orient='horizontal')  # q
b6.set(0)
b6.grid(column=1, row=6, sticky='w')

b7 = tk.Scale(root, from_=-3, to=3, length=450,  tickinterval=1, orient='horizontal') #y
b7.set(0)
b7.grid(column=1, row=7, sticky='w')

b8 = tk.Scale(root, from_=-20, to=20, length=450, tickinterval=5,
              orient='horizontal')  # angle of rod
b8.set(0)
b8.grid(column=1, row=8, sticky='w')

m1, m2, u, theta, q, y, v, end = 0, 0, 0, 0, 0, 0, 0, 0

label1 = tk.Label(root, text="y = ")
label1.grid(column=0, row=7)
label2 = tk.Label(root, text="Destination = ")
label2.grid(column=0, row=1)
label3 = tk.Label(root, text="m1 (Cart) = ")
label3.grid(column=0, row=2)
label4 = tk.Label(root, text="m2 (Bob) = ")
label4.grid(column=0, row=3)
label5 = tk.Label(root, text="u (Force) = ")
label5.grid(column=0, row=4)
label6 = tk.Label(root, text="v (Velocity) =")
label6.grid(column=0, row=5)
label7 = tk.Label(root, text="q (Angular vel) =")
label7.grid(column=0, row=6)
label8 = tk.Label(root, text="theta =")
label8.grid(column=0, row=8)


def clickc():
	b1.set(80)
	b2.set(10)
	b3.set(1)
	b4.set(1)
	b5.set(1)
	b6.set(0)
	b7.set(0)
	b8.set(0)


def clickb():
    global m1, m2, u, theta, q, y, v, end
    end = b1.get()
    m1 = b2.get()
    m2 = b3.get()
    u = b4.get()
    v = b5.get()
    q = b6.get()
    y = b7.get()
    theta = b8.get()
    start()


clicks = tk.Button(root, text="SIMULATE", padx=20, pady=5,
                   borderwidth=2, command=clickb)
clicks.grid(column=1, row=10)
clickr = tk.Button(root, text="RESET", padx=20, pady=5,
                   borderwidth=2, command=clickc)
clickr.grid(column=1, row=9)


m = GEKKO(remote=False)
m.time = np.linspace(0, 10, 100)
def start():
    end_loc = 90 #10-99
    #Parameters
    m1a = m.Param(value=m1)
    m2a = m.Param(value=m2)
    final = np.zeros(len(m.time))
    for i in range(len(m.time)):
        if m.time[i] < 8.5:
            final[i] = 0
        else:
            final[i] = 1
    final = m.Param(value=final)
    ua = m.Var(value=u)

    #State Variables
    theta_a = m.Var(value=theta)
    qa = m.Var(value=q)
    ya = m.Var(value=y)
    va = m.Var(value=v)

    epsilon = m.Intermediate(m2a/(m1a+m2a))

    #Defining the State Space Model
    m.Equation(ya.dt() == va)
    m.Equation(va.dt() == -epsilon*theta_a + ua)
    m.Equation(theta_a.dt() == qa)
    m.Equation(qa.dt() == theta_a - ua)

    m.Obj(final*ya**2)
    m.Obj(final*va**2)
    m.Obj(final*theta_a**2)
    m.Obj(final*qa**2)

    m.fix(ya, pos=end_loc, val=0.0)
    m.fix(va, pos=end_loc, val=0.0)
    m.fix(theta_a, pos=end_loc, val=0.0)
    m.fix(qa, pos=end_loc, val=0.0)

    m.Obj(0.001*ua**2)

    m.options.IMODE = 6  # MPC
    m.solve() 

    #Plotting the results
    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.plot(m.time, ua.value, 'm', lw=2)
    plt.legend([r'$u$'], loc=1)
    plt.ylabel('Force')
    plt.xlabel('Time')
    plt.xlim(m.time[0], m.time[-1])

    plt.subplot(222)
    plt.plot(m.time, va.value, 'g', lw=2)
    plt.legend([r'$v$'], loc=1)
    plt.ylabel('Velocity')
    plt.xlabel('Time')
    plt.xlim(m.time[0], m.time[-1])

    plt.subplot(223)
    plt.plot(m.time, ya.value, 'r', lw=2)
    plt.legend([r'$y$'], loc=1)
    plt.ylabel('Position')
    plt.xlabel('Time')
    plt.xlim(m.time[0], m.time[-1])

    plt.subplot(224)
    plt.plot(m.time, theta_a.value, 'y', lw=2)
    plt.plot(m.time, qa.value, 'c', lw=2)
    plt.legend([r'$\theta$', r'$q$'], loc=1)
    plt.ylabel('Angle')
    plt.xlabel('Time')
    plt.xlim(m.time[0], m.time[-1])

    plt.rcParams['animation.html'] = 'html5'

    x1 = ya.value
    y1 = np.zeros(len(m.time))
    x2 = 1*np.sin(theta_a.value)+x1
    x2b = 1.05*np.sin(theta_a.value)+x1
    y2 = 1*np.cos(theta_a.value)-y1
    y2b = 1.05*np.cos(theta_a.value)-y1

    fig = plt.figure(figsize=(16, 6.4))
    ax = fig.add_subplot(111, autoscale_on=False,
                        xlim=(-1.5, 0.5), ylim=(-0.4, 1.2))
    ax.set_xlabel('position')
    ax.get_yaxis().set_visible(False)

    crane_rail, = ax.plot([-1.5, 0.5], [-0.2, -0.2], 'k-', lw=4)
    start, = ax.plot([-1, -1], [-1.5, 1.5], 'k:', lw=2)
    objective, = ax.plot([0, 0], [-0.5, 1.5], 'k:', lw=2)
    mass1, = ax.plot([], [], linestyle='None', marker='s',
                    markersize=40, markeredgecolor='k',
                    color='orange', markeredgewidth=2)
    mass2, = ax.plot([], [], linestyle='None', marker='o',
                    markersize=20, markeredgecolor='k',
                    color='orange', markeredgewidth=2)
    line, = ax.plot([], [], 'o-', color='orange', lw=4,
                    markersize=6, markeredgecolor='k',
                    markerfacecolor='k')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    start_text = ax.text(-1.06, -0.3, 'start', ha='right')
    end_text = ax.text(0.06, -0.3, 'objective', ha='left')


    def init():
        mass1.set_data([], [])
        mass2.set_data([], [])
        line.set_data([], [])
        time_text.set_text('')
        return line, mass1, mass2, time_text


    def animate(i):
        mass1.set_data([x1[i]], [y1[i]-0.1])
        mass2.set_data([x2b[i]], [y2b[i]])
        line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        time_text.set_text(time_template % m.time[i])
        return line, mass1, mass2, time_text


    ani_a = animation.FuncAnimation(fig, animate,
                                    np.arange(1, len(m.time)),
                                    interval=40, blit=False, init_func=init)

    ani_a.save('Pendulum_Control.mp4',fps=42)

    plt.show()

root.mainloop()
