import tkinter as tk
import datetime
from numpy import sin, cos, radians
from math import pi, trunc
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import scipy.integrate as integrate
import matplotlib.pyplot as pp
import numpy as np
import matplotlib
from tkVideoPlayer import TkinterVideo
matplotlib.use('TKAgg')


dt = 0.05
Tmax = 10
t = np.arange(0.0, Tmax, dt)

#Y = .0 		# pendulum angular velocity
#th = pi/10  # pendulum angle
#x = .0		# cart position
#x0 = 0		# desired cart position
#Z = .0		# cart velocity

precision = 0.006
k = 1000.0  # Kalman filter coefficient
Kp_th = 50
Kd_th = 15
Kp_x = 3.1
Kd_x = 4.8

def trim(x, step):
    d = trunc(x / step)
    return step * d
#state = np.array([th, Y, x, Z, trim(th, precision), .0])


'''def step(t):
	if t < 5:
		return .0
	elif t >= 5 and t < 10:
		return 1.
	elif t >= 10 and t < 15:
		return -0.5
	else:
		return .0'''

root = tk.Tk()
root.attributes('-fullscreen', True)
root.title('Pendulum Simulation')
root.resizable(0,0)

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=6)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=16)
root.columnconfigure(4, weight=1)
root.columnconfigure(5, weight=1)
root.rowconfigure(0, weight=16) 
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)
root.rowconfigure(6, weight=1)

b1 = tk.Scale(root, from_=0, to=130, length=450, tickinterval=13, orient='horizontal') #cart position x10
b1.set(0)
b1.grid(column=1, row=1, sticky='w')
b2 = tk.Scale(root, from_=5, to=25, length=450, tickinterval=2, orient='horizontal') # Length of rod x10
b2.set(15)
b2.grid(column=1, row=2, sticky='w')
b3 = tk.Scale(root, from_=1, to=16, length=450, tickinterval=1, orient='horizontal') # mass x10
b3.set(5)
b3.grid(column=1, row=3, sticky='w') 
b4 = tk.Scale(root, from_=0, to=130, length=450, tickinterval=10, orient='horizontal')  # desired cart position x10
b4.set(50)
b4.grid(column=1, row=4, sticky='w')
b5 = tk.Scale(root, from_=0, to=80, length=450, tickinterval=5, orient='horizontal') # cart vel x10
b5.set(20)
b5.grid(column=1, row=5, sticky='w')
b6 = tk.Scale(root, from_=0, to=32, length=450, tickinterval=2, orient='horizontal') #pendulum ang vel x10
b6.set(0)
b6.grid(column=1, row=6, sticky='w')
'''b7 = tk.Scale(root, from_=0, to=50, length=450,  tickinterval=5, orient='horizontal')
b7.set(12)
b7.grid(column=1, row=6)'''
b8 = tk.Scale(root, from_=-40, to=40, length=450, tickinterval=5, orient='horizontal') #angle of rod
b8.set(0)
b8.grid(column=1, row=7, sticky='w')

label1 = tk.Label(root, text="Theta =")
label1.grid(column=0, row=7)
label2 = tk.Label(root, text="10* X =")
label2.grid(column=0, row=1)
label3 = tk.Label(root, text="10* L =")
label3.grid(column=0, row=2)
label4 = tk.Label(root, text="10* m =")
label4.grid(column=0, row=3)
label5 = tk.Label(root, text="10* X0 =")
label5.grid(column=0, row=4)
label6 = tk.Label(root, text="10* vel =")
label6.grid(column=0, row=5)
label7 = tk.Label(root, text="10* ang vel =")
label7.grid(column=0, row=6)

def clickc():
	b1.set(0)
	b2.set(15)
	b3.set(5)
	b4.set(50)
	b5.set(20)
	b6.set(0)
	b8.set(0)

def clickb():
	global x,x0,Y,L, Z,m,deg
	x = float(b1.get())/10.
	L = float(b2.get())/10.
	m = float(b3.get())/10.
	x0 = float(b4.get())/10.
	Z = float(b5.get())/10.
	deg = float(b8.get())
	Y = float(b6.get())/10.

	sim()
	print('Saved')
	outp()


clicks = tk.Button(root, text="SIMULATE", padx=20, pady=5, borderwidth=2, command=clickb)
clicks.grid(column=1, row=8)
clickr = tk.Button(root, text="RESET", padx=20, pady=5, borderwidth=2, command=clickc)
clickr.grid(column=1, row=9)

def sim():
	g = 9.81
	th = radians(deg)

	def derivatives(state, t):
		ds = np.zeros_like(state)

		_th = state[0]
		_Y = state[1]
		_x = state[2]
		_Z = state[3]
		# x0 = step(t)
		u = Kp_th * _th + Kd_th * _Y + Kp_x * (_x - x0) + Kd_x * _Z

		ds[0] = state[1]
		ds[1] = (g * sin(_th) - u * cos(_th)) / L
		ds[2] = state[3]
		ds[3] = u
		return ds
	state = np.array([th, Y, x, Z, trim(th, precision), .0])
	print("Integrating...")
	solution = integrate.odeint(derivatives, state, t)
	print("Done")

	ths = solution[:, 0]
	xs = solution[:, 2]

	pxs = L * sin(ths) + xs
	pys = L * cos(ths)

	fig = pp.figure()
	ax = fig.add_subplot(111, autoscale_on=False, ylim=(-1.5*L, 1.5*L), xlim=(-10*L, 10*L))
	#ax.set_aspect('equal')
	ax.grid()

	patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

	line, = ax.plot([], [], 'o-', lw=2)
	time_template = 'time = %.1fs'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

	cart_width = L/10.
	cart_height = L/20.

	def init():
		line.set_data([], [])
		time_text.set_text('')
		patch.set_xy((-cart_width/2, -cart_height/2))
		patch.set_width(cart_width)
		patch.set_height(cart_height)
		return line, time_text, patch

	def animate(i):
		thisx = [xs[i], pxs[i]]
		thisy = [0, pys[i]]

		line.set_data(thisx, thisy)
		time_text.set_text(time_template % (i*dt))
		patch.set_x(xs[i] - cart_width/2)
		return line, time_text, patch

	ani = animation.FuncAnimation(fig, animate, np.arange(1, len(solution)),interval=25, blit=True, init_func=init)

	print("Writing video...")
	Writer = animation.ImageMagickFileWriter
	writer = Writer(fps=25, metadata=dict(artist='Sergey Royz'), bitrate=1800)
	ani.save('controlled-cart.mp4', writer=writer)
	#pp.show()


def outp():
	vidplay = TkinterVideo(root, scaled=True)
	vidplay.load(r"controlled-cart.mp4")
	vidplay.grid(column=3, row=0, rowspan=5, sticky="nsew")
	vidplay.play() 

	def play_pause():
		if vidplay.is_paused():
			vidplay.play()
			play_pause_btn["text"] = "Pause"

		else:
			vidplay.pause()
			play_pause_btn["text"] = "Play"

	def seek(value):
		vidplay.seek(int(value))

	def update_duration(event):
		duration = vidplay.video_info()["duration"]
		end_time["text"] = str(datetime.timedelta(seconds=duration))
		progress_slider["to"] = duration

	def update_scale(event):
		progress_value.set(vidplay.current_duration())

	def video_ended(event):
		progress_slider.set(progress_slider["to"])
		play_pause_btn["text"] = "Play"
		progress_slider.set(0)

	play_pause_btn = tk.Button(root, text="Play", command=play_pause)
	play_pause_btn.grid(column=3, row=6 )

	start_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
	start_time.grid(column=2, row=5)

	progress_value = tk.IntVar(root)

	progress_slider = tk.Scale(root, variable=progress_value, from_=0, to=0, orient="horizontal", command=seek)
	# progress_slider.bind("<ButtonRelease-1>", seek)
	progress_slider.grid(column=3, row=5, sticky="ew")

	end_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
	end_time.grid(column=4, row=5)

	vidplay.bind("<<Duration>>", update_duration)
	vidplay.bind("<<SecondChanged>>", update_scale)
	vidplay.bind("<<Ended>>", video_ended)

	clicken = tk.Button(root, text="EXIT", padx=20, pady=5,borderwidth=2, command=lambda: root.quit() )

	#clickex.grid(column=4, row=8,columnspan=3,padx=2, pady=2)
	clicken.grid(column=3, row=7)


root.mainloop()
