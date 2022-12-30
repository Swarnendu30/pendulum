import gym
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import numpy as np
import time

env = gym.envs.make("CartPole-v1")

def plot_res(values, title=''):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def q_learning(env, model, episodes, gamma=0.9,epsilon=0.3, eps_decay=0.99,replay=False, replay_size=20,
               title='DQL', double=False,n_update=10, soft=False, verbose=True):
    
    final = []
    memory = []
    episode_i = 0
    sum_total_replay_time = 0
    for episode in range(episodes):
        episode_i += 1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()

        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()

            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                break

            if replay:
                t0 = time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1 = time.time()
                sum_total_replay_time += (t1-t0)
            else:
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * \
                    torch.max(q_values_next).item()
                model.update(state, q_values)

            state = next_state

        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        if(episode_i==150):
            plot_res(final, title)
            env.render()

        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)

    return final

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
episodes = 150
n_hidden = 50
lr = 0.001
simple_dqn = DQN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3)
