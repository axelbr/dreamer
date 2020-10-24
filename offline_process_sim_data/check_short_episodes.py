from tools import load_episodes
import matplotlib.pyplot as plt
import numpy as np

n_episodes = 100
datadir = "/home/luigi/Development/dreamer/logs/racecar_2020_10_22_10_22_34/episodes"
dataset = load_episodes(datadir, n_episodes)

episode = next(dataset)
while len(episode["time"]) < 10:
    episode = next(dataset)

reward = episode["reward"]
time = episode["time"]
x = episode["gps"][:, 0]
y = episode["gps"][:, 1]

print("[Info] Reward: {}".format(reward))
print("[Info] Time: {}".format(time))
print("[Info] X: {}".format(x))
print("[Info] Y: {}".format(y))

x_p = [x[i] for i in (np.argwhere(reward>0))]
y_p = [y[i] for i in (np.argwhere(reward>0))]
x_n = [x[i] for i in (np.argwhere(reward<0))]
y_n = [y[i] for i in (np.argwhere(reward<0))]
plt.plot(x, y, label="X-Y Trajectory")
plt.scatter(x_p, y_p, label="R=+1", c='g')
plt.scatter(x_n, y_n, label="R=-1", c='r')
plt.show()