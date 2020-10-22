from tools import load_episodes
import matplotlib.pyplot as plt

n_episodes = 100
datadir = "/home/luigi/Development/dreamer/logs/racecar_2020_10_21_12_27_38/episodes"
dataset = load_episodes(datadir, n_episodes)

episode = next(dataset)
while len(episode) > 10:
    episode = next(dataset)

print("[Info] Reward: {}".format(episode["reward"]))
print("[Info] Time: {}".format(episode["time"]))
print("[Info] X: {}".format(episode["gps"][:, 0]))
print("[Info] Y: {}".format(episode["gps"][:, 1]))

plt.plot(episode["gps"][:, 0], episode["gps"][:, 1], label="X-Y Trajectory")
plt.show()