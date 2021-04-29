from rerun import get_last_stamp, get_stamps
from smoothing import running_average, savgol_filter

import matplotlib
import matplotlib.pyplot as plt
import json

stamps = get_stamps()
# stamp = get_last_stamp()
stamp = stamps[1]
metric_file = f"./metrics/zombie_fight_2_{stamp}.json"

with open(metric_file) as json_file:
    metrics = json.load(json_file)

fig, axs = plt.subplots(1, 3)

for key, ax in zip(metrics, axs):
    metrics[key] = metrics[key][:1000]
    ax.set_title(key + " vs episodes")
    ax.set_xlabel("episodes")
    ax.set_ylabel(key)
    ax.plot(metrics[key], label="data")
    ax.plot(savgol_filter(metrics[key], len(metrics[key]) / 2, 4), label="savgol_filter")
    ax.plot(running_average(metrics[key]), label="running_average", )
    ax.legend()

fig.set_size_inches(24.5, 13)
plt.savefig(f"./graphs/zombie_fight_2_{stamp}.png")
plt.show()
