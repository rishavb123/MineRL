from smoothing import running_average, savgol_filter
import matplotlib.pyplot as plt
import json

metric_file = "./metrics/zombie_fight_2_1619572825.json"

with open(metric_file) as json_file:
    metrics = json.load(json_file)

fig, axs = plt.subplots(1, 3)

for key, ax in zip(metrics, axs):
    ax.set_title(key + " vs iterations")
    ax.set_xlabel("iterations")
    ax.set_ylabel(key)
    ax.plot(metrics[key], label="data")
    ax.plot(savgol_filter(metrics[key], len(metrics[key]) / 2, 4), label="savgol_filter")
    ax.plot(running_average(metrics[key]), label="running_average", )
    ax.legend()

plt.show()
