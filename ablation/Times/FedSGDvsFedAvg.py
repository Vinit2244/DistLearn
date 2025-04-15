import matplotlib.pyplot as plt
import numpy as np

models = ['DiabetesMLP', 'MNISTMLP', 'FashionMNISTCNN']

fedsgd_times = [9.05, 497, 956]
fedavg_times = [11.50, 1342.55, 3271.16]

labels = ['FedSGD', 'FedAvg']

y_limits = [
    max([t for t in [fedsgd_times[0], fedavg_times[0]] if t is not None]) + 5,
    max([t for t in [fedsgd_times[1], fedavg_times[1]] if t is not None]) + 100,
    max([t for t in [fedsgd_times[2], fedavg_times[2]] if t is not None]) + 200,
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Training Time Comparison (3 Epochs for FedAvg)', fontsize=14)

for i, ax in enumerate(axes):
    times = [fedsgd_times[i] if fedsgd_times[i] is not None else 0, fedavg_times[i]]
    bars = ax.bar(labels, times, color=['skyblue', 'salmon'])

    ax.set_title(models[i])
    ax.set_ylabel('Time (s)')
    ax.set_ylim(0, y_limits[i])

    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('FedSGDvsFedAvg/time_FedSGD_vs_FedAvg.png', dpi=300)
