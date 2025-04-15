import json
import matplotlib.pyplot as plt

with open("metrics.json", "r") as f:
    data = json.load(f)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

colors = ['blue', 'green', 'red', 'orange', 'purple']

for i, run in enumerate(data):
    rounds = list(range(1, len(run["losses"]) + 1))
    label = run["model_name"]

    axs[0].plot(rounds, run["losses"], color=colors[i], label=label)
    
    axs[1].plot(rounds, run["accuracies"], color=colors[i], label=label)

axs[0].set_title("Loss per Round")
axs[0].set_xlabel("Rounds")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

axs[1].set_title("Accuracy per Round")
axs[1].set_xlabel("Rounds")
axs[1].set_ylabel("Accuracy (%)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("metric_plots/metrics_plot.png", dpi=300)
