import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from JSON file
with open('fedmodcs.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
model_names = [entry["model_name"] for entry in data]
times = [entry["time"] for entry in data]

# Plot bar graph
plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, times, color=["skyblue", "orange", "green"])
plt.title("Time Taken by Each Model")
plt.xlabel("Model Name")
plt.ylabel("Time (seconds)")

# Add time labels on top of bars
for bar, time in zip(bars, times):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("fedmodcs_time_plot.png")

