import matplotlib.pyplot as plt

features = [
    "Typing Speed",
    "Inter-Key Delay",
    "Key Hold Time",
    "Pause Duration",
    "Error Rate",
    "Backspace Count"
]

importance = [0.12, 0.18, 0.10, 0.22, 0.25, 0.13]

plt.figure()
plt.bar(features, importance)
plt.title("Feature Importance for Cognitive Load Estimation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
