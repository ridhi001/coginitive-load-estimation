import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [0.86, 0.85, 0.84, 0.845]

plt.figure()
plt.bar(metrics, values)
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_performance.png")
plt.show()
