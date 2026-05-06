import matplotlib.pyplot as plt

import numpy as np

cm = np.array([
    [18, 2, 0],
    [3, 16, 1],
    [1, 2, 17]
])

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
