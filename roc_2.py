import numpy as np
import matplotlib.pyplot as plt

# Assuming TPR values for different threshold values
thresholds = np.linspace(0, 1, 100)
tpr = thresholds  # Assuming perfect TPR values
print(thresholds)
print(tpr)
# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
plt.plot(thresholds, tpr, linestyle='--', color='b',  label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
