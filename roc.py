import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def calculate_fpr_tpr(predicted_labels, ground_truth_labels):
    true_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
    true_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))

    print(true_positives)

    tpr = true_positives / (true_positives + false_negatives)
    print(tpr)
    fpr = false_positives / (false_positives + true_negatives)
    print(fpr)

    return fpr, tpr


def plot_single_roc_curve(thresholds, predicted_labels, ground_truth_labels):
    # Initialize arrays to store FPR and TPR values
    all_fpr = []
    all_tpr = []

    # Calculate FPR and TPR for each threshold
    for i, threshold in enumerate(thresholds):
        fpr, tpr = calculate_fpr_tpr(predicted_labels[i], ground_truth_labels)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # Combine FPR and TPR arrays
    combined_curve = np.column_stack((all_fpr, all_tpr))

    # Sort the combined curve by FPR
    sorted_combined_curve = combined_curve[np.argsort(combined_curve[:, 0])]
    print(sorted_combined_curve)

    # Plot the single ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(sorted_combined_curve[:, 0], sorted_combined_curve[:, 1], linestyle='--', color='b',
             label=f'Combined ROC Curve (AUC = {roc_auc_score(ground_truth_labels, predicted_labels[0]):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# Example usage:
thresholds = [0.2, 0.4, 0.6, 0.8]
predicted_labels_list = [
    np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]),  # For threshold 0.2
    np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]),  # For threshold 0.4
    np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]),  # For threshold 0.6
    np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])  # For threshold 0.8
]
ground_truth_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])

plot_single_roc_curve(thresholds, predicted_labels_list, ground_truth_labels)
