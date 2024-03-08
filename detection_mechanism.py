import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import pprint

from PIL import Image

STEP_SIZE = 0.5
TARGET_DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
pp = pprint.PrettyPrinter(indent=4)
MAX_REGION_COUNT_BENIGN = 1


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def count_white_regions(image, white_threshold):
    rows, cols = image.shape
    uf = UnionFind(rows * cols)

    for i in range(rows):
        for j in range(cols):
            if image[i, j] >= white_threshold:
                if i > 0 and image[i - 1, j] >= white_threshold:
                    uf.union(i * cols + j, (i - 1) * cols + j)
                if j > 0 and image[i, j - 1] >= white_threshold:
                    uf.union(i * cols + j, i * cols + (j - 1))

    regions = set()
    for i in range(rows):
        for j in range(cols):
            if image[i, j] >= white_threshold:
                regions.add(uf.find(i * cols + j))

    return len(regions)


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
    plt.plot(sorted_combined_curve[:, 0], sorted_combined_curve[:, 1], linestyle='--', color='b', marker='o',
             label=f'Combined ROC Curve (AUC = {roc_auc_score(ground_truth_labels, predicted_labels[0]):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


threshold_values = list()
for i in range(9):
    threshold_values.append(255 * 0.1 * (i + 1))

all_images = dict()
for target_digit in TARGET_DIGITS:
    all_images_target = list()
    files = os.listdir(os.path.join('adversarial_digits_{}'.format(STEP_SIZE), str(target_digit)))
    for file in sorted(files):
        print("File {}".format(file))
        image = Image.open(os.path.join(os.path.join('adversarial_digits_{}'.format(STEP_SIZE),
                                                     str(target_digit)), file))
        np_arr_img = np.array(image)
        all_images_target.append(np_arr_img)
    all_images[target_digit] = all_images_target

threshold_dicts = dict()

for particular_threshold in threshold_values:
    num_regions_curr_threshold = dict()

    for k, v in all_images.items():
        count_list = list()
        for im in v:
            count = count_white_regions(im, particular_threshold)
            count_list.append(count)
        num_regions_curr_threshold[k] = count_list

    print("Threshold {}".format(particular_threshold))
    pp.pprint(num_regions_curr_threshold)
    print("\n\n")
    threshold_dicts[particular_threshold] = num_regions_curr_threshold

converted_threshold_list_all = list()
for particular_threshold in sorted(threshold_dicts.keys()):
    c_p = list()
    for digit in sorted(threshold_dicts[particular_threshold].keys()):
        c_p = c_p + [int(i > MAX_REGION_COUNT_BENIGN) for i in threshold_dicts[particular_threshold][digit]]
    converted_threshold_list_all.append(c_p)

print(threshold_values)
pp.pprint(converted_threshold_list_all)

gt_labels_all = dict()
for target_digit in TARGET_DIGITS:
    label_filename = 'adversarial_labels_{}'.format(STEP_SIZE) + "/" + str(target_digit) + ".txt"
    if os.path.exists(label_filename):
        gt_labels = []
        with open(label_filename, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if "im" in line:
                    gt_labels.append(int(line.split("\n")[0].split(" ")[1].strip()))

        gt_labels_all[target_digit] = gt_labels

gt_labels_all_list = list()
for digit in sorted(gt_labels_all.keys()):
    gt_labels_all_list = gt_labels_all_list + gt_labels_all[digit]
print(gt_labels_all_list)


plot_single_roc_curve(threshold_values, np.array(converted_threshold_list_all),
                      np.array(gt_labels_all_list))
#
# # Example usage:
# # Assuming mnist_image is your 32x32 MNIST image as a 2D NumPy array
# # And threshold is the threshold value between 0 and 255
# mnist_image = np.random.randint(0, 256, size=(32, 32))
# threshold = 127  # Example threshold value
#
# # Count the number of white regions
# white_regions_count = count_white_regions(mnist_image, threshold)
#
# print("Number of white regions:", white_regions_count)
