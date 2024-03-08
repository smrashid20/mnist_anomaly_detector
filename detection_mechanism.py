import numpy as np


TARGET_DIGITS = [2]


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
            if image[i, j] > white_threshold:
                if i > 0 and image[i - 1, j] > white_threshold:
                    uf.union(i * cols + j, (i - 1) * cols + j)
                if j > 0 and image[i, j - 1] > white_threshold:
                    uf.union(i * cols + j, i * cols + (j - 1))

    regions = set()
    for i in range(rows):
        for j in range(cols):
            if image[i, j] > white_threshold:
                regions.add(uf.find(i * cols + j))

    return len(regions)


threshold_values = list()
for i in range(10):
    threshold_values.append(255 * 0.1 * (i+1))


# Example usage:
# Assuming mnist_image is your 32x32 MNIST image as a 2D NumPy array
# And threshold is the threshold value between 0 and 255
mnist_image = np.random.randint(0, 256, size=(32, 32))
threshold = 127  # Example threshold value

# Count the number of white regions
white_regions_count = count_white_regions(mnist_image, threshold)

print("Number of white regions:", white_regions_count)
