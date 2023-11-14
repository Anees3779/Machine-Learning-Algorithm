import numpy as np


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroid = None

    @staticmethod
    def euc(data_points, centroid):
        return np.sqrt(np.sum((centroid - data_points) ** 2, axis=1))

    def fit(self, X, max_iter=200):
        self.centroid = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iter):
            y = []

            for data_points in X:
                distances = KMeansClustering.euc(data_points, self.centroid)
                centroid_num = np.argmin(distances)
                y.append(centroid_num)

            y = np.array(y)

            centroid_indices = []
            for i in range(self.k):
                centroid_indices.append(np.argwhere(y == i))

            centroid_centers = []

            for i, indices in enumerate(centroid_indices):
                if len(indices) == 0:
                    centroid_centers.append(self.centroid[i])
                else:
                    centroid_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroid - np.array(centroid_centers)) < 0.0001:
                break
            else:
                self.centroid = np.array(centroid_centers)
        return y
