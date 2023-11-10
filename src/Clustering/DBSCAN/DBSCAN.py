import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import queue

class DBSCAN:
    def __init__(self, Eps, MinPt):
        self.Eps = Eps
        self.MinPt = MinPt
        self.NOISE = 0
        self.UNASSIGNED = 0
        self.core = -1
        self.edge = -2
        self.data = None  # To store the loaded data
        self.labels = None  # To store the cluster labels
        self.num_clusters = None  # To store the number of clusters

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data

    def select_features(self, data, feature_cols):
        if data is not None:
            return data[feature_cols].values
        else:
            raise ValueError("Data not loaded. Use load_data() first.")

    def neighbor_points(self, data, pointId):
        points = []
        for i in range(len(data)):
            if np.linalg.norm(data[i] - data[pointId]) <= self.Eps:
                points.append(i)
        return points

    def fit_predict(self,data):
        if data is not None:
            pointlabel = [self.UNASSIGNED] * len(data)
            pointcount = []
            corepoint = []
            noncore = []

            for i in range(len(data)):
                pointcount.append(self.neighbor_points(data,i))

            for i in range(len(pointcount)):
                if len(pointcount[i]) >= self.MinPt:
                    pointlabel[i] = self.core
                    corepoint.append(i)
                else:
                    noncore.append(i)

            for i in noncore:
                for j in pointcount[i]:
                    if j in corepoint:
                        pointlabel[i] = self.edge
                        break

            cl = 1
            for i in range(len(pointlabel)):
                q = queue.Queue()
                if pointlabel[i] == self.core:
                    pointlabel[i] = cl
                    for x in pointcount[i]:
                        if pointlabel[x] == self.core:
                            q.put(x)
                            pointlabel[x] = cl
                        elif pointlabel[x] == self.edge:
                            pointlabel[x] = cl
                    while not q.empty():
                        neighbors = pointcount[q.get()]
                        for y in neighbors:
                            if pointlabel[y] == self.core:
                                pointlabel[y] = cl
                                q.put(y)
                            if pointlabel[y] == self.edge:
                                pointlabel[y] = cl
                    cl += 1

            self.labels = pointlabel
            self.num_clusters = cl - 1
        else:
            raise ValueError("Data not loaded. Use load_data() first.")

    def plot_result(self, data, feature_cols):
        if data is not None and self.labels is not None:
            nPoints = len(data)
            scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
            for i in range(self.num_clusters):
                if i == 0:
                    color = 'blue'
                else:
                    color = scatterColors[i % len(scatterColors)]
                x1 = []; y1 = []
                for j in range(nPoints):
                    if self.labels[j] == i:
                        x1.append(data.iloc[j][feature_cols[0]])
                        y1.append(data.iloc[j][feature_cols[1]])
                plt.scatter(x1, y1, c=color, alpha=1, marker='.')

            plt.title(f'DBSCAN Clustering (Eps={self.Eps}, MinPts={self.MinPt})')
            plt.xlabel(feature_cols[0])
            plt.ylabel(feature_cols[1])
            plt.show()
        else:
            raise ValueError("Data or cluster labels not available. Use load_data() and fit_predict() first.")

