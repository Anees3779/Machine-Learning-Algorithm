import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import queue

# Define label for different point groups
NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2


# Function to find all neighbor points in radius
def neighbor_points(data, pointId, radius):
    points = []
    for i in range(len(data)):
        # Euclidean distance using L2 Norm
        if np.linalg.norm(data[i] - data[pointId]) <= radius:
            points.append(i)
    return points


# DB Scan algorithm
def dbscan(data, Eps, MinPt):
    # Initialize all point labels to unassigned
    pointlabel = [UNASSIGNED] * len(data)
    pointcount = []
    # Initialize lists for core/non-core points
    corepoint = []
    noncore = []

    # Find all neighbors for all points
    for i in range(len(data)):
        pointcount.append(neighbor_points(data, i, Eps))

    # Find all core points, edge points, and noise
    for i in range(len(pointcount)):
        if len(pointcount[i]) >= MinPt:
            pointlabel[i] = core
            corepoint.append(i)
        else:
            noncore.append(i)

    for i in noncore:
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i] = edge
                break

    # Start assigning points to clusters
    cl = 1
    # Using a Queue to put all neighbor core points in the queue and find neighbor's neighbor
    for i in range(len(pointlabel)):
        q = queue.Queue()
        if pointlabel[i] == core:
            pointlabel[i] = cl
            for x in pointcount[i]:
                if pointlabel[x] == core:
                    q.put(x)
                    pointlabel[x] = cl
                elif pointlabel[x] == edge:
                    pointlabel[x] = cl
            # Stop when all points in the Queue have been checked
            while not q.empty():
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if pointlabel[y] == core:
                        pointlabel[y] = cl
                        q.put(y)
                    if pointlabel[y] == edge:
                        pointlabel[y] = cl
            cl += 1  # Move to the next cluster

    return pointlabel, cl


# Function to plot the final result
def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if i == 0:
            # Plot all noise points as blue
            color = 'blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = [];
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')


# Load Data
df = pd.read_csv("Mall_Customers.csv")
train = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Set EPS and Minpoint
epss = [5, 10]
minptss = [5, 10]

# Find all clusters, outliers in different settings and print results
for eps in epss:
    for minpts in minptss:
        print('Set eps = ' + str(eps) + ', Minpoints = ' + str(minpts))
        pointlabel, cl = dbscan(train, eps, minpts)
        plotRes(train, pointlabel, cl)
        plt.show()
        print('number of clusters found: ' + str(cl - 1))
        counter = collections.Counter(pointlabel)
        print(counter)
        outliers = pointlabel.count(0)
        print('number of outliers found: ' + str(outliers) + '\n')
