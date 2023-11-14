# Your main script
from src.Clustering.utils.Mall_data_exploration import DataExploration
from src.Clustering.Meanshift.meanshift import MeanShiftClustering
import matplotlib.pyplot as plt

# Data Exploration
data_explorer = DataExploration(r"C:\Users\Anees Meer\Downloads\Mall_Customers.csv")
data_explorer.explore_data()
data_explorer.visualize_data()

# Mean Shift Clustering
X = data_explorer.df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].to_numpy()

mean_shift_model = MeanShiftClustering(bandwidth=1, n_iter=10)
mean_shift_model.fit(X)
clusters = mean_shift_model.predict(X)

# Plotting Mean Shift Clusters
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Mean Shift Clustering")
ax.scatter(X[:, 0], X[:, 1], c=clusters, s=50, cmap='inferno')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Mean shift Clustering Result')

plt.show()

# Plotting Mean Shift Clusters
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Mean Shift Clustering")
ax.scatter(X[:, 0], X[:, 2], c=clusters, s=50, cmap='inferno')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

plt.show()

# Plotting Mean Shift Clusters
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Mean Shift Clustering")
ax.scatter(X[:, 1], X[:, 2], c=clusters, s=50, cmap='inferno')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
