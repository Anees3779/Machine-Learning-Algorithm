import pandas as pd
from src.Clustering.utils.Mall_data_exploration import DataExploration
from src.Clustering.Kmeans.Kmeans import KMeansClustering
import matplotlib.pyplot as plt

# File path to the dataset
file_path = "Mall_Customers.csv"

# Create an instance of DataExploration
data_explore = DataExploration(file_path)

# Visualize data
data_explore.visualize_data()

# Load data for KMeans clustering
df_improve = data_explore.df_improve
data_for_clustering = df_improve[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Create an instance of KMeansClustering
kmeans = KMeansClustering(k=3)

# Fit and predict
labels = kmeans.fit(data_for_clustering)

# Plot KMeans clustering result
plt.scatter(data_for_clustering[:, 0], data_for_clustering[:, 1], c=labels)
plt.scatter(kmeans.centroid[:, 0], kmeans.centroid[:, 1], c=range(len(kmeans.centroid)), marker="*", s=200)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('KMeans Clustering Result')
plt.show()

# Plot KMeans clustering result
plt.scatter(data_for_clustering[:, 1], data_for_clustering[:, 2], c=labels)
plt.scatter(kmeans.centroid[:, 1], kmeans.centroid[:, 2], c=range(len(kmeans.centroid)), marker="*", s=200)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering Result')
plt.show()

# Plot KMeans clustering result
plt.scatter(data_for_clustering[:, 2], data_for_clustering[:, 0], c=labels)
plt.scatter(kmeans.centroid[:, 2], kmeans.centroid[:, 0], c=range(len(kmeans.centroid)), marker="*", s=200)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering Result')
plt.show()

