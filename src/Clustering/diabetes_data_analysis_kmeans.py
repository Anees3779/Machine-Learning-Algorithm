import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv(r'C:\Users\Anees Meer\diabetes.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.corr())
# Calculating the number of zero-values in in each column
print('rows having null Pregnancies value : {0}'.format(len(df.loc[df['Pregnancies'] == 0])))
print('rows having null Glucose value : {0}'.format(len(df.loc[df['Glucose'] == 0])))
print('rows having null BloodPressure value : {0}'.format(len(df.loc[df['BloodPressure'] == 0])))
print('rows having null SkinThickness value : {0}'.format(len(df.loc[df['SkinThickness'] == 0])))
print('rows having null Insulin value : {0}'.format(len(df.loc[df['Insulin'] == 0])))
print('rows having null BMI value : {0}'.format(len(df.loc[df['BMI'] == 0])))
print('rows having null DiabetesPedigreeFunction value : {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print('rows having null Age value : {0}'.format(len(df.loc[df['Age'] == 0])))
# Calculating the mean value of every column
means = df.iloc[:, 1:6].mean()
print(means)
nonzeros = list(df.columns[1:6])
for column in nonzeros:
    df[column] = df[column].replace(0, means[column])
# Preview of Changed Dataset
print(df.head())
outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)
df_improve =df[['Glucose','BMI','Age','Insulin','Outcome']]
print(df_improve.head())
print(df_improve.corr())
# Visualisation
# Heatmap
cor = df_improve.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()
# Pairplot
sns.pairplot(df_improve,hue='Outcome')
# Histogram
features = df_improve.columns.tolist()
features.remove('Outcome')
# Create histograms for each feature with 'Outcome' as the hue
for feature in features:
    sns.histplot(data=df_improve, x=feature, hue='Outcome', kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()
# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Fit', 'Diabetic'], autopct='%1.1f%%', startangle=140)
plt.title('Outcome Count by Fit and Diabetic')
plt.show()
# Plot the count of each class in the 'Outcome' column
sns.countplot(x=df_improve['Outcome'], data=df_improve)
plt.show()
X = df_improve.loc[:, ['Glucose', 'Insulin','BMI','Age']].to_numpy()
y = df_improve.loc[:, 'Outcome'].to_numpy()
#Standardising the Dataset
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
target = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# K-means Clustering from scratch
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster
        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx


    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            points = self.X[index]
            ax.scatter(points[:, 0], points[:, 1])

        centroids = np.array(self.centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", color="black", linewidth=2)

        plt.show()

# Create an instance of your KMeans class
kmeans = KMeans(K=2, max_iters=150, plot_steps=True)

# Fit the K-Means model to the training data (this is your clustering process)
kmeans_labels = kmeans.predict(X_train)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_train[kmeans_labels == 0, 0], X_train[kmeans_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_train[kmeans_labels == 1, 0], X_train[kmeans_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(kmeans.centroids[0][0], kmeans.centroids[0][1], s=300, c='black', marker='X', label='Centroid 1')
plt.scatter(kmeans.centroids[1][0], kmeans.centroids[1][1], s=300, c='black', marker='X', label='Centroid 2')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()