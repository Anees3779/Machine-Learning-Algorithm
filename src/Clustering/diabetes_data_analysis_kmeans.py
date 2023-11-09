import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Anees Meer\Downloads\Mall_Customers.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
# Calculating the number of zero-values in each column
print('rows having null CustomerID value : {0}'.format(len(df.loc[df['CustomerID'] == 0])))
print('rows having null Age value : {0}'.format(len(df.loc[df['Age'] == 0])))
print('rows having null Annual Income (k$) value : {0}'.format(len(df.loc[df['Annual Income (k$)'] == 0])))
print('rows having null Spending Score (1-100) value : {0}'.format(len(df.loc[df['Spending Score (1-100)'] == 0])))
Gender_counts = df['Gender'].value_counts()
print(Gender_counts)
df_improve = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print(df_improve.head())
# Visualisation
# Pair plot
sns.pairplot(df_improve, hue='Gender')
# Create histograms for each feature with 'Outcome' as the hue
# Histogram
features = df_improve.columns.tolist()
features.remove('Gender')
# Create histograms for each feature with 'Gender' as the hue
for feature in features:
    sns.histplot(data=df_improve, x=feature, hue='Gender', kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()
# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(Gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()
# Plot the count of each class in the 'Gender' column
sns.countplot(x=df_improve['Gender'], data=df_improve)
plt.show()

plt.figure(1, figsize=(12, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Annual Income (k$)', data=df_improve[df_improve['Gender'] == gender], s=200, alpha=0.5,
                label=gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()

plt.figure(1, figsize=(12, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df_improve[df_improve['Gender'] == gender],
                s=200, alpha=0.5, label=gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income (k$) vs Spending Score (1-100) w.r.t Gender')
plt.legend()
plt.show()

plt.figure(1, figsize=(12, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Spending Score (1-100)', data=df_improve[df_improve['Gender'] == gender], s=200, alpha=0.5,
                label=gender)
plt.xlabel('Age'), plt.ylabel('Spending Score (1-100)')
plt.title('Age vs Spending Score (1-100) w.r.t Gender')
plt.legend()
plt.show()

plt.figure(1, figsize=(12, 5))
n = 0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=cols, y='Gender', data=df_improve, palette='vlag')
    sns.swarmplot(x=cols, y='Gender', data=df_improve)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Boxplot & Swarm-plot' if n == 2 else '')
plt.show()


# K-means Clustering from scratch
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


# Cluster randomly distributed data
np.random.seed(1)
points = np.random.randint(0, 100, (100, 2))

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(points)

plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.scatter(kmeans.centroid[:, 0], kmeans.centroid[:, 1], c=range(len(kmeans.centroid)), marker="*", s=200)
plt.show()
