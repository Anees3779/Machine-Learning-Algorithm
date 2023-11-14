import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
# Visualisation
#   Pair plot
sns.pairplot(df_improve, hue='Gender')
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
plt.pie(Gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()
# Plot the count of each class in the 'Outcome' column
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
# Mean Shift Clustering from Scratch
X = df_improve.drop(columns=['Gender', 'Age'], axis=1)
print(X)
# Standardising the Dataset
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
bandwidth = 0.85
n_iter = 10
centroids = dict(zip(range(X.shape[0]), np.array([X[i, :] for i in range(X.shape[0])])))
for n in range(n_iter):
    new_centroids = centroids.copy()
    updated_sample = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        neighbor = []
        index = []
        for j in range(X.shape[0]):
            sample = X[j, :]
            if np.linalg.norm(sample - centroid) < bandwidth:
                neighbor.append(sample)
                index.append(j)
        new_centroid = np.average(neighbor, axis=0)
        for k in range(len(neighbor)):
            new_centroids[index[k]] = new_centroid
    centroids = new_centroids
final_centroids = list(new_centroids.values())
unique_centroids = np.unique(final_centroids, axis=0)
clusters = dict(zip(range(len(unique_centroids)), np.array([unique_centroids[i, :] for i in range(unique_centroids.shape[0])])))
getcluster = lambda val: [k for k, v in clusters.items() if (v == val).sum() == len(val)][0]
df_cluster = pd.DataFrame(columns=['Sample_data', 'Centroids', 'clusters'])
df_cluster['Sample_data'] = list(X)
df_cluster['Centroids'] = list(final_centroids)
df_cluster['clusters'] = [getcluster(final_centroids[i]) for i in range(len(final_centroids))]
df_cluster
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Actual Data showing clusters")
ax.scatter(X[:, 0], X[:, 1], c=df_cluster['clusters'].values, s=50, cmap='inferno')
plt.show()