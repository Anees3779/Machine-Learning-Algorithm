import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes.csv')
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
# Mean Shift Clustering from Scratch
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Actual Data showing 2 clusters")
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='inferno')
plt.show()
# Parameter for Mean shift clustering
bandwidth = 3
n_iter = 10
# Cluster centroids initialisation
centroids = dict(zip(range(X_train.shape[0]),np.array([X_train[i,:] for i in range(X_train.shape[0])])))
# iteration of mean shift clustering
for n in range(n_iter):
    new_centroids = centroids.copy()
    updated_sample = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        neighbor = []
        index = []
        for j in range(X_train.shape[0]):
            sample = X_train[j,:]
            if np.linalg.norm(sample-centroid)<bandwidth:
                neighbor.append(sample)
                index.append(j)
        new_centroid = np.average(neighbor,axis= 0)
        for k in range(len(neighbor)):
            new_centroids[index[k]] = new_centroid
    centroids = new_centroids
# Cluster centroids
final_centroids =list(new_centroids.values())
unique_centroids = np.unique(final_centroids,axis=0)
clusters = dict(zip(range(len(unique_centroids)),np.array([unique_centroids[i,:] for i in range(unique_centroids.shape[0])])))
getcluster = lambda val : [k for k,v in clusters.items() if (v == val).sum()==len(val)][0]
df_cluster = pd.DataFrame(columns=['Sample_data','Centroids','clusters'])
df_cluster['Sample_data']=list(X_train)
df_cluster['Centroids']  =list(final_centroids)
df_cluster['clusters']   = [getcluster(final_centroids[i]) for i in range(len(final_centroids))]
print(df_cluster)
# Plotting Final Cluster
fig,ax = plt.subplots(figsize=(10,5))
ax.set_title("Graph Mean Shift Clustering")
ax.scatter(X_train[:,0],X_train[:,1],c=df_cluster['clusters'].values,s=50,cmap = 'inferno')
plt.show()
