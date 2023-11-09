import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.clustering.KMeans import KMeans
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r'C:\Users\Anees Meer\diabetes.csv')

# Exploratory Data Analysis (EDA)

# Check for null values
print(df.isnull().sum())

# Replace zero values with NaN for relevant columns and calculate mean
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Select features
df_sel = df[['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']

# Standardize the data
scaler = StandardScaler()
X = df_sel.iloc[:, :-1].values
y = df_sel['Outcome'].values
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the number of clusters (K)
clusters = 2

# Create an instance of your KMeans class
kmeans = KMeans(K=clusters, max_iters=150, plot_steps=True)

# Fit the K-Means model to the training data
kmeans_labels = kmeans.predict(X_train)

# Calculate cluster centers
cluster_centers = kmeans.centroids

# Assign cluster labels to the test data
kmeans_labels_test = kmeans.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, kmeans_labels_test)
precision = precision_score(y_test, kmeans_labels_test)
recall = recall_score(y_test, kmeans_labels_test)
f1 = f1_score(y_test, kmeans_labels_test)
conf_matrix = confusion_matrix(y_test, kmeans_labels_test)
classification_rep = classification_report(y_test, kmeans_labels_test)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Plot K-Means graph
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=kmeans_labels_test, cmap='viridis', marker='o', label='Cluster')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', label='Cluster Centers', marker='x')
plt.title('K-Means Clustering')
plt.xlabel('Glucose')
plt.ylabel('Insulin')
plt.legend()
plt.show()
