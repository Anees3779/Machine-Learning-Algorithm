import matplotlib.pyplot as plt
from src.Clustering.DBSCAN.DBSCAN import DBSCAN
import pandas as pd

file_path = r"C:\Users\Anees Meer\Downloads\Mall_Customers.csv"
feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']

# Create an instance of DBSCAN
dbscan_model = DBSCAN(Eps=5, MinPt=5)

# Load data
data=dbscan_model.load_data(file_path)

# Select features
data_features = dbscan_model.select_features(data, feature_cols)

# Fit and predict
dbscan_model.fit_predict(data_features)

# Plot the results
dbscan_model.plot_result(data, feature_cols)
plt.show()
