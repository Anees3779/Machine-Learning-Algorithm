#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ## Load Dataset

# In[ ]:


df = pd.read_csv('diabetes.csv')

# ## Exploratry Data Analysis

# In[ ]:


# Data preview
df.head()

# In[ ]:


# for getting the shape (No of rows & columns)
df.shape

# In[ ]:


# Print Column name
df.columns

# In[ ]:


# getting brief summary of data
df.info()

# In[ ]:


# Checking Null Values
df.isnull().sum()

# In[ ]:


# Statistical summary of dataset
df.describe()

# In[ ]:


# Correlation Between Every Features of dataset
df.corr()

# In[ ]:


# Calculating the number of zero-values in in each column
print('rows having null Pregnancies value : {0}'.format(len(df.loc[df['Pregnancies'] == 0])))
print('rows having null Glucose value : {0}'.format(len(df.loc[df['Glucose'] == 0])))
print('rows having null BloodPressure value : {0}'.format(len(df.loc[df['BloodPressure'] == 0])))
print('rows having null SkinThickness value : {0}'.format(len(df.loc[df['SkinThickness'] == 0])))
print('rows having null Insulin value : {0}'.format(len(df.loc[df['Insulin'] == 0])))
print('rows having null BMI value : {0}'.format(len(df.loc[df['BMI'] == 0])))
print('rows having null DiabetesPedigreeFunction value : {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print('rows having null Age value : {0}'.format(len(df.loc[df['Age'] == 0])))

# In[ ]:


# Calculating the mean value of every column
means = df.iloc[:, 1:6].mean()
means

# In[ ]:


# Replacing the null Values by the mean of their column
nonzeros = list(df.columns[1:6])
for column in nonzeros:
    df[column] = df[column].replace(0, means[column])

# In[ ]:


# Preview of Changed Dataset
df.head()

# In[ ]:


# Count the number of occurrences of each unique value in the 'Outcome' column
outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)

# In[ ]:


# Selecting useful features from dataset
df_sel = df.loc[:, ['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']]

# In[ ]:


# Preview of selected features
df_sel.head()

# In[ ]:


# Calculating the min value of every featured column
df.min()

# ## Data Visualisation

# In[ ]:


# Heatmap
cor = df_sel.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()

# In[ ]:


# Pairplot
sns.pairplot(df_sel, hue='Outcome')

# In[ ]:


# Histogram
features = df_sel.columns.tolist()
features.remove('Outcome')
# Create histograms for each feature with 'Outcome' as the hue
for feature in features:
    sns.histplot(data=df_sel, x=feature, hue='Outcome', kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()

# In[ ]:


# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Fit', 'Diabetic'], autopct='%1.1f%%', startangle=140)
plt.title('Outcome Count by Fit and Diabetic')
plt.show()

# In[ ]:


# Plot the count of each class in the 'Outcome' column
sns.countplot(x=df_sel['Outcome'], data=df_sel)
plt.show()

# ## Model Building

# In[ ]:


# Selecting feature and target value from dataset
X = df_sel.loc[:, ['Glucose', 'Insulin', 'BMI', 'Age']].to_numpy()
y = df_sel.loc[:, 'Outcome'].to_numpy()

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# In[ ]:


df.shape

# In[ ]:


X_train.shape

# In[ ]:


X_test.shape

# In[ ]:


from sklearn.preprocessing import StandardScaler

fig, ax = plt.subplots(2, 4, figsize=[14, 6])

scaler = StandardScaler()
features = df.iloc[:, 0:8]
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=df.columns[0:8])
scaled_features_df.hist(ax=ax);

# In[ ]:


# Standardisation
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# In[ ]:


X_test = scaler.transform(X_test)

# ## K-Nearest Neighbors (KNN) Algorithm from Scratch

# In[ ]:


from collections import Counter


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KNN:
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_labels).most_common()
        return most_common[0][0]


# In[ ]:


# introdusing Classifier as clf
clf = KNN(k=15)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# In[ ]:


# Function for Calculating Accuracy
def calculate_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[ ]:


calculate_accuracy(y_test, predictions)

# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

# In[ ]:


print(cm)

# In[ ]:


import seaborn as sns

# Change figure size and increase dpi for better resolution
plt.figure(figsize=(6, 4))

ax = sns.heatmap(cm, annot=True, fmt='d')

# set x-axis label and ticks.
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Diabetes k-NN", fontsize=14, pad=20);

# In[ ]:


from sklearn.metrics import precision_score, recall_score

print(f"scikit-learn's Precision: {precision_score(y_test, predictions)}")
print(f"scikit-learn's Recall: {recall_score(y_test, predictions)}")

# In[ ]:


# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# In[ ]:




