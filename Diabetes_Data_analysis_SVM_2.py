#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:


# Loading Diabetes Dataset
df = pd.read_csv('diabetes.csv')

# In[ ]:


# Data Preview
df.head()

# ## Exploratory  Data Analysis

# In[ ]:


# for getting the shape (No of rows & columns)
df.shape

# In[ ]:


# Column Details
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


df.corr()

# In[ ]:


cor = df.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()

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


nonzeros = list(df.columns[1:6])
for column in nonzeros:
    df[column] = df[column].replace(0, means[column])

# In[ ]:


# Preview of Changed Dataset
df.head()

# In[ ]:


outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)

# In[ ]:


df_improve = df[['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']]

# In[ ]:


# improved dataset after datacleaning
df_improve.head()

# In[ ]:


df_improve.corr()

# ## Data Visualisation

# In[ ]:


# Heatmap
cor = df_improve.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()

# In[ ]:


# Pairplot
sns.pairplot(df_improve, hue='Outcome')

# In[ ]:


# Histogram
features = df_improve.columns.tolist()
features.remove('Outcome')
# Create histograms for each feature with 'Outcome' as the hue
for feature in features:
    sns.histplot(data=df_improve, x=feature, hue='Outcome', kde=True)
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
sns.countplot(x=df_improve['Outcome'], data=df_improve)
plt.show()

# ## Model Training

# In[ ]:


X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# In[ ]:


print(X)

# In[ ]:


print(y)

# In[ ]:


# Standardising the Dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# In[ ]:


scaler.fit(X)

# In[ ]:


standardized_data = scaler.transform(X)

# In[ ]:


print(standardized_data)

# In[ ]:


X = standardized_data
target = df['Outcome']

# In[ ]:


print(X)
print(y)

# In[ ]:


from sklearn.model_selection import train_test_split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[ ]:


# Standardising the Dataset
mean = X_train.mean()
std = X_test.std()

X_train = (X_train - mean) / std
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = (X_test - mean) / std
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# ## Defining SVM Classifier from Scratch

# In[ ]:


class SVM_Classifier:

    def __init__(self, learning_rate, lambda_parameter, no_of_iterations):
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter
        self.no_of_iterations = no_of_iterations

    def fit(self, X, y):
        self.m, self.n = X.shape

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.y = y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):

        y_label = np.where(self.y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):

            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if (condition == True):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat


# In[ ]:


# Assuming you have X_train and y_train as your training data
clf = SVM_Classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)

# In[ ]:


clf.fit(X_train, y_train)

# In[ ]:


X_train_predict_svm = clf.predict(X_train)
X_test_predict_svm = clf.predict(X_test)

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# In[ ]:


Train_data_accuracy = accuracy_score(y_train, X_train_predict_svm)

# In[ ]:


print(Train_data_accuracy)

# In[ ]:


X_test_predict_svm = clf.predict(X_test)

# In[ ]:


Test_data_accuracy = accuracy_score(y_test, X_test_predict_svm)

# In[ ]:


print(Test_data_accuracy)

# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, X_test_predict_svm)

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


# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, X_test_predict_svm))

# In[ ]:


