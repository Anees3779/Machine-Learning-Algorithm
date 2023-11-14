#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


df = pd.read_csv("diabetes.csv")

# In[3]:


df.head()

# ## Exploratory  Data Analysis

# In[4]:


# for getting the shape (No of rows & columns)
df.shape

# In[5]:


# Column Details
df.columns

# In[6]:


# getting brief summary of data
df.info()

# In[7]:


# Checking Null Values
df.isnull().sum()

# In[8]:


# Statistical summary of dataset
df.describe()

# In[9]:


df.corr()

# In[10]:


cor = df.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()

# In[11]:


# Calculating the number of zero-values in in each column
print('rows having null Pregnancies value : {0}'.format(len(df.loc[df['Pregnancies'] == 0])))
print('rows having null Glucose value : {0}'.format(len(df.loc[df['Glucose'] == 0])))
print('rows having null BloodPressure value : {0}'.format(len(df.loc[df['BloodPressure'] == 0])))
print('rows having null SkinThickness value : {0}'.format(len(df.loc[df['SkinThickness'] == 0])))
print('rows having null Insulin value : {0}'.format(len(df.loc[df['Insulin'] == 0])))
print('rows having null BMI value : {0}'.format(len(df.loc[df['BMI'] == 0])))
print('rows having null DiabetesPedigreeFunction value : {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print('rows having null Age value : {0}'.format(len(df.loc[df['Age'] == 0])))

# In[12]:


# Calculating the mean value of every column
means = df.iloc[:, 1:6].mean()
means

# In[13]:


nonzeros = list(df.columns[1:6])
for column in nonzeros:
    df[column] = df[column].replace(0, means[column])

# In[14]:


# Preview of Changed Dataset
df.head()

# In[15]:


outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)

# In[16]:


df_improve = df[['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']]

# In[17]:


# improved dataset after datacleaning
df_improve.head()

# In[18]:


df_improve.corr()

# ## Data Visualisation
#

# In[19]:


# Heatmap
cor = df_improve.corr()
sns.heatmap(cor, cmap="crest", annot=True)
plt.show()

# In[20]:


# Pairplot
sns.pairplot(df_improve, hue='Outcome')

# In[21]:


# Histogram
features = df_improve.columns.tolist()
features.remove('Outcome')
# Create histograms for each feature with 'Outcome' as the hue
for feature in features:
    sns.histplot(data=df_improve, x=feature, hue='Outcome', kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()

# In[22]:


# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Fit', 'Diabetic'], autopct='%1.1f%%', startangle=140)
plt.title('Outcome Count by Fit and Diabetic')
plt.show()

# In[23]:


# Plot the count of each class in the 'Outcome' column
sns.countplot(x=df_improve['Outcome'], data=df_improve)
plt.show()

# ## Naive Bayes

# In[24]:


X = df_improve.loc[:, ['Glucose', 'Insulin', 'BMI', 'Age']].to_numpy()
y = df_improve.loc[:, 'Outcome'].to_numpy()

# In[25]:


print(X)

# In[26]:


print(y)

# In[27]:


# Standardising the Dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# In[28]:


scaler.fit(X)

# In[29]:


standardized_data = scaler.transform(X)

# In[30]:


print(standardized_data)

# In[31]:


X = standardized_data
target = df['Outcome']

# In[32]:


print(X)
print(y)

# In[33]:


from sklearn.model_selection import train_test_split

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# In[35]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# ## Naive Bayes Algorithm

# In[36]:


import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# In[37]:


NB_Model = NaiveBayes()

# In[38]:


NB_Model.fit(X_train, y_train)

# In[39]:


y_pred_NB = NB_Model.predict(X_test)

# In[40]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[41]:


def evaluate_model_performance(y_train, y_test, y_pred_NB):
    test_accuracy = accuracy_score(y_test, y_pred_NB)
    test_precision = precision_score(y_test, y_pred_NB)
    test_recall = recall_score(y_test, y_pred_NB)
    test_f1 = f1_score(y_test, y_pred_NB)

    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)


# In[42]:


evaluate_model_performance(y_train, y_test, y_pred_NB)

# In[47]:


from sklearn import metrics

# In[48]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_NB)

# In[49]:


print(cm)

# In[51]:


cm = metrics.confusion_matrix(y_test, y_pred_NB)

# You have a typo in the following line. Instead of "pd.df(cfn_matrix)", you should use "pd.DataFrame(cfn_matrix)".
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# In[53]:


# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_NB))

# In[ ]: