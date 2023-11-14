import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df_improve = self.df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    def explore_data(self):
        print(self.df.head())
        print(self.df.shape)
        print(self.df.info())
        print(self.df.describe())
        print(self.df.isnull().sum())
        Gender_counts = self.df['Gender'].value_counts()
        print(Gender_counts)

    def visualize_data(self):
        # Pair plot
        sns.pairplot(self.df_improve, hue='Gender')
        plt.show()

        # Histograms
        features = self.df_improve.columns.tolist()
        features.remove('Gender')
        for feature in features:
            sns.histplot(data=self.df_improve, x=feature, hue='Gender', kde=True)
            plt.title(f'Histogram of {feature}')
            plt.show()

        # Pie chart for Gender Distribution
        Gender_counts = self.df['Gender'].value_counts()
        print(Gender_counts)
        plt.figure(figsize=(6, 6))
        plt.pie(Gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', startangle=140)
        plt.title('Gender Distribution')
        plt.show()

        # Count plot for each class in the 'Gender' column
        sns.countplot(x=self.df_improve['Gender'], data=self.df_improve)
        plt.show()

        # Scatter plots
        plt.figure(1, figsize=(12, 5))
        for gender in ['Male', 'Female']:
            plt.scatter(x='Age', y='Annual Income (k$)', data=self.df_improve[self.df_improve['Gender'] == gender],
                        s=200, alpha=0.5,
                        label=gender)
        plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
        plt.title('Age vs Annual Income w.r.t Gender')
        plt.legend()
        plt.show()

        plt.figure(1, figsize=(12, 5))
        for gender in ['Male', 'Female']:
            plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)',
                        data=self.df_improve[self.df_improve['Gender'] == gender],
                        s=200, alpha=0.5, label=gender)
        plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
        plt.title('Annual Income (k$) vs Spending Score (1-100) w.r.t Gender')
        plt.legend()
        plt.show()

        plt.figure(1, figsize=(12, 5))
        for gender in ['Male', 'Female']:
            plt.scatter(x='Age', y='Spending Score (1-100)', data=self.df_improve[self.df_improve['Gender'] == gender],
                        s=200, alpha=0.5,
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
            sns.violinplot(x=cols, y='Gender', data=self.df_improve, palette='vlag')
            sns.swarmplot(x=cols, y='Gender', data=self.df_improve)
            plt.ylabel('Gender' if n == 1 else '')
            plt.title('Boxplot & Swarm-plot' if n == 2 else '')
        plt.show()


