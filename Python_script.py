# Task 1: Load and Explore the Dataset

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, colums=iris.feature_names)
df['species'] = iris.target

# Map species to their names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Display the first few rows
print(df.head())

# Explore data structure and missing values
print(df.info())
print(df.describe())

# Task 2: Basic Data Analysis

# Compute basic statistics
print(df.describe())

# Group by species and compute mean of sepal length
print(df.groupby('species')['sepal length (cm)'].mean())

# Task 3: Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart
species_mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
plt.figure(figsize=(8, 6))
plt.plot(species_mean_length.index, species_mean_sepal_length.values, marker='o')
plt.title('Mean Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Sepal Length ('cm)')
plt.show()

# Bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal lenght (cm)', hue='species', data=df)
plt.title('Relationship between Sepal Length and Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()