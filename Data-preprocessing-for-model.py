#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Displaying the loaded data
print("Input Features (X):")
print(X)
print("\nTarget Variable (y):")
print(y)

# Handling missing data using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Displaying the data after handling missing values
print("\nInput Features after handling missing values:")
print(X)

# Encoding categorical data using OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Displaying the data after encoding categorical data
print("\nInput Features after encoding categorical data:")
print(X)

# Encoding the dependent variable using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Displaying the encoded target variable
print("\nEncoded Target Variable:")
print(y)

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Displaying the training and testing sets
print("\nTraining Features:")
print(X_train)
print("\nTesting Features:")
print(X_test)
print("\nTraining Target:")
print(y_train)
print("\nTesting Target:")
print(y_test)

# Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Displaying the scaled training features
print("\nScaled Training Features:")
print(X_train)

# Displaying the scaled testing features
print("\nScaled Testing Features:")
print(X_test)
