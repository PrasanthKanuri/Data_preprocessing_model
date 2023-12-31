# Data Processing and Machine Learning Pipeline

This repository contains a Python script for data processing and building a machine learning model using the scikit-learn library.

## Overview

The script performs the following tasks:

1. **Data Loading:** Reads a dataset from a CSV file (`Data.csv`).
2. **Handling Missing Values:** Uses the mean strategy to impute missing values in the dataset.
3. **One-Hot Encoding:** Applies one-hot encoding to categorical variables in the dataset.
4. **Label Encoding:** Encodes the target variable using label encoding.
5. **Train-Test Split:** Splits the dataset into training and testing sets.
6. **Feature Scaling:** Standardizes the numeric features using StandardScaler.

## Prerequisites

- Python 3.x
- Required Python packages: numpy, pandas, matplotlib, scikit-learn

File Structure
Data-preprocessing-for-model.py: The main Python script.
Data.csv: Input dataset in CSV format.