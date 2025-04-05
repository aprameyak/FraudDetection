# Fraud Detection 

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white&style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)

## About

This project is a two-part Python suite for exploring **fraud detection in financial transactions** and **spam classification in email content**.

- The **fraud detection module** analyzes real-world transaction data to identify fraud patterns and visualize key metrics.
- The **spam classifier** is trained on the Enron email dataset to detect spam using a logistic regression model with high accuracy.

## Features

### Fraud Detection
- Load and inspect transaction data (`transactions.csv`)
- Visualize transaction types and fraud frequency with bar charts
- Detect top recipients and fraudulent accounts
- Compare balance changes in cash-out transactions
- Group by origin accounts and analyze recipient diversity

### Spam Classification
- Load and preprocess spam/ham emails from the Enron dataset
- Clean email text with regex-based preprocessing
- Convert text to feature vectors using `CountVectorizer`
- Train a logistic regression model to classify emails
- Output model accuracy, confusion matrix, and detailed classification metrics

## Technology Stack

- **Language**: Python  
- **Data Analysis**: Pandas  
- **Visualization**: Matplotlib  
- **Machine Learning**: Scikit-Learn (Logistic Regression, CountVectorizer)  
- **Email Dataset**: Enron Email Dataset (Ham and Spam folders)

## Dataset Notes

- Transaction dataset must be saved as `transactions.csv` in the project root
- Enron spam and ham emails must be organized in:

