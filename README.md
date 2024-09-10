# 💳 Credit Card Fraud Detection System

## Overview

The **Credit Card Fraud Detection System** is a web application built using Python, Streamlit, and machine learning techniques such as **PCA**, **Isolation Forest**, and **Local Outlier Factor (LOF)**. The system is designed to detect fraudulent transactions based on patterns in transaction data, with visualization capabilities and evaluation metrics to assist in identifying anomalies effectively.

## Features

- **Data Upload**: Upload your transaction data from Excel files.
- **Data Visualization**: Visualize the distribution of fraud and non-fraud transactions, along with transaction amounts.
- **Data Preprocessing**: Standardize data using `StandardScaler` and reduce dimensions using `PCA`.
- **Fraud Detection**:
  - **Isolation Forest**: Detect outliers based on unsupervised learning.
  - **Local Outlier Factor (LOF)**: Detect anomalies using a density-based approach.
- **Model Evaluation**: Evaluate model performance using metrics such as classification reports and ROC AUC scores.
- **Detected Frauds**: Display detected fraudulent transactions with visualization.

## Technologies Used

- **Backend**: Python (Streamlit)
- **Machine Learning**: 
  - `PCA` for dimensionality reduction
  - `Isolation Forest` and `Local Outlier Factor` for anomaly detection
- **Data Visualization**: Seaborn and Matplotlib
- **File Handling**: Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ms-shashank/detecting-fraudulent-transactions.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## How It Works

1. **Upload Transaction Data**: The app accepts an Excel file as input. Ensure your data contains `Amount` and `Time` columns.
2. **Data Preprocessing**: The app scales transaction data (`Amount`, `Time`) and applies PCA for dimensionality reduction.
3. **Fraud Detection**: 
   - The **Isolation Forest** and **LOF** models are trained on the transformed data to detect anomalies.
   - Detected fraudulent transactions are displayed with a visualization of anomalies.
4. **Model Evaluation**: If your data includes the `Class` column, model performance is evaluated and displayed with metrics like the classification report and ROC AUC score.

## Usage

1. **Upload your data**: Choose an Excel file with transaction data.
2. **View Results**: 
   - Visualizations for transaction distributions and detected anomalies.
   - Detected fraudulent transactions.
3. **Evaluate Models**: View model performance metrics if labeled data (`Class`) is available.
