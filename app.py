import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, roc_auc_score

def detect_fraud(new_data, model):
    # Scaling
    new_data[['Scaled_Amount', 'Scaled_Time']] = scaler.transform(new_data[['Amount', 'Time']])
    
    # PCA Transformation
    new_data[['PCA1', 'PCA2']] = pca.transform(new_data[['Scaled_Amount', 'Scaled_Time']])
    
    # Prediction
    predictions = model.predict(new_data[['PCA1', 'PCA2']])
    fraudulent_transactions = new_data[predictions == -1]
    
    return fraudulent_transactions

def streamlit_app():
    st.title("Credit Card Fraud Detection")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        # Load the Excel file
        xls = pd.ExcelFile(uploaded_file)
        
        # If the file has only one sheet, use that sheet
        if len(xls.sheet_names) == 1:
            sheet_name = xls.sheet_names[0]
        else:
            # Otherwise, allow the user to select which sheet to use
            sheet_name = st.selectbox("Select the sheet", xls.sheet_names)
        
        transactions = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Verify that 'Amount' and 'Time' columns are present
        if 'Amount' in transactions.columns and 'Time' in transactions.columns:
            if 'Class' in transactions.columns:
                # Visualize the distribution of classes
                fig, ax = plt.subplots()
                sns.countplot(x='Class', data=transactions, ax=ax)
                ax.set_title('Distribution of Class (Fraud vs Non-Fraud)')
                st.pyplot(fig)

                # Visualize the transaction amount distribution
                fig, ax = plt.subplots()
                sns.histplot(transactions[transactions['Class'] == 0]['Amount'], bins=50, kde=True, color='blue', label='Non-Fraud', ax=ax)
                sns.histplot(transactions[transactions['Class'] == 1]['Amount'], bins=50, kde=True, color='red', label='Fraud', ax=ax)
                ax.legend()
                ax.set_title('Transaction Amount Distribution')
                st.pyplot(fig)
            else:
                st.warning("The 'Class' column is missing, so fraud vs non-fraud visualizations are skipped.")

            # Standardize the data
            global scaler, pca, isolation_forest, lof
            scaler = StandardScaler()
            transactions[['Scaled_Amount', 'Scaled_Time']] = scaler.fit_transform(transactions[['Amount', 'Time']])

            # PCA transformation
            pca = PCA(n_components=2)
            transactions[['PCA1', 'PCA2']] = pca.fit_transform(transactions[['Scaled_Amount', 'Scaled_Time']])

            # Train Isolation Forest
            isolation_forest = IsolationForest(contamination=0.0017, random_state=42)
            transactions['IF_Pred'] = isolation_forest.fit_predict(transactions[['PCA1', 'PCA2']])

            # Train Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.0017)
            transactions['LOF_Pred'] = lof.fit_predict(transactions[['PCA1', 'PCA2']])

            if 'Class' in transactions.columns:
                # Evaluate Isolation Forest
                st.write("### Isolation Forest Evaluation:")
                st.text(classification_report(transactions['Class'], transactions['IF_Pred']))
                st.write("ROC AUC:", roc_auc_score(transactions['Class'], transactions['IF_Pred']))

                # Evaluate LOF
                st.write("### LOF Evaluation:")
                st.text(classification_report(transactions['Class'], transactions['LOF_Pred']))
                st.write("ROC AUC:", roc_auc_score(transactions['Class'], transactions['LOF_Pred']))

                # Scatter plot for anomalies detection
                fig, ax = plt.subplots()
                sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=transactions, palette='coolwarm', alpha=0.6, ax=ax)
                ax.set_title('Anomalies Detection Scatter Plot')
                st.pyplot(fig)

            # Detect fraud in the uploaded data
            frauds = detect_fraud(transactions, isolation_forest)
            st.write("### Detected Fraudulent Transactions")
            st.write(frauds)

            # Scatter plot for detected frauds
            fig, ax = plt.subplots()
            sns.scatterplot(x='PCA1', y='PCA2', hue=frauds.index, palette='coolwarm', ax=ax, data=frauds)
            ax.set_title('Detected Anomalies in New Data')
            st.pyplot(fig)

        else:
            st.error("The uploaded file does not contain 'Amount' and 'Time' columns.")
    else:
        st.info("Please upload an Excel file.")

if __name__ == "__main__":
    streamlit_app()

