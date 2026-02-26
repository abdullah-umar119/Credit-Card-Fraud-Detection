import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload a CSV file containing transaction data.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    if st.button("Run Fraud Detection"):

        # Remove Class column if present
        if "Class" in data.columns:
            data_features = data.drop("Class", axis=1)
        else:
            data_features = data

        # Predict
        predictions = model.predict(data_features)
        probabilities = model.predict_proba(data_features)[:, 1]

        # Add results to dataframe
        data["Prediction"] = predictions
        data["Fraud_Probability"] = probabilities

        st.subheader("Prediction Results")
        st.write(data.head())

        # Summary statistics
        fraud_count = sum(predictions)
        total = len(predictions)
        fraud_percent = (fraud_count / total) * 100

        st.success(f"Total Transactions: {total}")
        st.error(f"Fraud Transactions Detected: {fraud_count}")
        st.warning(f"Fraud Percentage: {fraud_percent:.2f}%")

        # Download results button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            csv,
            "fraud_results.csv",
            "text/csv"
        )

        # Pie chart visualization
        st.subheader("Fraud vs Legitimate Distribution")

        labels = ['Legitimate', 'Fraud']
        values = [total - fraud_count, fraud_count]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.2f%%')
        ax.set_title("Fraud Detection Distribution")

        st.pyplot(fig)