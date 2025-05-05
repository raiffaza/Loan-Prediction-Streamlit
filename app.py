import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('knn_model.pkl')

# Define the prediction function
def make_prediction(input_features):
    prediction = model.predict(input_features)
    return prediction[0]

# Streamlit app
def main():
    st.title('Loan Approval Prediction App')

    # Input fields for user data
    income = st.number_input('Income')
    credit_score = st.number_input('Credit Score')
    loan_amount = st.number_input('Loan Amount')
    dti_ratio = st.number_input('Debt to Income Ratio')
    loan_to_income = st.number_input('Loan to Income Ratio')
    risk_score = st.number_input('Risk Score')

    # Convert inputs into a numpy array and reshape for prediction
    if st.button('Predict'):
        input_data = np.array([income, credit_score, loan_amount, dti_ratio, loan_to_income, risk_score]).reshape(1, -1)
        prediction = make_prediction(input_data)
        if prediction == 1:
            st.success('Loan Approved')
        else:
            st.error('Loan Rejected')

# Run the app
if __name__ == '__main__':
    main()
