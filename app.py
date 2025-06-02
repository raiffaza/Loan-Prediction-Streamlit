import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model_path = r'H:\Kuliah\Bootcamp\Finpro\knn_model.pkl'
scaler_path = r'H:\Kuliah\Bootcamp\Finpro\scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Set page config
st.set_page_config(page_title="JPMorgan Loan Prediction App", page_icon="🏦")

# Centered Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("JPMorgan.png", width=250)

# Title and Description
st.title("JPMorgan Loan Prediction App")
st.markdown("""
### JPMorgan, a globally recognized leader in banking and financial services, is leveraging its extensive expertise to bring you an innovative loan evaluation system.
Utilizing cutting-edge technology and data-driven decision-making, JPMorgan introduces a smart loan evaluation system powered by machine learning.
This application enables users to quickly assess their eligibility for a loan by simply entering key financial information.
Receive an instant prediction on whether your loan application is likely to be approved or rejected, all while benefiting from a secure, transparent, and efficient process.
""")

# Business Problem Section
st.markdown("""
## Business Problem
Financial institutions need a reliable way to determine whether a loan application from a potential borrower should be approved or rejected. A well-designed loan approval process helps minimize the risk of non-performing loans and ensures that only financially viable applicants are approved.

Manual processes are time-consuming, subject to human bias, and can struggle to keep up with a high volume of applications. In addition, traditional methods may not be able to scale effectively as the number of applications increases.

Using automation through machine learning can address these challenges and bring greater efficiency, fairness, and speed to the loan evaluation process.
""")

# Purpose of this website
st.markdown("""
## Purpose of this website
The objectives of using loan prediction in this application are:
- To assist in the automatic decision-making process of whether a loan will be approved or rejected, based on prospective borrowers' data.
- To improve the accuracy and consistency of loan approval decisions, minimizing human subjectivity and bias.
- To expedite the loan evaluation process, ensuring efficiency and scalability, especially in high-application volume environments.
""")

# Input Section
st.header("Enter Your Data")
income = st.number_input("Income", min_value=0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850)
loan_amount = st.number_input("Loan Amount", min_value=0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
employment_status = st.selectbox(
    "Employment Status",
    [0, 1],
    index=1,
    format_func=lambda x: "Employed" if x == 1 else "Unemployed"
)

# Submit Button
submit_button = st.button("Check Loan Eligibility")

# Prediction Result (appears only after button click)
if submit_button:
    # Create input data as DataFrame
    input_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [employment_status]
    })

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]

    # Display result in an attractive manner
    st.markdown("## 📊 Prediction Result")
    if prediction == 1:
        st.success("✅ Congratulations! Your loan is likely to be **Approved**.", icon="💰")
    else:
        st.error("❌ Unfortunately, your loan may be **Rejected**.", icon="🚫")

    # Additional Info
    st.markdown("""
    ---
    ### Explanation
    - `Text` (string): User-provided reason for requesting a loan.
    - `Income` (numeric): Applicant's income.
    - `Credit_Score` (numeric): Applicant’s credit score.
    - `Loan_Amount` (numeric): Amount of loan requested.
    - `DTI_Ratio` (numeric): Debt-to-Income ratio.
    - `Employment_Status` (categorical): Employment status (`employed` / `unemployed`).
    """)
