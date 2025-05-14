import streamlit as st
import joblib
import pandas as pd

# Load pre-trained model and scaler
MODEL_PATH = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error("Error loading model or scaler. Please check the file paths.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="JPMorgan Loan Prediction",
    page_icon="🏦",
    layout="centered"
)

# --- HEADER ---
# Display JPMorgan Logo (centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("JPMorgan.png", width=250)

# Title and Subtitle
st.title("JPMorgan Loan Prediction App")
st.markdown("<h3 style='text-align: center;'>Fast • Secure • Intelligent</h3>", unsafe_allow_html=True)

# --- ABOUT JP MORGAN SECTION ---
st.markdown("""
---
## About JPMorgan Chase

> **JPMorgan Chase & Co.** is one of the most prestigious and influential banks in the world. As the largest bank in the United States by assets, it serves millions of customers globally through its diverse financial services — including investment banking, asset management, commercial banking, and consumer finance.

With a strong commitment to **technology, innovation, and customer-centric solutions**, JPMorgan has been at the forefront of leveraging **machine learning and artificial intelligence** to improve decision-making, risk assessment, and customer experience across its operations.

This web application reflects that same spirit of innovation — bringing **cutting-edge machine learning models** into the heart of loan approval processes to deliver faster, smarter, and fairer decisions for customers.
""")

# --- PURPOSE OF THE APP ---
st.markdown("""
---
## Purpose of This Tool

This interactive platform helps users **predict whether a loan will be approved or rejected**, based on key financial metrics such as:

- Income  
- Credit Score  
- Loan Amount  
- Debt-to-Income Ratio  
- Employment Status  

### 🎯 Objectives:
- Automate the loan evaluation process  
- Minimize human bias and errors  
- Deliver instant, data-driven predictions  
- Help applicants understand their eligibility before applying

This aligns perfectly with **JPMorgan’s mission to build smarter, scalable, and secure financial systems powered by AI and machine learning**.
""")

# --- INPUT FORM ---
st.markdown("---")
st.header("Enter Your Financial Information")

# Input fields
income = st.number_input("Monthly Income ($)", min_value=0, help="Enter your total monthly income")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, help="Your FICO credit score")
loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0)
dti_ratio = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, help="Total monthly debt payments divided by gross monthly income")
employment_status = st.selectbox(
    "Employment Status",
    options=[0, 1],
    format_func=lambda x: "Unemployed" if x == 0 else "Employed",
    help="Are you currently employed?"
)

# Submit button
submit_button = st.button("Check Loan Eligibility", use_container_width=True)

# --- PREDICTION LOGIC ---
if submit_button:
    # Create DataFrame from input
    input_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [employment_status]
    })

    # Scale input data using the loaded scaler
    try:
        input_data_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error("Error scaling input data. Please ensure all inputs are valid.")
        st.stop()

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]

    # --- DISPLAY RESULT ---
    st.markdown("## 📊 Prediction Result")
    if prediction == 1:
        st.success("✅ Congratulations! Your loan is likely to be **Approved**.", icon="💰")
    else:
        st.error("❌ Unfortunately, your loan may be **Rejected**.", icon="🚫")

    # --- ADDITIONAL INFO ---
    st.markdown("""
    ---
    ### 🔍 Explanation of Features

    - **Income**: Higher income generally increases chances of approval.
    - **Credit Score**: A score above 700 is typically considered good.
    - **Loan Amount**: Larger loans carry higher risk.
    - **DTI Ratio**: Lower ratios indicate better repayment ability.
    - **Employment Status**: Employed individuals are seen as lower risk.

    """)
    st.markdown("Powered by Machine Learning | Naufal Dzakia Raiffaza")
