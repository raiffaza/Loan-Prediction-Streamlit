import joblib
import pandas as pd
import time  
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# Define paths
model_path = r'H:\Kuliah\Bootcamp\Finpro\rf_model.pkl'
scaler_path = r'H:\Kuliah\Bootcamp\Finpro\scaler.pkl'

# Measure the start time
start_time = time.time()

try:
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Example: data will be approved
income = 61853    
credit_score = 732    
loan_amount = 19210    
dti_ratio = 44.13    
employment_status = 1  # 1 for employed, 0 for unemployed

# Create a DataFrame with the features
input_data = pd.DataFrame([[income, credit_score, loan_amount, dti_ratio, employment_status]],
                          columns=['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status'])

# Scale the input data using the same scaler as the training data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

# Print result
if prediction == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

# Measure and print runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Prediction time for Random Forest: {runtime:.4f} seconds")
