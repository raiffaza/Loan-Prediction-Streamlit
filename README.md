# Loan-Prediction-Streamlit
> ## Business Problem

Companies or financial institutions need to determine whether a loan application from a potential borrower should be approved or rejected. This process is important to minimize the risk of non-performing loans and ensure that only financially viable borrowers are approved. If done manually, this process can be time-consuming, prone to bias, and inefficient, especially if the volume of applications is high.

> ## Purpose of Machine Learning

The purpose of using machine learning in this case is to automate the loan application evaluation process by building a predictive model that can classify loan applications into two categories: `Approved` or `Rejected`. The model uses features such as Income, Credit Score, Loan Amount, DTI Ratio, and Employment Status to predict loan eligibility.

Specifically, the machine learning objectives of this project are:
- Help decide automatically whether a loan application will be approved or rejected based on the prospective borrower's data.
- Improve accuracy and consistency in the loan approval process by reducing human subjectivity and bias.
- Speed up the loan application evaluation process so that it is more efficient and scalable.

The model used is `K-Nearest Neighbors (KNN)`, which has been optimized with GridSearchCV to determine the best parameters, and trained to recognize patterns from historical loan application data with the label `Approved` or `Rejected`.

Notes: 
- If u want to try run in your own computer to test runtime between Random Forest and KNN, you can check `predict KNN/RF.py`

----
