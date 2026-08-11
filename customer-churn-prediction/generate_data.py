import pandas as pd
import numpy as np

np.random.seed(21)
n = 2500

contract_types = ['Month-to-month', 'One year', 'Two year']
internet_service = ['DSL', 'Fiber optic', 'No']
payment_method = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']

rows = []
for i in range(n):
    tenure = np.random.randint(0, 73)
    contract = np.random.choice(contract_types, p=[0.55, 0.25, 0.2])
    monthly_charges = round(np.random.uniform(18, 120), 2)
    internet = np.random.choice(internet_service, p=[0.35, 0.45, 0.2])
    payment = np.random.choice(payment_method)
    tech_support = np.random.choice(['Yes', 'No'])
    online_security = np.random.choice(['Yes', 'No'])
    num_complaints = np.random.poisson(1.2)
    senior_citizen = np.random.choice([0, 1], p=[0.85, 0.15])
    total_charges = round(monthly_charges * tenure * np.random.uniform(0.95, 1.05), 2)

    # underlying churn probability logic
    churn_score = 0
    churn_score += 0.35 if contract == 'Month-to-month' else (0.1 if contract == 'One year' else 0)
    churn_score += 0.15 if internet == 'Fiber optic' else 0
    churn_score += 0.1 if tech_support == 'No' else 0
    churn_score += 0.1 if online_security == 'No' else 0
    churn_score += 0.03 * num_complaints
    churn_score += 0.1 if payment == 'Electronic check' else 0
    churn_score -= 0.004 * tenure
    churn_score += np.random.normal(0, 0.15)

    churn = 1 if churn_score > 0.35 else 0

    rows.append([f"CUST{1000+i}", senior_citizen, tenure, contract, internet, payment,
                 tech_support, online_security, num_complaints, monthly_charges,
                 total_charges, churn])

df = pd.DataFrame(rows, columns=[
    'CustomerID', 'SeniorCitizen', 'Tenure', 'Contract', 'InternetService',
    'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'NumComplaints',
    'MonthlyCharges', 'TotalCharges', 'Churn'
])
df.to_csv('/home/claude/portfolio/customer-churn-prediction/data/telco_churn.csv', index=False)
print(df.shape)
print(df['Churn'].value_counts(normalize=True))
