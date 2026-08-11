# Customer Churn Prediction

## Business Question
Which customers are likely to cancel their subscription, so the
business can proactively offer retention deals before they leave?

## Dataset
Simulated telecom customer data (2,500 customers) styled after the
well-known Telco Customer Churn dataset — contract type, tenure,
internet service, support add-ons, complaints, and charges.

## Approach
1. EDA: churn rate broken down by contract type (a standard churn lever)
2. One-hot encoded categorical features (Contract, Internet Service, etc.)
3. Trained Logistic Regression and Random Forest classifiers
4. Evaluated with accuracy, precision/recall, and **ROC-AUC** (the
   right metric here since churn/no-churn is imbalanced and business
   cares about ranking risk, not just a single cutoff)

## Results
- **Logistic Regression: 79.6% accuracy, 0.883 ROC-AUC** (best model)
- Random Forest: 79.2% accuracy, 0.878 ROC-AUC
- Customers on **month-to-month contracts churn far more** than those
  on 1- or 2-year contracts — the single strongest churn signal.

## Recommendation
Target month-to-month customers with incentives to move to annual
contracts, and prioritize outreach to high-complaint, low-tenure customers.

## How to Run
```
pip install pandas scikit-learn matplotlib seaborn
python generate_data.py
python model.py
```

## Tools Used
Python, pandas, scikit-learn, matplotlib, seaborn

---
### How to explain this in an interview (60 seconds)
"I built a churn classifier for a telecom-style dataset. Instead of
just optimizing accuracy, I used ROC-AUC as the main metric since churn
is imbalanced and the business cares about ranking customers by risk,
not just a yes/no prediction. The strongest signal by far was contract
type — month-to-month customers churn much more than annual customers —
so my recommendation was to focus retention offers on converting
month-to-month customers to longer contracts."
