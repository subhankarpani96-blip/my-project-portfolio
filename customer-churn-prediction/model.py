"""
Customer Churn Prediction
--------------------------
Goal: predict which customers are likely to cancel their subscription,
so the business can target them with retention offers.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay

sns.set_style('whitegrid')
df = pd.read_csv('data/telco_churn.csv')

# EDA: churn rate by contract type (classic churn insight)
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
contract_churn.plot(kind='barh', ax=ax, color='crimson')
ax.set_title('Churn Rate by Contract Type')
ax.set_xlabel('Churn Rate')
plt.tight_layout()
plt.savefig('outputs_churn_by_contract.png', dpi=120)
plt.close()

features = ['SeniorCitizen', 'Tenure', 'Contract', 'InternetService', 'PaymentMethod',
            'TechSupport', 'OnlineSecurity', 'NumComplaints', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

categorical = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical)
], remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results[name] = (acc, auc, pipe)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds))

best_name = max(results, key=lambda k: results[k][1])
best_acc, best_auc, best_pipe = results[best_name]

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax)
ax.set_title(f'ROC Curve - {best_name}')
plt.tight_layout()
plt.savefig('outputs_roc_curve.png', dpi=120)
plt.close()

print(f"\nBest model: {best_name} | Accuracy: {best_acc:.3f} | ROC-AUC: {best_auc:.3f}")
