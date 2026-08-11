"""
Cricket Match Win Prediction
-----------------------------
Goal: predict whether TeamA wins, based on form, venue, toss, and
head-to-head history. Classic binary classification problem.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('data/matches.csv')

# Feature engineering: differences matter more than raw values
df['FormDiff'] = df['TeamA_RecentForm'] - df['TeamB_RecentForm']
df['AvgScoreDiff'] = df['TeamA_AvgScore'] - df['TeamB_AvgScore']
df['H2HDiff'] = df['H2H_A_Wins'] - df['H2H_B_Wins']
df['TossWonByA'] = (df['TossWinner'] == df['TeamA']).astype(int)

features = ['Venue', 'TossDecision', 'FormDiff', 'AvgScoreDiff', 'H2HDiff', 'TossWonByA']
X = df[features]
y = df['TeamA_Wins']

categorical = ['Venue', 'TossDecision']
numeric = ['FormDiff', 'AvgScoreDiff', 'H2HDiff', 'TossWonByA']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical)
], remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

# Use best model for confusion matrix plot
best_name = max(results, key=results.get)
best_pipe = Pipeline([('prep', preprocessor), ('clf', models[best_name])])
best_pipe.fit(X_train, y_train)
preds = best_pipe.predict(X_test)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['TeamB Wins', 'TeamA Wins'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('outputs_confusion_matrix.png', dpi=120)
plt.close()

print(f"\nBest model: {best_name} with accuracy {results[best_name]:.3f}")
