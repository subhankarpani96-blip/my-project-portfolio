# Cricket Match Win Prediction

## Business Question
Given two teams, their recent form, head-to-head record, venue, and
toss result — can we predict who wins?

## Dataset
Simulated dataset of 1,200 international matches with recent form,
average scores, head-to-head record, venue, and toss details.

## Approach
1. Feature engineering: converted raw stats into **differences**
   (Team A's form minus Team B's form, etc.) — differences are more
   predictive than raw values for head-to-head problems.
2. Encoded categorical features (Venue, Toss Decision) with one-hot encoding.
3. Trained and compared two models: Logistic Regression and Random Forest.
4. Evaluated with accuracy, precision/recall, and a confusion matrix.

## Results
- **Logistic Regression: 73.8% accuracy** (best model)
- Random Forest: 71.3% accuracy
- Logistic Regression generalized slightly better here — a good example
  of a simpler model beating a more complex one on limited data.

## How to Run
```
pip install pandas numpy scikit-learn matplotlib
python generate_data.py
python model.py
```

## Tools Used
Python, pandas, scikit-learn, matplotlib

---
### How to explain this in an interview (60 seconds)
"I built a binary classifier to predict cricket match winners using
form, head-to-head history, venue, and toss data. The key modeling
decision was engineering *difference* features — Team A's form minus
Team B's — rather than feeding in raw stats, since what matters is
the relative gap between teams. I compared Logistic Regression against
Random Forest and picked Logistic Regression since it generalized
better and was more interpretable — I could show which factors moved
the win probability most."
