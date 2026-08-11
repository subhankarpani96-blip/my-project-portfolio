"""
Retail Sales Analysis
----------------------
Business question: Where is the company making/losing money, and what
should it change to grow profit, not just revenue?
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
df = pd.read_csv('data/superstore_sales.csv', parse_dates=['OrderDate'])

# 1. Basic cleaning check
print("Missing values:\n", df.isnull().sum())
print("\nShape:", df.shape)

# 2. Sales & Profit by Category
cat_summary = df.groupby('Category')[['Sales', 'Profit']].sum().sort_values('Profit')
print("\nCategory summary:\n", cat_summary)

fig, ax = plt.subplots(figsize=(8, 5))
cat_summary.plot(kind='barh', ax=ax)
ax.set_title('Total Sales vs Profit by Category')
plt.tight_layout()
plt.savefig('outputs_sales_by_category.png', dpi=120)
plt.close()

# 3. Discount vs Profit relationship (key business insight)
discount_profit = df.groupby('Discount')['Profit'].mean()
fig, ax = plt.subplots(figsize=(8, 5))
discount_profit.plot(kind='bar', ax=ax, color='indianred')
ax.set_title('Average Profit Drops Sharply as Discount Increases')
ax.set_ylabel('Avg Profit ($)')
plt.tight_layout()
plt.savefig('outputs_discount_vs_profit.png', dpi=120)
plt.close()

# 4. Monthly sales trend
monthly = df.set_index('OrderDate').resample('ME')['Sales'].sum()
fig, ax = plt.subplots(figsize=(10, 5))
monthly.plot(ax=ax, marker='o')
ax.set_title('Monthly Sales Trend')
plt.tight_layout()
plt.savefig('outputs_monthly_trend.png', dpi=120)
plt.close()

# 5. Region performance
region_summary = df.groupby('Region')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False)
print("\nRegion summary:\n", region_summary)

print("\nDone. Charts saved as PNGs in this folder.")
