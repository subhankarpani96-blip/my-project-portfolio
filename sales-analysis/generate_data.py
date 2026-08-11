import pandas as pd
import numpy as np

np.random.seed(42)

n = 3000
regions = ['East', 'West', 'North', 'South']
categories = ['Furniture', 'Office Supplies', 'Technology']
sub_categories = {
    'Furniture': ['Chairs', 'Tables', 'Bookcases'],
    'Office Supplies': ['Binders', 'Paper', 'Storage'],
    'Technology': ['Phones', 'Accessories', 'Machines']
}
segments = ['Consumer', 'Corporate', 'Home Office']

dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
order_dates = np.random.choice(dates, n)

rows = []
for i in range(n):
    category = np.random.choice(categories, p=[0.25, 0.45, 0.30])
    sub_category = np.random.choice(sub_categories[category])
    region = np.random.choice(regions)
    segment = np.random.choice(segments)
    quantity = np.random.randint(1, 10)
    base_price = {'Furniture': 250, 'Office Supplies': 40, 'Technology': 400}[category]
    unit_price = base_price * np.random.uniform(0.6, 1.6)
    sales = round(unit_price * quantity, 2)
    discount = np.random.choice([0, 0.1, 0.15, 0.2, 0.3], p=[0.4, 0.25, 0.15, 0.1, 0.1])
    profit_margin = np.random.uniform(0.05, 0.35) - discount * 0.6
    profit = round(sales * profit_margin, 2)
    rows.append([i+1, order_dates[i], region, segment, category, sub_category, quantity, sales, discount, profit])

df = pd.DataFrame(rows, columns=['OrderID', 'OrderDate', 'Region', 'Segment', 'Category',
                                   'SubCategory', 'Quantity', 'Sales', 'Discount', 'Profit'])
df = df.sort_values('OrderDate').reset_index(drop=True)
df.to_csv('/home/claude/portfolio/sales-analysis/data/superstore_sales.csv', index=False)
print(df.shape)
print(df.head())
