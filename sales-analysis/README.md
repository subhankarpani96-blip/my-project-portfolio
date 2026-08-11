# Retail Sales & Profit Analysis

## Business Question
Where is the company making and losing money, and what should it change
to grow **profit**, not just revenue?

## Dataset
Simulated retail sales data (3,000 orders, 2023–2024) styled after the
classic Superstore dataset — Region, Segment, Category, Sales, Discount, Profit.

## Approach
1. Loaded and checked the data for missing values (`pandas`)
2. Aggregated Sales and Profit by Category and Region
3. Investigated the relationship between Discount level and average Profit
4. Plotted the monthly sales trend to spot seasonality
5. Visualized everything with `matplotlib` / `seaborn`

## Key Findings
- **Technology** drives the most revenue and profit; **Office Supplies**
  contributes revenue but thin margins.
- **Average profit drops sharply as discount level rises** — discounts
  above 20% are close to break-even or loss-making.
- Regional performance is fairly even, so the bigger lever is discount
  policy, not regional expansion.

## Recommendation
Cap discounts on low-margin categories (Office Supplies) and reserve
deep discounts for high-margin Technology items where profit can absorb it.

## How to Run
```
pip install pandas numpy matplotlib seaborn
python generate_data.py   # creates the dataset
python analysis.py        # runs the analysis, saves charts as PNG
```

## Tools Used
Python, pandas, matplotlib, seaborn

---
### How to explain this in an interview (60 seconds)
"I analyzed retail sales data to find what was actually driving profit,
not just revenue. I found that Technology was the strongest category,
but the more interesting insight was that heavy discounting was quietly
killing margins — profit dropped fast above a 20% discount. I'd recommend
the business cap discounts on low-margin items and focus promotions where
margin can absorb them."
