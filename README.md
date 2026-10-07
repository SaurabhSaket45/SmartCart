# SmartCart: Customer Segmentation

Unsupervised machine learning project that groups supermarket customers into segments based on their demographics, income, and purchasing behavior, so marketing can be targeted instead of one-size-fits-all.

## Overview
- **Dataset:** 2,240 customers, 22 raw features (income, education, household, product spending, purchase channels, web visits, campaign response)
- **Goal:** Identify distinct customer groups and profile them by income and spending patterns

## Workflow
1. **Data cleaning:** filled missing income values with the median; removed outliers (age > 90, income > 600,000)
2. **Feature engineering:** created `Age`, `Customer_Tenure`, `Total_Spending`, `Total_Children`, and `Living_With` (simplified from marital status); grouped education into 3 levels
3. **Preprocessing:** one-hot encoded categorical features and standardized all features
4. **Dimensionality reduction:** PCA down to 3 components for visualization and clustering
5. **Choosing K:** elbow method (with KneeLocator) and silhouette score, which pointed to **K = 4**
6. **Clustering:** K-Means and Agglomerative (Ward linkage), both with 4 clusters
7. **Cluster profiling:** compared segments by income, total spending, and purchase behavior

## Key Findings
- Customers split into 4 segments, with two higher-income groups (~$71-73K average) and two lower-income groups (~$37-40K average)
- Higher-income segments show more web purchases and fewer deal purchases than lower-income ones

## Tech Stack
Python, pandas, NumPy, scikit-learn, kneed, matplotlib, seaborn

## How to Run
```bash
git clone <your-repo-url>
cd smartcart
pip install pandas matplotlib seaborn scikit-learn kneed
jupyter lab SmartCart.ipynb
```

## Possible Improvements
- Name each segment and add targeted marketing recommendations
- Try DBSCAN or Gaussian Mixture Models
- Wrap the model in a small web app for live segment prediction
