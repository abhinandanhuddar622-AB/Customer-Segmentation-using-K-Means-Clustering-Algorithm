# 🛍️ Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Segment mall customers into meaningful groups based on **Age**, **Annual Income**, and **Spending Score** using K-Means, Hierarchical, and DBSCAN clustering — enabling targeted marketing strategies.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Business Insights](#business-insights)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 📖 Overview

Businesses often treat all customers the same, wasting marketing resources. This project uses **unsupervised machine learning** to automatically group 200 mall customers into 5 distinct segments — helping businesses:

- Identify high-value customers
- Design targeted marketing campaigns
- Increase sales and customer satisfaction
- Make data-driven decisions

---

## 📊 Dataset

**File:** `Mall_Customers.csv`  
**Source:** [Kaggle — Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
**Records:** 200 customers

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Customer age (years) |
| `Annual_Income` | Annual income in $k |
| `Spending_Score` | Mall-assigned score (1–100) |

---

## 📁 Project Structure

```
customer-segmentation/
│
├── customer_segmentation_kmeans.py   # Main project script
├── Mall_Customers.csv                # Dataset
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
│
└── outputs/                          # Auto-generated plots
    ├── eda_plots.png
    ├── gender_dist.png
    ├── elbow_method.png
    ├── clusters_income_spending.png
    ├── clusters_age_spending.png
    └── cluster_sizes.png
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations |
| Scikit-learn | K-Means, DBSCAN, Hierarchical clustering |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/abhinandanhuddar622-AB/Customer-Segmentation-using-K-Means-Clustering-Algorithm.git
cd customer-segmentation
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python customer_segmentation_kmeans.py
```

The script will:
1. Load and preprocess the dataset
2. Generate EDA plots
3. Run the Elbow Method to find optimal K
4. Train K-Means with K=5
5. Output cluster summary and business insights
6. Save all visualizations to the current directory

### Predict segment for a new customer
```python
predict_segment(age=28, annual_income=70, spending_score=80)
# Output: Age:28 | Income:70k | Score:80 → Cluster 3 (High Income, High Spend)
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Optimal Clusters (K) | **5** |
| Silhouette Score (K-Means) | **0.417** |
| Silhouette Score (Hierarchical) | **0.390** |
| DBSCAN Clusters Found | **6** |
| Total Customers | **200** |

### Cluster Summary

| Cluster | Avg Age | Avg Income | Avg Spending | Type |
|---|---|---|---|---|
| 1 | 46 | $27k | 18 | Low Income, Low Spend |
| 2 | 25 | $41k | 62 | Young Moderate Spenders |
| 3 | 33 | $86k | 82 | High Income, High Spend ⭐ |
| 4 | 40 | $86k | 19 | High Income, Low Spend |
| 5 | 56 | $54k | 49 | Moderate Customers |

---

## 🖼️ Visualizations

| Plot | Description |
|---|---|
| `eda_plots.png` | Distribution of Age, Income, Spending Score |
| `elbow_method.png` | WCSS vs K — find optimal clusters |
| `clusters_income_spending.png` | Scatter: Income vs Spending (5 clusters) |
| `clusters_age_spending.png` | Scatter: Age vs Spending (5 clusters) |
| `cluster_sizes.png` | Bar chart of customers per cluster |

---

## 💡 Business Insights

| Cluster | Segment | Strategy |
|---|---|---|
| 1 | Low Income, Low Spend | Budget deals & discount coupons |
| 2 | Young Moderate Spenders | Social media campaigns |
| 3 | High Income, High Spend | Premium loyalty programs ⭐ |
| 4 | High Income, Low Spend | Re-engagement & exclusive offers |
| 5 | Moderate Customers | Standard seasonal promotions |

---

## 🚀 Future Enhancements

- [ ] DBSCAN and Hierarchical clustering deep analysis
- [ ] Real-time customer segmentation via web app (Flask/Streamlit)
- [ ] Integration with CRM systems
- [ ] AI-based product recommendation per segment
- [ ] Dashboard using Power BI / Tableau

---

## 🙋 Author

Abhinandan V Huddar  
📧 abhinandanhuddar622@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/abhinandan-v-huddar-79a29937a) | [GitHub](https://github.com/abhinandanhuddar622-AB)

---

> ⭐ If you found this project helpful, please give it a star!
