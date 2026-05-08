# ============================================================
# Customer Segmentation using K-Means Clustering
# Dataset: Mall_Customers.csv
# Tech Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("Mall_Customers.csv")
df.columns = ["Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score"]

print("── Dataset Sample ──")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
df["Gender_Code"] = LabelEncoder().fit_transform(df["Gender"])  # Male=1, Female=0

features = ["Age", "Annual_Income", "Spending_Score"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled successfully.\n")

# ─────────────────────────────────────────────
# 3. EDA — Visualisation
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Exploratory Data Analysis", fontsize=14)

df["Age"].hist(ax=axes[0], color="#378ADD", bins=15, edgecolor="white")
axes[0].set_title("Age Distribution")

df["Annual_Income"].hist(ax=axes[1], color="#1D9E75", bins=15, edgecolor="white")
axes[1].set_title("Annual Income Distribution")

df["Spending_Score"].hist(ax=axes[2], color="#7F77DD", bins=15, edgecolor="white")
axes[2].set_title("Spending Score Distribution")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=120); plt.show()

# Gender count
plt.figure(figsize=(4, 3))
df["Gender"].value_counts().plot(kind="bar", color=["#E24B4A","#378ADD"], edgecolor="white")
plt.title("Gender Distribution"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("gender_dist.png", dpi=120); plt.show()

# ─────────────────────────────────────────────
# 4. ELBOW METHOD — Find Optimal K
# ─────────────────────────────────────────────
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(K_range, wcss, "o-", color="#E24B4A", linewidth=2, markersize=7)
plt.axvline(x=5, color="#1D9E75", linestyle="--", label="Optimal K=5")
plt.title("Elbow Method — Optimal K"); plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS"); plt.legend(); plt.tight_layout()
plt.savefig("elbow_method.png", dpi=120); plt.show()
print("Elbow Method: Optimal K = 5\n")

# ─────────────────────────────────────────────
# 5. APPLY K-MEANS CLUSTERING (K=5)
# ─────────────────────────────────────────────
K = 5
kmeans = KMeans(n_clusters=K, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, df["Cluster"])
print(f"K-Means trained | Silhouette Score: {score:.3f}\n")

# Cluster summary
print("── Cluster Summary ──")
summary = df.groupby("Cluster")[["Age","Annual_Income","Spending_Score"]].mean().round(1)
summary["Count"] = df["Cluster"].value_counts().sort_index()
print(summary, "\n")

# ─────────────────────────────────────────────
# 6. CLUSTER VISUALISATION
# ─────────────────────────────────────────────
colors = ["#E24B4A","#1D9E75","#378ADD","#BA7517","#7F77DD"]
labels = ["Low Income\nLow Spend","High Income\nHigh Spend",
          "Low Income\nHigh Spend","High Income\nLow Spend","Moderate"]

# Plot 1: Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
for i in range(K):
    cluster_data = df[df["Cluster"] == i]
    plt.scatter(cluster_data["Annual_Income"], cluster_data["Spending_Score"],
                c=colors[i], label=f"Cluster {i+1}: {labels[i]}", s=70, alpha=0.8)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 1], centroids[:, 2], s=250, c="black", marker="*", label="Centroids", zorder=5)
plt.title("Customer Segments — Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1-100)")
plt.legend(loc="upper left", fontsize=8); plt.tight_layout()
plt.savefig("clusters_income_spending.png", dpi=120); plt.show()

# Plot 2: Age vs Spending Score
plt.figure(figsize=(8, 5))
for i in range(K):
    cluster_data = df[df["Cluster"] == i]
    plt.scatter(cluster_data["Age"], cluster_data["Spending_Score"],
                c=colors[i], label=f"Cluster {i+1}", s=70, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 2], s=250, c="black", marker="*", zorder=5)
plt.title("Customer Segments — Age vs Spending Score")
plt.xlabel("Age"); plt.ylabel("Spending Score"); plt.legend()
plt.tight_layout(); plt.savefig("clusters_age_spending.png", dpi=120); plt.show()

# Plot 3: Cluster size bar chart
plt.figure(figsize=(6, 3))
df["Cluster"].value_counts().sort_index().plot(
    kind="bar", color=colors, edgecolor="white")
plt.title("Customers per Cluster"); plt.xlabel("Cluster"); plt.ylabel("Count")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("cluster_sizes.png", dpi=120); plt.show()

# ─────────────────────────────────────────────
# 7. FUTURE ENHANCEMENT — Hierarchical & DBSCAN
# ─────────────────────────────────────────────
hier = AgglomerativeClustering(n_clusters=5)
df["Hier_Cluster"] = hier.fit_predict(X_scaled)
h_score = silhouette_score(X_scaled, df["Hier_Cluster"])
print(f"Hierarchical Clustering Silhouette Score: {h_score:.3f}")

dbscan = DBSCAN(eps=0.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(df["DBSCAN_Cluster"])) - (1 if -1 in df["DBSCAN_Cluster"].values else 0)
print(f"DBSCAN found {n_clusters_db} clusters\n")

# ─────────────────────────────────────────────
# 8. PREDICT SEGMENT FOR NEW CUSTOMER
# ─────────────────────────────────────────────
def predict_segment(age, annual_income, spending_score):
    data = scaler.transform([[age, annual_income, spending_score]])
    cluster = kmeans.predict(data)[0]
    print(f"Age:{age} | Income:{annual_income}k | Score:{spending_score} → Cluster {cluster+1} ({labels[cluster]})")

print("── Predicting New Customer Segment ──")
predict_segment(28, 70, 80)    # Young high earner, high spender
predict_segment(55, 25, 20)    # Older low income, low spender
predict_segment(35, 55, 50)    # Moderate customer

# ─────────────────────────────────────────────
# 9. BUSINESS INSIGHTS
# ─────────────────────────────────────────────
print("\n── Business Insights ──")
insights = {
    0: "Low Income, Low Spend   → Budget deals & discounts",
    1: "High Income, High Spend → Premium loyalty programs",
    2: "Low Income, High Spend  → Upsell opportunities",
    3: "High Income, Low Spend  → Re-engagement campaigns",
    4: "Moderate Customers      → Standard promotions",
}
for k, v in insights.items():
    print(f"  Cluster {k+1}: {v}")

print("\nDone. All plots saved.")
# Outputs: eda_plots.png, gender_dist.png, elbow_method.png,
#          clusters_income_spending.png, clusters_age_spending.png, cluster_sizes.png
