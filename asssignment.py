import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


np.random.seed(42)


def generate_data(n=100):
    patient_ids = np.arange(1, n + 1)
    prescription_intervals = np.random.randint(20, 60, size=n)  
    refill_gaps = np.random.randint(0, 20, size=n)  
    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Interval_Days": prescription_intervals,
        "Refill_Gap": refill_gaps
    })
    df["Days_Covered"] = df["Interval_Days"]  
    return df


def compute_see(df):
    df["SEE"] = df["Days_Covered"] / (df["Days_Covered"] + df["Refill_Gap"])
    return df


def standardize_data(df):
    scaler = StandardScaler()
    df["Interval_Days_Scaled"] = scaler.fit_transform(df[["Interval_Days"]])
    return df, scaler


def apply_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(df[["Interval_Days_Scaled"]])
    return df, kmeans


def apply_dbscan(df, eps=0.7, min_samples=3):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["DBSCAN_Cluster"] = dbscan.fit_predict(df[["Interval_Days_Scaled"]])
    return df, dbscan


def plot_see_vs_clusters(df, cluster_col, title):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[cluster_col], y=df["SEE"], palette="Set2")
    plt.title(f"SEE Scores Across {title}")
    plt.xlabel(cluster_col)
    plt.ylabel("Sessa Empirical Estimator (SEE)")
    plt.show()


def compare_see_clusters(df):
    if "KMeans_Cluster" in df.columns and "DBSCAN_Cluster" in df.columns:
        comparison = df.groupby(["KMeans_Cluster", "DBSCAN_Cluster"])["SEE"].mean().reset_index()
        print("\nSEE Cluster Comparison:\n", comparison)
        return comparison
    else:
        print("Error: Clustering columns are missing.")
        return None


df = generate_data()
df = compute_see(df)
df, scaler = standardize_data(df)
df, kmeans = apply_kmeans(df)
df, dbscan = apply_dbscan(df)


plot_see_vs_clusters(df, "KMeans_Cluster", "K-Means Clustering")
plot_see_vs_clusters(df, "DBSCAN_Cluster", "DBSCAN Clustering")


compare_see_clusters(df)
