

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage

# Step 2: Load Dataset (Iris Dataset)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

print("First 5 Rows of Dataset:")
print(X.head())

# Step 3: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 4: Elbow Method to find Optimal K


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 5: Apply K-Means Clustering


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

X['Cluster'] = clusters

print("\nClustered Data Sample:")
print(X.head())

# Visualization of Clusters
plt.figure(figsize=(6,4))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Step 6: Hierarchical Clustering


linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(16,10))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()