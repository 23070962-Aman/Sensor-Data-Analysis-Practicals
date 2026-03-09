# =========================================
# Practical: Principal Component Analysis
# =========================================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (Reduce to 2 Dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Explained Variance Ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Step 6: Total Variance Preserved
total_variance = sum(pca.explained_variance_ratio_)
print("Total Variance Preserved:", total_variance)

# Step 7: Create DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Step 8: Visualization
plt.figure(figsize=(8,6))

for target in pca_df['Target'].unique():
    subset = pca_df[pca_df['Target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], label=target)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction")
plt.legend()
plt.show()