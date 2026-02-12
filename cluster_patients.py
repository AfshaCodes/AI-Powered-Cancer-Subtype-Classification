import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import sys
import os

# Import VAE model
sys.path.append('models')
from vae_model import VAE

print("=" * 70)
print("PATIENT CLUSTERING USING VAE LATENT SPACE")
print("=" * 70)

# ===== LOAD TRAINED MODEL =====
print("\n[1/6] Loading trained VAE model...")

device = torch.device('cpu')
model = VAE(input_dim=5000, latent_dim=50)
model.load_state_dict(torch.load('models/vae_best.pth', map_location=device))
model = model.to(device)
model.eval()

print("✓ Model loaded successfully")

# ===== LOAD DATA =====
print("\n[2/6] Loading data...")

X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
train_ids = np.load('data/train_patient_ids.npy', allow_pickle=True)  # FIX
test_ids = np.load('data/test_patient_ids.npy', allow_pickle=True)    # FIX

# Combine for full analysis
X_all = np.vstack([X_train, X_test])
patient_ids_all = np.concatenate([train_ids, test_ids])

print(f"✓ Total patients: {X_all.shape[0]}")
print(f"✓ Features per patient: {X_all.shape[1]}")

# ===== EXTRACT LATENT REPRESENTATIONS =====
print("\n[3/6] Extracting latent representations...")

X_tensor = torch.FloatTensor(X_all).to(device)

with torch.no_grad():
    mu, logvar = model.encode(X_tensor)
    latent_features = mu.cpu().numpy()  # Use mean of latent distribution

print(f"✓ Latent shape: {latent_features.shape}")
print(f"✓ Compressed from {X_all.shape[1]} → {latent_features.shape[1]} dimensions")

# Save latent features
np.save('data/latent_features.npy', latent_features)
print("✓ Saved: data/latent_features.npy")

# ===== DETERMINE OPTIMAL NUMBER OF CLUSTERS =====
print("\n[4/6] Determining optimal number of clusters...")

# Test different numbers of clusters
k_range = range(2, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent_features)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(latent_features, labels))
    davies_bouldin_scores.append(davies_bouldin_score(latent_features, labels))

# Plot elbow curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elbow method
axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

# Silhouette score (higher is better)
axes[1].plot(k_range, silhouette_scores, 'o-', color='green', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (Higher = Better)')
axes[1].grid(alpha=0.3)

# Davies-Bouldin score (lower is better)
axes[2].plot(k_range, davies_bouldin_scores, 'o-', color='red', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Score (Lower = Better)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/cluster_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved: results/cluster_optimization.png")
plt.close()

# Recommend optimal k
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal k (by Silhouette): {optimal_k_silhouette}")
print(f"  Silhouette scores: {dict(zip(k_range, [f'{s:.3f}' for s in silhouette_scores]))}")

# ===== PERFORM FINAL CLUSTERING =====
print(f"\n[5/6] Performing K-Means clustering with k={optimal_k_silhouette}...")

optimal_k = optimal_k_silhouette  # You can change this manually if needed
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(latent_features)

# Count patients per cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster sizes:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} patients ({count/len(cluster_labels)*100:.1f}%)")

# Save clustering results
results_df = pd.DataFrame({
    'patient_id': patient_ids_all,
    'cluster': cluster_labels
})
results_df.to_csv('results/patient_clusters.csv', index=False)
print("\n✓ Saved: results/patient_clusters.csv")

# ===== VISUALIZE WITH UMAP =====
print("\n[6/6] Creating UMAP visualization...")

# Reduce to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
latent_2d = reducer.fit_transform(latent_features)

# Save 2D coordinates
np.save('data/latent_2d_umap.npy', latent_2d)

# Create beautiful visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Colored by cluster
scatter1 = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                           c=cluster_labels, cmap='tab10', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('UMAP 1', fontsize=12)
axes[0].set_ylabel('UMAP 2', fontsize=12)
axes[0].set_title(f'Patient Clusters (k={optimal_k})', fontsize=14, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Add cluster centers
centers_2d = reducer.transform(kmeans.cluster_centers_)
axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], 
                marker='X', s=300, c='red', edgecolors='black', linewidth=2,
                label='Cluster Centers', zorder=5)
axes[0].legend()

# Plot 2: Density plot
from scipy.stats import gaussian_kde
xy = latent_2d.T
z = gaussian_kde(xy)(xy)
scatter2 = axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                           c=z, cmap='viridis', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('UMAP 1', fontsize=12)
axes[1].set_ylabel('UMAP 2', fontsize=12)
axes[1].set_title('Patient Distribution Density', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Density')

plt.tight_layout()
plt.savefig('results/umap_visualization.png', dpi=200, bbox_inches='tight')
print("✓ Plot saved: results/umap_visualization.png")
plt.close()

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✅ CLUSTERING COMPLETE!")
print("=" * 70)
print(f"\nIdentified {optimal_k} distinct cancer subtypes")
print(f"Silhouette score: {silhouette_scores[optimal_k-2]:.3f}")
print(f"\n📁 Files created:")
print("  - data/latent_features.npy (50D latent representations)")
print("  - data/latent_2d_umap.npy (2D UMAP coordinates)")
print("  - results/patient_clusters.csv (cluster assignments)")
print("  - results/cluster_optimization.png (k selection)")
print("  - results/umap_visualization.png (cluster visualization)")
print("\n🎯 Next step: Analyze clinical differences between clusters!")