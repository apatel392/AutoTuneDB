from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
import numpy as np
from data import metrics_data

# Dimensionality Reduction
fa = FactorAnalysis(n_components=5).fit(metrics_data)  # Adjust n_components as needed
reduced_metrics = fa.transform(metrics_data)

# Clustering 
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(reduced_metrics)
cluster_labels = kmeans.labels_

# Define metric names
metric_names = [
    "metric_pages_read",
    "metric_pages_written",
    "metric_cache_hits",
    "metric_cache_misses",
    "metric_lock_wait_time"
]

# Select representative metrics for each cluster
representative_metrics = []
for cluster in range(num_clusters):
    # Get the points in the current cluster
    cluster_points = metrics_data[cluster_labels == cluster]
    
    # Calculate the centroid of the cluster in original metric space
    centroid = cluster_points.mean(axis=0)
    
    # Append the centroid as the representative metrics for this cluster
    representative_metrics.append(centroid)

# Print each cluster's representative metrics with labels
for i, metrics in enumerate(representative_metrics):
    print(f"\nCluster {i + 1} Representative Metrics:")
    for name, value in zip(metric_names, metrics):
        print(f"  {name}: {value}")
    print()  # Blank line for readability between clusters
