from scipy.spatial.distance import euclidean
import numpy as np
from data import metrics_data  # Example historical metrics

def find_similar_workload(current_metrics, historical_metrics):
    distances = [euclidean(current_metrics, hist_metrics) for hist_metrics in historical_metrics]
    best_match_index = np.argmin(distances)
    return best_match_index
 
# Test with current metrics 
current_metrics = np.array([48000, 23000, 15000, 2700, 2.0]) 
best_match_index = find_similar_workload(current_metrics, metrics_data)
print("Best matching workload index:", best_match_index)
