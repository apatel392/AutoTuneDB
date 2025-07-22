from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
import numpy as np
from data import knob_data, performance_metric

# Define knob names and include an intercept term to match feature count
knob_names = ["intercept", "buffer_pool_size", "cache_size", "log_file_size", "thread_concurrency", "query_cache_size"]

# Scale knob data to improve convergence
scaler = StandardScaler()
knob_data_scaled = scaler.fit_transform(knob_data)

# Generate Polynomial Features with a higher degree if needed
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=True)
knob_data_poly = poly.fit_transform(knob_data_scaled)

# Lasso Regression for Feature Selection with increased iterations
lasso = Lasso(alpha=0.001, max_iter=50000).fit(knob_data_poly, performance_metric.ravel())
important_knobs_indices = np.where(lasso.coef_ != 0)[0]

# Map indices to knob names (or interaction terms)
important_knobs = [poly.get_feature_names_out(knob_names)[i] for i in important_knobs_indices]
print("Important knobs selected:", important_knobs)
