import GPyOpt
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from GPy.kern import Matern52

# Define the objective function, with scaling to smooth variability
def objective_function(config):
    # actual DBMS performance function
    performance = np.sum((config - np.array([5000, 1000, 250, 4, 500]))**2)
    return performance / 1e7  # Scale down the objective value for smoother optimization

# Define bounds for each knob
bounds = [
    {'name': 'buffer_pool_size', 'type': 'continuous', 'domain': (256, 8192)},
    {'name': 'cache_size', 'type': 'continuous', 'domain': (64, 2048)},
    {'name': 'log_file_size', 'type': 'continuous', 'domain': (64, 512)},
    {'name': 'thread_concurrency', 'type': 'continuous', 'domain': (1, 32)},
    {'name': 'query_cache_size', 'type': 'continuous', 'domain': (0, 1024)}
]

# Set up Bayesian Optimization with adjusted parameters
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI',               # Use Expected Improvement
    acquisition_jitter=0.001,             # Lower jitter to reduce exploration
    initial_design_numdata=10,           # Use more initial points
    kernel=Matern52(input_dim=5)         # Use Matern kernel for more flexibility
)

# Run Bayesian Optimization with more iterations and check for convergence
start_time = time.time()
max_iter = 50
convergence_threshold = 0.01  # Convergence threshold (1% difference)
converged = False

for i in range(max_iter):
    optimizer.run_optimization(max_iter=1)
    
    # Check convergence based on last 10 iterations
    if i >= 10:
        recent_values = optimizer.Y[-10:]  # Get the last 10 objective values
        if np.max(recent_values) - np.min(recent_values) < convergence_threshold:
            converged = True
            print(f"Convergence reached after {i + 1} iterations.")
            break

execution_time = time.time() - start_time

# Get the best configuration and objective value
best_config = optimizer.X[np.argmin(optimizer.Y)]
best_objective_value = np.min(optimizer.Y)

# Define knob names for output
knob_names = ["buffer_pool_size", "cache_size", "log_file_size", "thread_concurrency", "query_cache_size"]

# Create a labeled dictionary of the best configuration
best_config_labeled = {knob: value for knob, value in zip(knob_names, best_config)}

# Display important metrics
print("Best configuration found:")
for knob, value in best_config_labeled.items():
    print(f"  {knob}: {value}")

print(f"\nBest Objective Value (Performance): {best_objective_value}")
print(f"Execution Time (seconds): {execution_time}")
print(f"Number of Evaluations: {len(optimizer.Y)}")

# Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(optimizer.Y) + 1), optimizer.Y, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence Plot")
plt.grid(True)
plt.show()
