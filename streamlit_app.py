import streamlit as st
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from scipy.spatial.distance import euclidean
import GPyOpt
from GPy.kern import Matern52
import matplotlib.pyplot as plt
import time 

import pandas as pd

import pandas as pd

# Load dataset
dataset = pd.read_csv('dbms_tuning_data.csv')

# Extract knob data, metrics data, and performance metric
knob_data = dataset[['buffer_pool_size', 'cache_size', 'log_file_size', 
                     'thread_concurrency', 'query_cache_size']].to_numpy()

metrics_data = dataset[['metric_pages_read', 'metric_pages_written', 'metric_cache_hits', 
                        'metric_cache_misses', 'metric_lock_wait_time']].to_numpy()

performance_metric = dataset['performance_latency_99th_percentile'].to_numpy()


def calculate_latency(config, metrics_data=None):
    """
    Calculate latency using consistent formula from data generation
    Args:
        config: array [buffer_pool_size, cache_size, log_file_size, thread_concurrency, query_cache_size]
        metrics_data: Current metrics including lock_wait_time
    """
    buffer_pool_size = config[0]
    thread_concurrency = config[3]
    
    # If metrics data is not provided, use a default lock_wait_time
    lock_wait_time = 1.0  # default value matching data generation normal distribution mean
    if metrics_data is not None:
        lock_wait_time = metrics_data[4]  # metric_lock_wait_time is the 5th metric
    
    # Add noise consistent with data generation
    noise = np.random.normal(0, 3.9)  # Using same noise distribution as data generation
    
    # Use exact same formula as data generation
    performance_latency = (
        720 / np.log1p(buffer_pool_size) + 
        560 / np.sqrt(max(1, thread_concurrency)) + 
        lock_wait_time * 15 + 
        noise
    )
    
    # Ensure minimum latency of 100ms as in data generation
    return np.clip(performance_latency, 100, None)

# Initialize session state for tracking steps and execution status
if "current_step" not in st.session_state:
    st.session_state.current_step = 0  # 0 for the title screen
if "step_executed" not in st.session_state:
    st.session_state.step_executed = [False] * 4  # False for each step


# Sidebar with dynamic step highlight and navigation
def sidebar_with_dynamic_navigation():
    if st.session_state.current_step == 0:
        st.sidebar.markdown(
            f"""
            <div style="background-color: #FF4B4B; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                HOME
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for i in range(1, 5):
            step_title = [
                "Workload Characterization",
                "Important Knobs Determination",
                "Workload Mapping",
                "Configuration Recommendation",
            ][i - 1]

            if i == st.session_state.current_step:
                # Highlight the current step in red
                st.sidebar.markdown(
                    f"""
                    <div style="background-color: #FF4B4B; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                        Step {i}: {step_title}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Inactive or completed steps
                st.sidebar.markdown(
                    f"""
                    <div style="padding: 10px; margin-bottom: 10px; color: gray; text-align: center;">
                        Step {i}: {step_title}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


sidebar_with_dynamic_navigation()


# Function to mark step as completed and navigate to the next step
def mark_step_completed_and_next():
    st.session_state.step_executed[st.session_state.current_step - 1] = True
    if st.session_state.current_step < 4:
        st.session_state.current_step += 1


# Function to navigate back to the title screen
def go_to_title_screen():
    st.session_state.current_step = 0
    # Reset step execution flags
    st.session_state.step_executed = [False] * 4


# Title Screen
if st.session_state.current_step == 0:
    st.title("Welcome to the DBMS Tuning App")
    st.write("Navigate through the steps sequentially to tune your DBMS.")
    if st.button("Start Now"):
        st.session_state.current_step = 1

# Step 1: Workload Characterization
elif st.session_state.current_step == 1:
    st.title("Step 1: Workload Characterization")
    if not st.session_state.step_executed[0]:
        if st.button("Run Workload Characterization"):
            # Dimensionality Reduction
            fa = FactorAnalysis(n_components=5).fit(metrics_data)
            reduced_metrics = fa.transform(metrics_data)

            # Clustering
            num_clusters = 5
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(reduced_metrics)
            cluster_labels = kmeans.labels_

            # Representative Metrics
            metric_names = [
                "metric_pages_read",
                "metric_pages_written",
                "metric_cache_hits",
                "metric_cache_misses",
                "metric_lock_wait_time",
            ]
            representative_metrics = []
            for cluster in range(num_clusters):
                cluster_points = metrics_data[cluster_labels == cluster]
                centroid = cluster_points.mean(axis=0)
                representative_metrics.append(centroid)

            # Display Results
            st.write("### Representative Metrics for Clusters")
            for i, metrics in enumerate(representative_metrics):
                st.write(f"Cluster {i + 1}:")
                for name, value in zip(metric_names, metrics):
                    st.write(f"  - {name}: {value}")

            st.button("Next", on_click=mark_step_completed_and_next)

# Step 2: Important Knobs Determination
elif st.session_state.current_step == 2:
    st.title("Step 2: Identify Important Knobs")
    if not st.session_state.step_executed[1]:
        if st.button("Run Important Knobs Determination"):
            from xgboost import XGBRegressor
            from sklearn.inspection import permutation_importance
            
            # Feature names for knobs
            knob_names = [
                "buffer_pool_size",
                "cache_size",
                "log_file_size",
                "thread_concurrency",
                "query_cache_size",
            ]

            # Preprocessing knob data
            scaler = StandardScaler()
            knob_data_scaled = scaler.fit_transform(knob_data)

            # Polynomial Features for Lasso Regression
            poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=True)
            knob_data_poly = poly.fit_transform(knob_data_scaled)

            # Lasso Regression
            lasso = Lasso(alpha=0.005, max_iter=50000).fit(knob_data_poly, performance_metric.ravel())
            lasso_important_indices = np.where(lasso.coef_ != 0)[0]
            lasso_important_knobs = [
                poly.get_feature_names_out(knob_names)[i] for i in lasso_important_indices
            ]

            # XGBoost Regression
            xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
            xgb_model.fit(knob_data, performance_metric)

            # Feature importance from XGBoost
            xgb_importances = xgb_model.feature_importances_
            xgb_important_knobs = [
                knob_names[i] for i in np.argsort(xgb_importances)[::-1] if xgb_importances[i] > 0
            ]

            # Display Results
            st.write("### Important Knobs Identified by Lasso Regression")
            for knob in lasso_important_knobs:
                st.write(f"- {knob}")

            st.write("### Important Knobs Identified by XGBoost")
            for knob in xgb_important_knobs:
                st.write(f"- {knob}")

            st.button("Next", on_click=mark_step_completed_and_next)


# Step 3: Workload Mapping
elif st.session_state.current_step == 3:
    st.title("Step 3: Workload Mapping")
    if "current_metrics" not in st.session_state:
        st.session_state.current_metrics = "48000, 23000, 15000, 2700, 2.0"  # Default input

    if not st.session_state.step_executed[2]:
        st.text_input(
            "Enter current metrics (comma-separated):",
            value=st.session_state.current_metrics,
            key="current_metrics_input",  # Tie input to session state
        )

        if st.button("Run Workload Mapping"):
            def find_similar_workload(current_metrics, historical_metrics):
                distances = [
                    euclidean(current_metrics, hist_metrics)
                    for hist_metrics in historical_metrics
                ]
                best_match_index = np.argmin(distances)
                return best_match_index

            try:
                # Parse the input from session state
                current_metrics = np.array(
                    [float(x) for x in st.session_state.current_metrics_input.split(",")]
                )
                best_match_index = find_similar_workload(current_metrics, metrics_data)

                # Display Results
                st.write(f"### Best Matching Workload Index: {best_match_index}")
            except ValueError:
                st.error("Invalid input format. Please enter numeric values separated by commas.")

        st.button("Next", on_click=mark_step_completed_and_next)

# Step 4: Configuration Recommendation
elif st.session_state.current_step == 4:
    st.title("Step 4: Recommend Configuration")
    if not st.session_state.step_executed[3]:
        if st.button("Run Configuration Recommendation"):
            # Get current metrics if available
            try:
                current_metrics = np.array(
                    [float(x) for x in st.session_state.current_metrics_input.split(",")]
                ) if "current_metrics_input" in st.session_state else None
            except:
                current_metrics = np.array([40000, 25000, 15000, 2500, 1.0])  # Default metrics
                st.warning("Using default metrics for optimization")

            def objective_function(config):
                """Objective function that directly minimizes latency"""
                config = np.atleast_2d(config) if len(config.shape) == 1 else config
                latencies = []
                
                for conf in config:
                    latency = calculate_latency(conf, current_metrics)
                    latencies.append(latency)
                
                return np.array(latencies).reshape(-1, 1)

            # Define bounds for each knob
            bounds = [
                {"name": "buffer_pool_size", "type": "continuous", "domain": (1024, 16384)},
                {"name": "cache_size", "type": "continuous", "domain": (512, 8196)},
                {"name": "log_file_size", "type": "continuous", "domain": (64, 512)},
                {"name": "thread_concurrency", "type": "continuous", "domain": (1, 64)},
                {"name": "query_cache_size", "type": "continuous", "domain": (0, 2048)}
            ]

            # Run optimization
            optimizer = GPyOpt.methods.BayesianOptimization(
                f=objective_function,
                domain=bounds,
                acquisition_type="EI",
                acquisition_jitter=0.005,
                initial_design_numdata=15,
                kernel=Matern52(input_dim=5),
            )

            # Progress bar for optimization
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            max_iter = 80
            convergence_threshold = 0.01
            converged = False

            for i in range(max_iter):
                optimizer.run_optimization(max_iter=1)
                progress = (i + 1) / max_iter
                progress_bar.progress(progress)
                status_text.text(f'Optimization progress: {int(progress*100)}%')

                if i >= 5:
                    recent_values = optimizer.Y[-5:]
                    if np.max(recent_values) - np.min(recent_values) < convergence_threshold:
                        converged = True
                        st.write(f"Convergence reached after {i + 1} iterations.")
                        break

            execution_time = time.time() - start_time

            # Get best configuration
            best_config = optimizer.X[np.argmin(optimizer.Y)]

            # Display Best Configuration
            st.write("### Best Configuration Found")
            knob_names = [
                "buffer_pool_size",
                "cache_size", 
                "log_file_size",
                "thread_concurrency",
                "query_cache_size"
            ]
            
            # Create a formatted display of the configuration
            st.write("#### Recommended Knob Settings")
            for knob, value in zip(knob_names, best_config):
                # Format values based on knob type
                if knob in ["buffer_pool_size", "cache_size", "log_file_size"]:
                    st.write(f"- {knob}: {value:.0f} MB")
                elif knob == "thread_concurrency":
                    st.write(f"- {knob}: {value:.0f} threads")
                elif knob == "query_cache_size":
                    st.write(f"- {knob}: {value:.0f} MB")

            # Add a divider before performance metrics
            st.markdown("---")

            # Calculate latencies for distribution
            n_simulations = 1000
            latencies = []
            
            # Progress bar for latency calculations
            latency_progress_bar = st.progress(0)
            latency_status = st.empty()

            for sim in range(n_simulations):
                latencies.append(calculate_latency(best_config, current_metrics))
                progress = (sim + 1) / n_simulations
                latency_progress_bar.progress(progress)
                latency_status.text(f'Calculating latency: {int(progress*100)}%')

            percentile_99 = np.percentile(latencies, 99)
            median_latency = np.median(latencies)

            # Visualization
            st.write("### Performance Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # Convergence Plot
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.plot(range(1, len(optimizer.Y) + 1), optimizer.Y, marker="o", linestyle="-")
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Latency (ms)")
                ax1.set_title("Optimization Progress ")
                ax1.grid(True)
                st.pyplot(fig1)
                
            with col2:
                # Latency Distribution
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.hist(latencies, bins=50, edgecolor='black')
                ax2.axvline(percentile_99, color='r', linestyle='dashed', 
                          label=f'99th Percentile ({percentile_99:.2f}ms)')
                ax2.axvline(median_latency, color='g', linestyle='dashed', 
                          label=f'Median ({median_latency:.2f}ms)')
                ax2.set_xlabel("Latency (ms)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Latency Distribution\n(Tighter is Better)")
                ax2.legend()
                st.pyplot(fig2)

            # Add summary statistics
            st.write("### Performance Summary")
            st.write(f"- Median Latency: {median_latency:.2f}ms")
            st.write(f"- 99th Percentile Latency: {percentile_99:.2f}ms")
            st.write(f"- Latency Range: {min(latencies):.2f}ms - {max(latencies):.2f}ms")
            st.write(f"- Standard Deviation: {np.std(latencies):.2f}ms")

            st.button("Done", on_click=go_to_title_screen)