# AutoTuneDB: Machine Learning-Based DBMS Configuration Tuning

A capstone project that automates the tuning of database management system (DBMS) configurations using machine learning techniques. The system is inspired by OtterTune and enhanced with advanced methods including clustering, regression, and Bayesian optimization to dynamically optimize DBMS performance across diverse workloads.

## Problem Statement

Modern DBMSs expose hundreds of tunable parameters ("knobs") that significantly influence system performance. Manual tuning is error-prone, time-consuming, and inefficient—especially for complex workloads. This project aims to automate DBMS configuration tuning using data-driven approaches to:
- Characterize workloads
- Identify critical knobs
- Recommend optimal configurations
- Continuously adapt to changing workloads

## Key Features

- **Workload Clustering** using Factor Analysis and k-means  
- **Knob Impact Analysis** via Lasso Regression and XGBoost  
- **Bayesian Optimization** with Gaussian Process Regression for configuration recommendations  
- **Dynamic Feedback Loop** to handle real-time workload changes  
- **Performance Dashboard** (UI) for monitoring and insights  

## System Architecture

```
                   ┌────────────────────┐
                   │   Data Collection  │
                   └────────┬───────────┘
                            ↓
         ┌───────────────────────────────┐
         │ Workload Characterization     │
         │ (Factor Analysis + K-means)   │
         └────────┬──────────────────────┘
                  ↓
      ┌────────────────────────────┐
      │   Knob Impact Analysis     │
      │ (Lasso Regression, XGBoost)│
      └────────┬───────────────────┘
               ↓
     ┌─────────────────────────────┐
     │ Configuration Optimization  │
     │ (Bayesian Opt + GPs)        │
     └────────┬────────────────────┘
              ↓
       ┌─────────────────────────┐
       │     Feedback Loop       │
       └─────────────────────────┘
```

## Methodology

### 1. Workload Characterization
- Reduce dimensionality with **Factor Analysis**
- Cluster workloads using **k-means**

### 2. Knob Impact Analysis
- Identify critical knobs using:
  - **Lasso Regression** for sparse linear relationships
  - **XGBoost** for capturing non-linear interactions

### 3. Configuration Optimization
- Leverage **Bayesian Optimization** with Gaussian Process Regression to efficiently search the configuration space

### 4. Adaptive Feedback Loop
- Continuously monitors performance metrics to update configurations in real-time


## System Requirements

### 1. OS
- Windows 10 or later  
- MacOS 10.15+  
- Ubuntu 18.04 or later  

### 2. Python
- Python 3.8+

### 3. Libraries
Install dependencies:
```bash
pip install -r requirements.txt
```

Includes:
- streamlit (1.25+)
- numpy (1.26+)
- scikit-learn (1.3+)
- matplotlib (3.7+)
- GPy (1.13.2)
- GPyOpt (1.2.6)

If needed:
```bash
pip install GPy
pip install GPyOpt
pip install pandas
```

## How to Run

### Option 1: Run Full Streamlit App
```bash
cd path/to/project
```
```bash
streamlit run streamlit_app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

### Option 2: Run Modules Individually
```bash
# Workload Characterization
python workload_characterization.py

# Important Knobs Determination
python knob_importance.py

# Workload Mapping
python workload_mapping.py

# Configuration Recommendation
python configuration_recommendation.py
```

## Step-by-Step Instructions

1. **Workload Characterization**  
   Input metrics or use defaults → Run → View clustered workloads

2. **Important Knobs Determination**  
   Click Run → View top knobs and interactions

3. **Workload Mapping**  
   Enter new metrics → Map → See best matching workload index

4. **Configuration Recommendation**  
   Run → View recommended config, convergence plot → Return or restart

## Evaluation Highlights

- **Latency reduced** from ~450ms to ~165ms within 20 iterations  
- **99th percentile latency**: 175.33ms  
- **Standard deviation**: 3.88ms — indicating consistent performance  
- Demonstrated scalability and adaptability across read-heavy workloads

## Future Improvements

- Add **multi-objective optimization** (e.g., energy efficiency)  
- Experiment with **Reinforcement Learning** for continuous adaptation  
- Integrate **TPC-H benchmark** for robust evaluations  

## Tech Stack

- Python (scikit-learn, XGBoost, GPy)  
- PostgreSQL (simulated workloads)  
- Jupyter Notebooks / Scripts  
- Optional: Cloud compute resources for optimization

## Author

**Akshar Patel**  
Graduate Student, Georgia State University  
Email: [apatel392@student.gsu.edu](mailto:apatel392@student.gsu.edu)  
GitHub: [@apatel392](https://github.com/apatel392)

## Troubleshooting

- **Missing packages?**  
  Run: `pip install -r requirements.txt`

- **ModuleNotFoundError for GPyOpt or Streamlit?**  
  Run:
  ```bash
  pip install GPy
  pip install GPyOpt
  pip install streamlit
  ```


Feel free to open an issue or reach out via email if you need help or want to collaborate!
