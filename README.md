
##  Theoretical Foundations (TFG Structure)

This repository supports the full written structure of the associated Bachelor Thesis:

### 1. Introduction
- Motivation and context  
- Objectives of the project  
- Role of QKD networks in EuroQCI / MADQCI  

### 2. State of the Art
- Fundamentals of QKD and operational metrics (QBER, Key Rate)
- Standardisation efforts (ITU-T Y.38xx, ETSI QKD-ISG)
- Classical network monitoring tools (R, Python, EDA toolkits)
- ML approaches for anomaly detection in QKD (IF, AE, LSTM-AE)
- Control and management layers (SDN, KMS, Prometheus/Grafana)

### 3. Methodology & Dataset Analysis
- Description of QTI (experimental) and Toshiba (operational) datasets  
- Temporal structuring and pre-processing  
- Statistical analysis, stability, kurtosis, skewness  
- Multivariate correlation (QBER ↔ SKR ↔ optical losses)  
- Outlier detection with Z-Score  

### 4. ML-Based Anomaly Detection
- Functional requirements for QKD monitoring  
- Feature engineering for operational metrics  
- Isolation Forest workflow  
- Autoencoder / LSTM Autoencoder architecture and loss functions  

### 5. Evaluation & Results
- ROC curves, thresholds, reconstruction error  
- Comparison of Z-Score vs IF vs Autoencoder  
- Case analysis (network fault vs active attack)  
- Integration perspective (SDN routing, KMS decisions)

### 6. Conclusions & Future Work
- Summary of contributions  
- Limitations and industrial impact  
- Future extensions (real-time monitoring, online learning)

### 7. Impact Analysis
- Technological and national security implications (EUROQCI)  
- Alignment with Sustainable Development Goals (SDG)

---

##  Technologies

### **R**
- `tidyverse`, `xts`, `dplyr`, `ggplot2`
- Used for EDA, dataset merging, statistical characterization  

### **Python**
- `numpy`, `pandas`, `scikit-learn`, `tensorflow`  
- Used for preprocessing, feature engineering, and ML pipelines  

### **Machine Learning**
- **Isolation Forest** (unsupervised anomaly detection)  
- **Autoencoders** (dense & LSTM variants)  

### **Tooling**
- VSCode + RStudio  
- GitHub (dev/main branch workflow)  
- Conda environment  

