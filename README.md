# Network Traffic Anomaly Detection

## Project Overview
This project is to compare anomaly detection in network traffic between supervised and unsupervised learning. The primary dataset is CICIDS2017, a dataset from the Canadian Institute for Cybersecurity.

## Goals
- Detect network attacks and anomalies in traffic data
- Compare supervised (Random Forest, XGBoost) and unsupervised (Isolation Forest, Autoencoder) models
- Visualize feature importance, clustering, and attack patterns

## Vibes
1. **Project Setup**: Organize folders, set up environment, and install dependencies.
2. **Data Collection & Cleaning**: Download CICIDS2017, preprocess, and clean the data.
3. **Feature Engineering**: Create and select relevant features, normalize data, and split into train/test sets.
4. **Supervised Modeling**: Train and evaluate Random Forest and XGBoost models.
5. **Unsupervised Modeling**: Train and evaluate Isolation Forest and Autoencoder models.
6. **Visualizations**: Plot feature importance, t-SNE clustering, and time-based attack patterns.

## Folder Structure
- `/data`: Raw and processed datasets
- `/src`: Source code for preprocessing, feature engineering, and modeling
- `/visualizations`: Scripts and outputs for plots
- `/notebooks`: Jupyter notebooks for small tests

## How to run

you do need at least 
Python 3.8 or higher and 
at least 20GB of free disk space (for raw data and processed files).

yes i used an llm to generate this readme file.

### Step 1: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/cxhungg/network_anomaly_detection.git
cd Network_Anomaly

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download CICIDS2017 Dataset

**Important**: The CICIDS2017 dataset is not included in this repository due to size limitations (~800MB). You must download it manually.

1. **Visit the official dataset page**: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

2. **Download the dataset**: Look for "CICIDS2017 Dataset" and download the zip file containing all CSV files.

3. **Extract the files**: Extract the downloaded zip file and locate the following 8 CSV files:
   - `Monday-WorkingHours.pcap_ISCX.csv`
   - `Tuesday-WorkingHours.pcap_ISCX.csv`
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
   - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

4. **Place files in data folder**: Copy all 8 CSV files to the `/data` folder in your project directory.

### Step 3: Data Preprocessing Pipeline

Run the following scripts in order to process the raw data:

```bash
# 1. Merge and label the data
python src/merge_and_label.py

# 2. Clean infinite values and handle missing data
python src/clean_values.py

# 3. Perform feature engineering and create train/test splits
python src/feature_engineering.py
```

**Expected Output Files** (in `/data` folder):
- `merged_data_with_labels.csv` (~5.3GB)
- `cleaned_data_with_labels.csv` (~3.8GB)
- `features.csv` (~8.3GB)
- `train_data.csv` (~6.6GB)
- `test_data.csv` (~1.6GB)
- `scaler.pkl` (~5KB)

### Step 4: Train Supervised Models

```bash
# Train Random Forest and XGBoost models
python src/supervised_models.py
```

**Expected Output Files** (in `/data` folder):
- `random_forest_model.pkl`
- `xgboost_model.pkl` 

**Expected Output Files** (in `/visualizations` folder):
- `roc_curves.png`
- `confusion_matrices.png`
- `feature_importance.png`

### Step 5: Train Unsupervised Models

```bash
# Train Isolation Forest and Autoencoder models
python src/unsupervised_models.py
```

**Expected Output Files** (in `/data` folder):
- `isolation_forest_model.pkl` 
- `autoencoder_model.keras` 
- `autoencoder_threshold.pkl` 

**Expected Output Files** (in `/visualizations` folder):
- `unsupervised_roc_curves.png`
- `unsupervised_confusion_matrices.png`
- `autoencoder_training_history.png`

### Step 6: Generate Visualizations

```bash
# Create visualizations
python src/visualizations.py
```

**Expected Output Files** (in `/visualizations` folder):
- `project_summary.png`
- `feature_importance_comparison.png`
- `random_forest_feature_importance.png`
- `xgboost_feature_importance.png`
- `tsne_clustering.png`
- `pca_analysis.png`
- `model_performance_roc.png`
- `model_performance_table.png`
- `feature_distributions.png`
- `correlation_heatmap.png`
- `supervised_vs_unsupervised_comparison.png`

## Expected Results

Based on this implementation, you should achieve similar results:

### Supervised Models
- **XGBoost**: 96.69% AUC, 92.82% Accuracy
- **Random Forest**: 96.33% AUC, 92.66% Accuracy

### Unsupervised Models
- **Autoencoder**: 87.14% AUC, 51.99% Accuracy
- **Isolation Forest**: 41.62% AUC, 34.11% Accuracy

## Output Files

### Data Files (`/data`) 
- Raw CSV files (8 files, ~800 MB total)
- Processed datasets (5 files, ~26.4 GB total)
- Trained models (4 files, ~8 MB total)
- Preprocessing files (1 file, ~4 KB)

### Visualization Files (`/visualizations`)

- Model performance comparisons
- Feature importance analysis
- Data distribution visualizations
- Clustering and dimensionality analysis

## References
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
