# Network Traffic Anomaly Detection


I did this project is to compare anomaly detection in network traffic between supervised and unsupervised learning. The primary dataset is CICIDS2017, a dataset from the Canadian Institute for Cybersecurity.

## Goals
- Detect network attacks and anomalies in traffic data
- Compare supervised (Random Forest, XGBoost) and unsupervised (Isolation Forest, Autoencoder) models
- Visualize feature importance, clustering, and attack patterns

## What I did

1. **Data Collection & Cleaning**: Download CICIDS2017, preprocess, and clean the data.
2. **Feature Engineering**: Create and select relevant features, normalize data, and split into train/test sets.
3. **Supervised Modeling**: Train and evaluate Random Forest and XGBoost models.
4. **Unsupervised Modeling**: Train and evaluate Isolation Forest and Autoencoder models.
5. **Visualizations**: Plot feature importance, t-SNE clustering, and time-based attack patterns.

## Folder Structure
- `/data`: Raw and processed datasets
- `/src`: Source code for preprocessing, feature engineering, and modeling
- `/visualizations`: Scripts and outputs for plots
- `/notebooks`: Jupyter notebooks for small tests

## How to run

you do need at least 
Python 3.8 or higher and 
at least 20GB of free disk space (for raw data and processed files).

You have to download the CICIDS2017 dataset yourself since its too big for github (its roughly 800 mb). It has 8 csv files. They have two versions of the data, I used the machine learning csv files.
   - `Monday-WorkingHours.pcap_ISCX.csv`
   - `Tuesday-WorkingHours.pcap_ISCX.csv`
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
   - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

I would just put all 8 CSV files in the `/data` folder 

Then run the following

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

then to train the models, just run `src/supervised_models.py` (for random forest and XGBoost) and `src/unsupervised_models.py` (for autoencoder and isolation forest)

to generate the diagrams run `src/visualizations.py`

Based on this implementation, you should get similar results:

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
