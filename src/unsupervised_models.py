import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import machine learning libraries
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def load_data():

    data_dir = Path('../data')

    train_data = pd.read_csv(data_dir / 'train_data.csv')
    test_data = pd.read_csv(data_dir / 'test_data.csv')
    
    feature_columns = [col for col in train_data.columns if col not in ['is_attack', ' Label']]
    X_train = train_data[feature_columns]
    y_train = train_data['is_attack']
    X_test = test_data[feature_columns]
    y_test = test_data['is_attack']
    
    
    return X_train, y_train, X_test, y_test

def train_isolation_forest(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("ISOLATION FOREST MODEL")
    print("="*60)
    
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]
    
    print(f"Training on {X_train_benign.shape[0]:,} benign samples")
    

    iso_forest = IsolationForest(
        contamination=0.1, 
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )

    iso_forest.fit(X_train_benign)
    
    # since isolation Forest returns -1 for anomalies, 1 for normal
    y_pred_iso = iso_forest.predict(X_test)
    y_pred = (y_pred_iso == -1).astype(int)  # convert to 1 = anomaly
    
    #negative values = more abnomal
    y_scores = iso_forest.score_samples(X_test)
    # Convert to positive scores , higher = more abnomal
    y_scores = -y_scores
    
    accuracy = (y_pred == y_test).mean()
    auc_score = roc_auc_score(y_test, y_scores)
    
    print(f"\nIsolation Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    

    # print(f"\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    model_path = Path('../data/isolation_forest_model.pkl')
    joblib.dump(iso_forest, model_path)
    
    return iso_forest, y_pred, y_scores

def build_autoencoder(input_dim, encoding_dim=32):


    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def train_autoencoder(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("AUTOENCODER MODEL")
    print("="*60)
    
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]
    
    print(f"Training on {X_train_benign.shape[0]:,} benign samples")
    
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim, encoding_dim=32)
    
    print("Autoencoder:")
    autoencoder.summary()
    
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # train on benign data
    history = autoencoder.fit(
        X_train_benign, X_train_benign,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    X_test_reconstructed = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
    

    X_train_benign_reconstructed = autoencoder.predict(X_train_benign)
    benign_mse = np.mean(np.power(X_train_benign - X_train_benign_reconstructed, 2), axis=1)
    threshold = np.percentile(benign_mse, 95)  # 95th percentile as threshold
    
    #print(f"Reconstruction error threshold: {threshold}")
    
    y_pred = (mse > threshold).astype(int)
    y_scores = mse  # use reconstruction error as anomaly score
    
    accuracy = (y_pred == y_test).mean()
    auc_score = roc_auc_score(y_test, y_scores)
    
    print(f"\nAutoencoder Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    # print(f"\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    model_path = Path('../data/autoencoder_model.keras')
    autoencoder.save(model_path, save_format='keras')

    threshold_path = Path('../data/autoencoder_threshold.pkl')
    joblib.dump(threshold, threshold_path)
    
    return autoencoder, y_pred, y_scores, threshold, history

def plot_unsupervised_results(iso_results, auto_results, X_test, y_test):

    print("\n" + "="*60)
    print("CREATING UNSUPERVISED MODEL PLOTS")
    print("="*60)

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    iso_fpr, iso_tpr, _ = roc_curve(y_test, iso_results[2])
    iso_auc = roc_auc_score(y_test, iso_results[2])
    plt.plot(iso_fpr, iso_tpr, label=f'Isolation Forest (AUC = {iso_auc:.3f})')

    auto_fpr, auto_tpr, _ = roc_curve(y_test, auto_results[2])
    auto_auc = roc_auc_score(y_test, auto_results[2])
    plt.plot(auto_fpr, auto_tpr, label=f'Autoencoder (AUC = {auto_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Unsupervised Models - ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'unsupervised_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cm_iso = confusion_matrix(y_test, iso_results[1])
    sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Isolation Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_auto = confusion_matrix(y_test, auto_results[1])
    sns.heatmap(cm_auto, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Autoencoder Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'unsupervised_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


    if len(auto_results) > 4: 
        history = auto_results[4]
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'autoencoder_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def compare_supervised_unsupervised():
    print("\n" + "="*60)
    print("SUPERVISED vs UNSUPERVISED COMPARISON")
    print("="*60)
    
    try:

        rf_model = joblib.load('../data/random_forest_model.pkl')
        xgb_model = joblib.load('../data/xgboost_model.pkl')
        

        test_data = pd.read_csv('../data/test_data.csv')
        feature_columns = [col for col in test_data.columns if col not in ['is_attack', ' Label']]
        X_test = test_data[feature_columns]
        y_test = test_data['is_attack']
        

        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
        
        print(f"Supervised Models AUC Scores:")
        print(f"Random Forest: {rf_auc:.4f}")
        print(f"XGBoost: {xgb_auc:.4f}")

        plots_dir = Path('../visualizations')
        plots_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(12, 8))

        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)
        
        plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)
        plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})', linewidth=2)
    
        try:
            iso_model = joblib.load('../data/isolation_forest_model.pkl')
            auto_model = None
            try:
                auto_model = keras.models.load_model('../data/autoencoder_model.keras')
            except:
                try:
                    auto_model = keras.models.load_model('../data/autoencoder_model.h5')
                except:
                    print("could not load autoencoder model")
            
            threshold = joblib.load('../data/autoencoder_threshold.pkl')
            
            iso_scores = -iso_model.score_samples(X_test)
            iso_fpr, iso_tpr, _ = roc_curve(y_test, iso_scores)
            iso_auc = roc_auc_score(y_test, iso_scores)
            plt.plot(iso_fpr, iso_tpr, label=f'Isolation Forest (AUC = {iso_auc:.3f})', linewidth=2)
            
            if auto_model is not None:
                X_test_reconstructed = auto_model.predict(X_test)
                auto_scores = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
                auto_fpr, auto_tpr, _ = roc_curve(y_test, auto_scores)
                auto_auc = roc_auc_score(y_test, auto_scores)
                plt.plot(auto_fpr, auto_tpr, label=f'Autoencoder (AUC = {auto_auc:.3f})', linewidth=2)
            
        except FileNotFoundError as e:
            print(f"unsupervised models not found: {e}")
        except Exception as e:
            print(f"Error: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Supervised vs Unsupervised Models - ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'supervised_vs_unsupervised_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except FileNotFoundError:
        print("supervised models not found")

def main():

    
    X_train, y_train, X_test, y_test = load_data()

    iso_results = train_isolation_forest(X_train, y_train, X_test, y_test)
    

    auto_results = train_autoencoder(X_train, y_train, X_test, y_test)

    plot_unsupervised_results(iso_results, auto_results, X_test, y_test)
    compare_supervised_unsupervised()

if __name__ == "__main__":
    main() 