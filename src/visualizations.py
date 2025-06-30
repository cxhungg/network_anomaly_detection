import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_and_models():
    

    data_dir = Path('../data')
    train_data = pd.read_csv(data_dir / 'train_data.csv')
    test_data = pd.read_csv(data_dir / 'test_data.csv')

    feature_columns = [col for col in train_data.columns if col not in ['is_attack', ' Label']]
    X_train = train_data[feature_columns]
    y_train = train_data['is_attack']
    X_test = test_data[feature_columns]
    y_test = test_data['is_attack']
 
    models = {}
    try:
        models['random_forest'] = joblib.load(data_dir / 'random_forest_model.pkl')
        print("Random Forest model loaded")
    except FileNotFoundError:
        print("✗ Random Forest model not found")
    
    try:
        models['xgboost'] = joblib.load(data_dir / 'xgboost_model.pkl')
        print("XGBoost model loaded")
    except FileNotFoundError:
        print("XGBoost model not found")
    
    try:
        models['isolation_forest'] = joblib.load(data_dir / 'isolation_forest_model.pkl')
        print("Isolation Forest model loaded")
    except FileNotFoundError:
        print("Isolation Forest model not found")
    
    try:
        models['autoencoder'] = keras.models.load_model(data_dir / 'autoencoder_model.keras')
        print("Autoencoder model loaded")
    except FileNotFoundError:
        try:
            models['autoencoder'] = keras.models.load_model(data_dir / 'autoencoder_model.h5')
            print("Autoencoder model loaded (HDF5 format)")
        except FileNotFoundError:
            print("Autoencoder model not found")
    
    return X_train, y_train, X_test, y_test, models

def plot_feature_importance(X_train, y_train, models):

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)
    if 'random_forest' in models:
        rf_importance = models['random_forest'].feature_importances_
        rf_features = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        top_features = rf_features.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(plots_dir / 'random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


    if 'xgboost' in models:
        xgb_importance = models['xgboost'].feature_importances_
        xgb_features = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        top_features = xgb_features.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(plots_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


    if 'random_forest' in models and 'xgboost' in models:

        rf_top = rf_features.head(15)
        xgb_top = xgb_features.head(15)
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Random Forest
        ax1.barh(range(len(rf_top)), rf_top['importance'])
        ax1.set_yticks(range(len(rf_top)))
        ax1.set_yticklabels(rf_top['feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Random Forest - Top 15 Features')
        ax1.invert_yaxis()
        
        # XGBoost
        ax2.barh(range(len(xgb_top)), xgb_top['importance'])
        ax2.set_yticks(range(len(xgb_top)))
        ax2.set_yticklabels(xgb_top['feature'])
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('XGBoost - Top 15 Features')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_tsne_clustering(X_test, y_test, sample_size=10000):

    

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)
    

    if len(X_test) > sample_size:
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[indices]
        y_sample = y_test.iloc[indices]
    else:
        X_sample = X_test
        y_sample = y_test
    

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=1000)
    X_tsne = tsne.fit_transform(X_sample)

    plt.figure(figsize=(12, 8))

    benign_mask = y_sample == 0
    attack_mask = y_sample == 1
    
    plt.scatter(X_tsne[benign_mask, 0], X_tsne[benign_mask, 1], 
               c='blue', alpha=0.6, s=20, label='Benign', edgecolors='white', linewidth=0.5)
    plt.scatter(X_tsne[attack_mask, 0], X_tsne[attack_mask, 1], 
               c='red', alpha=0.6, s=20, label='Attack', edgecolors='white', linewidth=0.5)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Clustering of Network Traffic Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'tsne_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_analysis(X_test, y_test, sample_size=50000):


    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)

    if len(X_test) > sample_size:
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[indices]
        y_sample = y_test.iloc[indices]
    else:
        X_sample = X_test
        y_sample = y_test

    

    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_sample)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    benign_mask = y_sample == 0
    attack_mask = y_sample == 1
    
    plt.scatter(X_pca[benign_mask, 0], X_pca[benign_mask, 1], 
               c='blue', alpha=0.6, s=10, label='Benign')
    plt.scatter(X_pca[attack_mask, 0], X_pca[attack_mask, 1], 
               c='red', alpha=0.6, s=10, label='Attack')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA - First Two Components')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    top_components = 20
    plt.bar(range(1, top_components + 1), pca.explained_variance_ratio_[:top_components])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA - Top 20 Components')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_comparison(X_test, y_test, models):

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)
    

    model_results = {}

    if 'random_forest' in models:
        rf_pred = models['random_forest'].predict(X_test)
        rf_pred_proba = models['random_forest'].predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        model_results['Random Forest'] = {
            'predictions': rf_pred,
            'scores': rf_pred_proba,
            'auc': rf_auc
        }
    
    if 'xgboost' in models:
        xgb_pred = models['xgboost'].predict(X_test)
        xgb_pred_proba = models['xgboost'].predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
        model_results['XGBoost'] = {
            'predictions': xgb_pred,
            'scores': xgb_pred_proba,
            'auc': xgb_auc
        }

    if 'isolation_forest' in models:
        iso_scores = -models['isolation_forest'].score_samples(X_test)
        iso_pred = (iso_scores > np.percentile(iso_scores, 95)).astype(int)
        iso_auc = roc_auc_score(y_test, iso_scores)
        model_results['Isolation Forest'] = {
            'predictions': iso_pred,
            'scores': iso_scores,
            'auc': iso_auc
        }
    
    if 'autoencoder' in models:
        try:
            X_test_reconstructed = models['autoencoder'].predict(X_test)
            auto_scores = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
            auto_pred = (auto_scores > np.percentile(auto_scores, 95)).astype(int)
            auto_auc = roc_auc_score(y_test, auto_scores)
            model_results['Autoencoder'] = {
                'predictions': auto_pred,
                'scores': auto_scores,
                'auc': auto_auc
            }
        except Exception as e:
            print(f"Error with autoencoder: {e}")

    plt.figure(figsize=(12, 8))
    
    for model_name, results in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['scores'])
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {results["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Performance Comparison - ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'model_performance_roc.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = [['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1-Score']]
    
    for model_name, results in model_results.items():
        accuracy = (results['predictions'] == y_test).mean()
        report = classification_report(y_test, results['predictions'], output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        
        table_data.append([
            model_name,
            f"{accuracy:.3f}",
            f"{results['auc']:.3f}",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}"
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(plots_dir / 'model_performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_distribution_analysis(X_test, y_test):

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)

    sample_size = min(50000, len(X_test))
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[indices]
    y_sample = y_test.iloc[indices]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    important_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets']
    available_features = [f for f in important_features if f in X_sample.columns]
    
    if len(available_features) >= 4:
        features_to_plot = available_features[:4]
    else:

        features_to_plot = X_sample.columns[:4]
    
    for i, feature in enumerate(features_to_plot):
        row, col = i // 2, i % 2
        
        benign_data = X_sample[y_sample == 0][feature]
        attack_data = X_sample[y_sample == 1][feature]
        
        axes[row, col].hist(benign_data, bins=50, alpha=0.7, label='Benign', density=True)
        axes[row, col].hist(attack_data, bins=50, alpha=0.7, label='Attack', density=True)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Density')
        axes[row, col].set_title(f'Distribution of {feature}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    
    feature_vars = X_sample.var().sort_values(ascending=False)
    top_features = feature_vars.head(15).index
    
    correlation_matrix = X_sample[top_features].corr()
    
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Top 15 Features')
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    


def main():

    X_train, y_train, X_test, y_test, models = load_data_and_models()
    
    plot_feature_importance(X_train, y_train, models)
    plot_tsne_clustering(X_test, y_test)
    plot_pca_analysis(X_test, y_test)
    plot_model_performance_comparison(X_test, y_test, models)
    plot_data_distribution_analysis(X_test, y_test)
    

if __name__ == "__main__":
    main() 