import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data_dir = Path('../data')
    train_data = pd.read_csv(data_dir / 'train_data.csv')
    test_data = pd.read_csv(data_dir / 'test_data.csv')
    
    # print(f"Training set shape: {train_data.shape}")
    # print(f"Test set shape: {test_data.shape}")

    feature_columns = [col for col in train_data.columns if col not in ['is_attack', ' Label']]
    X_train = train_data[feature_columns]
    y_train = train_data['is_attack']
    X_test = test_data[feature_columns]
    y_test = test_data['is_attack']
    
    # print(f"X_train shape: {X_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # print(f"\nTraining set class distribution:")
    # print(y_train.value_counts())
    # print(f"Training set attack rate: {y_train.mean():.2%}")
    
    # print(f"\nTest set class distribution:")
    # print(y_test.value_counts())
    # print(f"Test set attack rate: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test

def train_random_forest(X_train, y_train, X_test, y_test):

    print("\n" + "="*60)
    print("RANDOM FOREST MODEL")
    print("="*60)
    

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    

    accuracy = rf_model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    # print(f"\nclassification Report:")
    # print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n 10 most important features:")
    print(feature_importance.head(10))
    

    #saving model
    model_path = Path('../data/random_forest_model.pkl')
    joblib.dump(rf_model, model_path)
    
    return rf_model, y_pred, y_pred_proba, feature_importance

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("XGBOOST MODEL")
    print("="*60)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    

    xgb_model.fit(X_train, y_train)
    

    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    accuracy = xgb_model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nXGBoost Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    # print(f"\nclassification report:")
    # print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n 10 most important features:")
    print(feature_importance.head(10))
    
    model_path = Path('../data/xgboost_model.pkl')
    joblib.dump(xgb_model, model_path)
    
    return xgb_model, y_pred, y_pred_proba, feature_importance

def hyperparameter_tuning_xgboost(X_train, y_train):
    #didnt run this because it takes too long
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    

    base_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    model_path = Path('../data/xgboost_tuned_model.pkl')
    joblib.dump(best_model, model_path)
    
    return best_model

def plot_results(rf_results, xgb_results, X_test, y_test):
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)

    plots_dir = Path('../visualizations')
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_results[2])
    rf_auc = roc_auc_score(y_test, rf_results[2])
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
    

    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_results[2])
    xgb_auc = roc_auc_score(y_test, xgb_results[2])
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    cm_rf = confusion_matrix(y_test, rf_results[1])
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_xgb = confusion_matrix(y_test, xgb_results[1])
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('XGBoost Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    rf_importance = rf_results[3].head(10)
    axes[0].barh(range(len(rf_importance)), rf_importance['importance'])
    axes[0].set_yticks(range(len(rf_importance)))
    axes[0].set_yticklabels(rf_importance['feature'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest - Top 10 Features')
    axes[0].invert_yaxis()

    xgb_importance = xgb_results[3].head(10)
    axes[1].barh(range(len(xgb_importance)), xgb_importance['importance'])
    axes[1].set_yticks(range(len(xgb_importance)))
    axes[1].set_yticklabels(xgb_importance['feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('XGBoost - Top 10 Features')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    
    X_train, y_train, X_test, y_test = load_data()
    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    
    plot_results(rf_results, xgb_results, X_test, y_test)

if __name__ == "__main__":
    main() 