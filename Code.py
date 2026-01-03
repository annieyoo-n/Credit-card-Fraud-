import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_and_preprocess_data():
    file_path = r'C:\Users\zulfi\Documents\OneDrive\Desktop\VSC\ML AS - 02\creditcard.csv'
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        print("Dataset Shape:", df.shape)
        print("\nAvailable Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        if 'Class' not in df.columns:
            print("\nError: 'Class' column not found. Available columns are:")
            print(df.columns.tolist())
            return None, None
        
        X: pd.DataFrame = df.drop('Class', axis=1)
        y: pd.Series = df['Class']
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X[['Amount', 'Time']])
        X['Amount'] = scaled[:, 0]
        X['Time'] = scaled[:, 1]
        
        return X, y
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def handle_imbalance(X, y):
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y) # type: ignore
    print("Resampled dataset shape:", X_resampled.shape)
    print("Resampled class distribution:")
    print(pd.Series(y_resampled).value_counts()) # type: ignore
    return X_resampled, y_resampled

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    train_time = time.time() - start_time
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n{model_name} Results:")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'training_time': train_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics
    }

def tune_hyperparameters(model, param_grid, X_train, y_train, model_name):
    print(f"\nTuning {model_name} hyperparameters...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def main():
    print("Credit Card Fraud Detection: Random Forest vs XGBoost")
    print("=" * 60)
    
    X, y = load_and_preprocess_data()
    if X is None:
        return
    
    X_resampled, y_resampled = handle_imbalance(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    rf_model = RandomForestClassifier(
        n_estimators=50,  
        max_depth=10,    
        n_jobs=-1,
        random_state=42
    )
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    rf_results = train_and_evaluate(rf_model, X_train, X_test, y_train, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_results['predictions'], "Random Forest")
    
    xgb_results = train_and_evaluate(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
    plot_confusion_matrix(y_test, xgb_results['predictions'], "XGBoost")

    plt.figure(figsize=(10, 8))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_results['probabilities'])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_results['probabilities'])
    
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_results["metrics"]["roc_auc"]:.2f})')
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_results["metrics"]["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()
    
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<15} {'ROC-AUC':<10} {'PR-AUC':<10} {'Training Time (s)':<15}")
    print("-" * 60)
    print(f"{'Random Forest':<15} {rf_results['metrics']['roc_auc']:<10.4f} {rf_results['metrics']['pr_auc']:<10.4f} {rf_results['metrics']['training_time']:<15.2f}")
    print(f"{'XGBoost':<15} {xgb_results['metrics']['roc_auc']:<10.4f} {xgb_results['metrics']['pr_auc']:<10.4f} {xgb_results['metrics']['training_time']:<15.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()