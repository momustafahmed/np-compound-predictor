"""
Machine learning models module for compound activity prediction.
Implements multiple classification algorithms with evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import joblib


def get_model_configs():
    """
    Define configurations for all models.
    
    Returns:
        Dictionary of model names and their configurations
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'description': 'Fast linear model, good baseline'
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'description': 'Ensemble of decision trees, robust and interpretable'
        },
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'description': 'Gradient boosting, typically best performance'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'description': 'Sequential ensemble method'
        },
        'SVM': {
            'model': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'description': 'Support Vector Machine with RBF kernel'
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                n_jobs=-1
            ),
            'description': 'Instance-based learning algorithm'
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'description': 'Probabilistic classifier based on Bayes theorem'
        }
    }
    
    return models


def handle_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance in training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: Resampling method ('smote', 'none')
        
    Returns:
        Resampled X_train and y_train
    """
    if method == 'smote':
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    else:
        return X_train, y_train


def train_model(model, X_train, y_train, model_name='Model'):
    """
    Train a single model.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    print(f"✓ {model_name} trained successfully!")
    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a trained model and return comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['average_precision'] = average_precision_score(y_test, y_proba)
        metrics['y_proba'] = y_proba
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_test, y_pred, target_names=['Inactive', 'Active'], output_dict=True
    )
    
    return metrics


def print_model_metrics(metrics):
    """
    Print model evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print(f"\n{'='*60}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"Avg Precision: {metrics['average_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"{'='*60}")


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    else:
        return None


def train_all_models(X_train, y_train, X_test, y_test, feature_names, 
                     use_smote=True, models_to_train=None):
    """
    Train and evaluate all models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        use_smote: Whether to apply SMOTE
        models_to_train: List of model names to train (None = all)
        
    Returns:
        Dictionary containing trained models and their metrics
    """
    # Handle class imbalance
    if use_smote:
        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Filter models if specified
    if models_to_train:
        model_configs = {k: v for k, v in model_configs.items() if k in models_to_train}
    
    # Train and evaluate each model
    results = {}
    
    for model_name, config in model_configs.items():
        try:
            # Train model
            model = config['model']
            trained_model = train_model(model, X_train_balanced, y_train_balanced, model_name)
            
            # Evaluate model
            metrics = evaluate_model(trained_model, X_test, y_test, model_name)
            
            # Get feature importance if available
            feature_importance = get_feature_importance(trained_model, feature_names)
            
            # Store results
            results[model_name] = {
                'model': trained_model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'description': config['description']
            }
            
            # Print metrics
            print_model_metrics(metrics)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return results


def save_models(results, output_dir='models'):
    """
    Save trained models and their metrics to disk.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save models
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, result in results.items():
        # Save model
        model_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}.joblib"
        joblib.dump(result['model'], model_filename)
        print(f"Saved {model_name} to {model_filename}")
        
        # Save metrics
        metrics_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}_metrics.joblib"
        joblib.dump(result['metrics'], metrics_filename)
        
        # Save feature importance if available
        if result['feature_importance'] is not None:
            fi_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}_feature_importance.csv"
            result['feature_importance'].to_csv(fi_filename, index=False)


def load_model(model_path):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(model_path)


def compare_models(results):
    """
    Create a comparison dataframe of all models.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame comparing all models
    """
    comparison = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics.get('roc_auc', np.nan),
            'Avg Precision': metrics.get('average_precision', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    return comparison_df


if __name__ == "__main__":
    print("Models module loaded successfully!")
    print(f"Available models: {list(get_model_configs().keys())}")
