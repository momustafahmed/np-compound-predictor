"""
Training script for natural product compound activity prediction.
Trains multiple ML models and saves them with their metrics.
"""

import sys
import joblib
from preprocessing import get_full_pipeline
from models import train_all_models, save_models, compare_models


def main():
    """Main training pipeline."""
    print("="*70)
    print("NATURAL PRODUCT COMPOUND ACTIVITY PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    data = get_full_pipeline(filepath='np.csv', test_size=0.2, random_state=42)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  - Training samples: {data['X_train'].shape[0]}")
    print(f"  - Test samples: {data['X_test'].shape[0]}")
    print(f"  - Number of features: {len(data['feature_cols'])}")
    print(f"  - Class balance (train): {data['y_train'].mean():.2%} active compounds")
    
    # Step 2: Train all models
    print("\n[2/4] Training machine learning models...")
    results = train_all_models(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        feature_names=data['feature_cols'],
        use_smote=True  # Handle class imbalance
    )
    
    # Step 3: Compare models
    print("\n[3/4] Comparing model performance...")
    comparison_df = compare_models(results)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    
    # Step 4: Save models and artifacts
    print("\n[4/4] Saving models and artifacts...")
    save_models(results, output_dir='models')
    
    # Save preprocessing artifacts
    joblib.dump(data['scaler'], 'models/scaler.joblib')
    joblib.dump(data['feature_cols'], 'models/feature_cols.joblib')
    joblib.dump(data['label_encoders'], 'models/label_encoders.joblib')
    
    # Save comparison results
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    
    # Save complete results
    joblib.dump(results, 'models/training_results.joblib')
    
    print("\n✓ All models and artifacts saved successfully!")
    print(f"  - Models directory: ./models/")
    print(f"  - Comparison results: ./models/model_comparison.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYou can now run the Streamlit dashboard:")
    print("  streamlit run app.py")
    print("="*70)
    
    return results, comparison_df


if __name__ == "__main__":
    try:
        results, comparison = main()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
