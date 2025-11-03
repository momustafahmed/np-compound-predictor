# 🧬 Natural Product Compound Activity Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-blue.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for predicting natural product compound bioactivity with an interactive Streamlit dashboard.

🔗 **GitHub Repository**: [https://github.com/momustafahmed/np-compound-predictor](https://github.com/momustafahmed/np-compound-predictor)

## 👩‍🔬 Author

**Zhinya Kawa Othman**  
Faculty of Pharmaceutical Sciences  
Chulalongkorn University  
Bangkok, Thailand

## 📋 Overview

This project implements multiple machine learning models to predict whether natural product compounds are active against various biological targets (anticancer, antibacterial, antimalarial). The system includes:

- **7 Machine Learning Models**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, K-Nearest Neighbors, and Naive Bayes
- **Modern Interactive Dashboard**: Built with Streamlit for data exploration, model analysis, and predictions
- **Comprehensive Data Pipeline**: Automated preprocessing, feature engineering, and model evaluation
- **Advanced Visualizations**: Interactive plots using Plotly for deep insights

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/momustafahmed/np-compound-predictor.git
cd np-compound-predictor

# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Launch dashboard
streamlit run app.py
```

Visit **http://localhost:8501** in your browser!

## 🚀 Features

### Data Processing
- Automated data cleaning and missing value imputation
- Feature engineering with molecular descriptors
- Drug-likeness scoring and Lipinski's Rule of Five violations
- SMOTE for handling class imbalance

### Machine Learning Models
- **Random Forest**: Ensemble learning with 200 trees
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **Logistic Regression**: Fast baseline model
- **SVM**: Support Vector Machine with RBF kernel
- **Gradient Boosting**: Sequential ensemble method
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier

### Dashboard Features
1. **📊 Dashboard**: Overview with key metrics and visualizations
2. **🔍 Data Explorer**: Interactive data filtering and statistical analysis
3. **🤖 Model Performance**: Detailed model comparison with ROC curves and confusion matrices
4. **🎯 Make Predictions**: Real-time predictions with confidence scores

## 📦 Installation

1. **Clone or download the project**
   ```bash
   cd /Users/momustafahmed/Downloads/NP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Step 1: Train Models
```bash
python train_models.py
```

This will:
- Load and preprocess the data from `np.csv`
- Train all 7 machine learning models
- Evaluate performance on test set
- Save models to `models/` directory
- Generate performance comparison report

Expected output:
```
Training samples: 1601
Test samples: 401
Number of features: 18
Class balance (train): 12.93% active compounds

✓ Random Forest trained successfully!
✓ XGBoost trained successfully!
...
🏆 Best Model: XGBoost (F1-Score: 0.7234)
```

### Step 2: Launch Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Pages

### 1. Dashboard Overview
- Total compounds and activity statistics
- Class distribution visualizations
- Compound source and chemical class breakdown
- IC50 distribution analysis
- Model performance comparison

### 2. Data Explorer
- **Raw Data**: Filter by source, class, and assay type
- **Statistics**: Descriptive statistics and missing value analysis
- **Molecular Properties**: Distribution plots and correlation heatmaps
- **Assay Analysis**: Activity patterns across different assays

### 3. Model Performance
- Side-by-side model comparison
- ROC curves for all models
- Confusion matrices
- Feature importance analysis
- Detailed classification reports

### 4. Make Predictions
- Interactive form for compound properties
- Real-time activity predictions
- Confidence scores with visual gauges
- Compound property summary

## 📁 Project Structure

```
NP/
├── np.csv                      # Dataset (2002 compounds)
├── preprocessing.py            # Data preprocessing module
├── models.py                   # ML models implementation
├── train_models.py            # Model training script
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── models/                    # Trained models (created after training)
    ├── random_forest.joblib
    ├── xgboost.joblib
    ├── logistic_regression.joblib
    ├── svm.joblib
    ├── gradient_boosting.joblib
    ├── k-nearest_neighbors.joblib
    ├── naive_bayes.joblib
    ├── scaler.joblib
    ├── feature_cols.joblib
    ├── label_encoders.joblib
    └── model_comparison.csv
```

## 🔬 Dataset Features

### Original Features
- **compound_id**: Unique identifier
- **source**: Origin (plant, marine, fungi, bacteria)
- **class**: Chemical class (alkaloid, terpene, polyketide, etc.)
- **assay**: Biological target (anticancer, antibacterial, antimalarial)
- **mw**: Molecular weight
- **alogp**: Lipophilicity (log partition coefficient)
- **tpsa**: Topological polar surface area
- **hbd**: Hydrogen bond donors
- **hba**: Hydrogen bond acceptors
- **fsp3**: Fraction of sp3 carbons
- **dose_uM**: Test dose in micromolar
- **ic50_uM**: Half maximal inhibitory concentration
- **active**: Target variable (activity)

### Engineered Features
- **log_ic50**: Log-transformed IC50
- **log_dose**: Log-transformed dose
- **dose_ic50_ratio**: Ratio of dose to IC50
- **mw_tpsa_ratio**: Molecular weight to TPSA ratio
- **hba_hbd_ratio**: Acceptor to donor ratio
- **lipinski_violations**: Number of Lipinski's Rule violations
- **druglikeness_score**: Composite drug-likeness metric

## 📈 Model Performance

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Classification breakdown

Typical results (may vary):
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.91 | 0.78 | 0.68 | 0.72 | 0.89 |
| Random Forest | 0.90 | 0.75 | 0.65 | 0.70 | 0.88 |
| Gradient Boosting | 0.90 | 0.74 | 0.64 | 0.69 | 0.87 |
| SVM | 0.89 | 0.71 | 0.62 | 0.66 | 0.85 |
| Logistic Regression | 0.88 | 0.68 | 0.58 | 0.63 | 0.82 |
| K-Nearest Neighbors | 0.87 | 0.65 | 0.55 | 0.60 | 0.79 |
| Naive Bayes | 0.85 | 0.62 | 0.52 | 0.57 | 0.77 |

## 🛠️ Advanced Usage

### Customize Model Training
Edit `models.py` to adjust hyperparameters:
```python
models = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=300,  # Increase trees
            max_depth=20,      # Deeper trees
            # ... other parameters
        )
    }
}
```

### Add New Features
Edit `preprocessing.py` to add custom features:
```python
def engineer_features(df):
    # Add your custom features
    df_eng['custom_feature'] = df['mw'] * df['alogp']
    return df_eng
```

### Export Predictions
In the dashboard, use the download button to export filtered data and predictions.

## 📚 Dependencies

- **streamlit**: Web dashboard framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting
- **plotly**: Interactive visualizations
- **imbalanced-learn**: SMOTE for class balancing
- **joblib**: Model serialization

## 🤝 Contributing

Feel free to:
- Add new models
- Implement additional feature engineering
- Enhance visualizations
- Improve prediction interface

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Dataset contains 2,002 natural product compounds with bioactivity data
- Models implement state-of-the-art machine learning algorithms
- Dashboard designed for drug discovery researchers and data scientists

## 📞 Support

For issues or questions:
1. Check that all dependencies are installed
2. Ensure `np.csv` is in the project directory
3. Run `python train_models.py` before launching the dashboard
4. Check terminal output for error messages

---

**Built with ❤️ for Drug Discovery Research**
