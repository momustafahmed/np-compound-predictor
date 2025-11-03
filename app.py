"""
Modern Streamlit Dashboard for Natural Product Compound Activity Prediction
A comprehensive interactive web application for exploring data, model performance, and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NP Compound Activity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the dataset."""
    df = pd.read_csv('np.csv')
    return df


@st.cache_resource
def load_models_and_artifacts():
    """Load trained models and preprocessing artifacts."""
    if not os.path.exists('models'):
        return None
    
    artifacts = {}
    
    # Load comparison results
    if os.path.exists('models/model_comparison.csv'):
        artifacts['comparison'] = pd.read_csv('models/model_comparison.csv')
    
    # Load training results
    if os.path.exists('models/training_results.joblib'):
        artifacts['results'] = joblib.load('models/training_results.joblib')
    
    # Load preprocessing artifacts
    if os.path.exists('models/scaler.joblib'):
        artifacts['scaler'] = joblib.load('models/scaler.joblib')
    if os.path.exists('models/feature_cols.joblib'):
        artifacts['feature_cols'] = joblib.load('models/feature_cols.joblib')
    if os.path.exists('models/label_encoders.joblib'):
        artifacts['label_encoders'] = joblib.load('models/label_encoders.joblib')
    
    return artifacts


def plot_class_distribution(df):
    """Plot class distribution."""
    fig = go.Figure()
    
    class_counts = df['active'].value_counts()
    
    fig.add_trace(go.Bar(
        x=['Inactive', 'Active'],
        y=[class_counts[False], class_counts[True]],
        marker_color=['#ef4444', '#10b981'],
        text=[class_counts[False], class_counts[True]],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Compound Activity Distribution',
        xaxis_title='Class',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_molecular_properties(df):
    """Plot distribution of molecular properties."""
    properties = ['mw', 'alogp', 'tpsa', 'hbd', 'hba']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Molecular Weight', 'LogP', 'TPSA', 'H-Bond Donors', 'H-Bond Acceptors']
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for prop, (row, col) in zip(properties, positions):
        fig.add_trace(
            go.Histogram(x=df[prop], name=prop, showlegend=False, marker_color='#667eea'),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Molecular Property Distributions")
    return fig


def plot_source_class_distribution(df):
    """Plot compound source and class distributions."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Compound Source', 'Chemical Class'],
        specs=[[{'type':'pie'}, {'type':'pie'}]]
    )
    
    # Source distribution
    source_counts = df['source'].value_counts()
    fig.add_trace(
        go.Pie(labels=source_counts.index, values=source_counts.values, name='Source'),
        row=1, col=1
    )
    
    # Class distribution
    class_counts = df['class'].value_counts()
    fig.add_trace(
        go.Pie(labels=class_counts.index, values=class_counts.values, name='Class'),
        row=1, col=2
    )
    
    fig.update_layout(height=400)
    return fig


def plot_model_comparison(comparison_df):
    """Plot model performance comparison."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto',
            ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Inactive', 'Predicted Active'],
        y=['Actual Inactive', 'Actual Active'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        height=400
    )
    
    return fig


def plot_roc_curve(results):
    """Plot ROC curves for all models."""
    fig = go.Figure()
    
    for model_name, result in results.items():
        metrics = result['metrics']
        if 'roc_curve' in metrics:
            roc_data = metrics['roc_curve']
            auc = metrics['roc_auc']
            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f"{model_name} (AUC={auc:.3f})"
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier',
        showlegend=True
    ))
    
    fig.update_layout(
        title='ROC Curves - All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_feature_importance(feature_importance_df, model_name):
    """Plot feature importance."""
    fig = go.Figure(go.Bar(
        x=feature_importance_df['importance'],
        y=feature_importance_df['feature'],
        orientation='h',
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        title=f'Top Feature Importance - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig


def plot_activity_by_assay(df):
    """Plot activity distribution by assay type."""
    activity_by_assay = df.groupby(['assay', 'active']).size().reset_index(name='count')
    
    fig = px.bar(
        activity_by_assay,
        x='assay',
        y='count',
        color='active',
        barmode='group',
        title='Compound Activity by Assay Type',
        labels={'count': 'Number of Compounds', 'assay': 'Assay Type', 'active': 'Active'},
        color_discrete_map={False: '#ef4444', True: '#10b981'}
    )
    
    fig.update_layout(height=400)
    return fig


def plot_ic50_distribution(df):
    """Plot IC50 distribution."""
    fig = go.Figure()
    
    # Log-scale histogram
    fig.add_trace(go.Histogram(
        x=np.log10(df['ic50_uM']),
        marker_color='#667eea',
        nbinsx=50
    ))
    
    fig.update_layout(
        title='IC50 Distribution (log scale)',
        xaxis_title='log10(IC50 µM)',
        yaxis_title='Count',
        height=400
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🧬 Natural Product Compound Activity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning Dashboard for Drug Discovery</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        artifacts = load_models_and_artifacts()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure 'np.csv' exists in the current directory and models have been trained.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/dna.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["📊 Dashboard", "🔍 Data Explorer", "🤖 Model Performance", "🎯 Make Predictions"]
        )
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        st.metric("Total Compounds", len(df))
        st.metric("Active Compounds", df['active'].sum())
        st.metric("Activity Rate", f"{df['active'].mean():.1%}")
        
        if artifacts and 'comparison' in artifacts:
            st.markdown("---")
            st.markdown("### Best Model")
            best_model = artifacts['comparison'].iloc[0]
            st.success(f"**{best_model['Model']}**")
            st.metric("F1-Score", f"{best_model['F1-Score']:.3f}")
    
    # Main content
    if page == "📊 Dashboard":
        show_dashboard(df, artifacts)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)
    elif page == "🤖 Model Performance":
        show_model_performance(artifacts)
    elif page == "🎯 Make Predictions":
        show_predictions(artifacts)


def show_dashboard(df, artifacts):
    """Show main dashboard overview."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Compounds", f"{len(df):,}")
    with col2:
        st.metric("Active Compounds", f"{df['active'].sum():,}")
    with col3:
        st.metric("Activity Rate", f"{df['active'].mean():.1%}")
    with col4:
        if artifacts and 'comparison' in artifacts:
            best_f1 = artifacts['comparison'].iloc[0]['F1-Score']
            st.metric("Best Model F1", f"{best_f1:.3f}")
        else:
            st.metric("Models Trained", "0")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_class_distribution(df), use_container_width=True)
        st.plotly_chart(plot_activity_by_assay(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(plot_source_class_distribution(df), use_container_width=True)
        st.plotly_chart(plot_ic50_distribution(df), use_container_width=True)
    
    # Model comparison
    if artifacts and 'comparison' in artifacts:
        st.markdown("---")
        st.subheader("🏆 Model Performance Comparison")
        st.plotly_chart(plot_model_comparison(artifacts['comparison']), use_container_width=True)


def show_data_explorer(df):
    """Show data exploration page."""
    st.header("🔍 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📈 Statistics", "🔬 Molecular Properties", "🧪 Assay Analysis"])
    
    with tab1:
        st.subheader("Dataset Sample")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            source_filter = st.multiselect("Source", df['source'].unique(), default=df['source'].unique())
        with col2:
            class_filter = st.multiselect("Chemical Class", df['class'].unique(), default=df['class'].unique())
        with col3:
            assay_filter = st.multiselect("Assay Type", df['assay'].unique(), default=df['assay'].unique())
        
        # Filter data
        filtered_df = df[
            (df['source'].isin(source_filter)) &
            (df['class'].isin(class_filter)) &
            (df['assay'].isin(assay_filter))
        ]
        
        st.write(f"Showing {len(filtered_df)} compounds")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_compounds.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
    
    with tab3:
        st.subheader("Molecular Property Distributions")
        st.plotly_chart(plot_molecular_properties(df), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['mw', 'alogp', 'tpsa', 'hbd', 'hba', 'fsp3', 'dose_uM', 'ic50_uM']
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Assay Analysis")
        
        # Activity by source and assay
        pivot_table = pd.crosstab(df['source'], df['assay'], df['active'], aggfunc='sum')
        st.write("Active Compounds by Source and Assay")
        st.dataframe(pivot_table, use_container_width=True)
        
        # IC50 by class
        st.subheader("IC50 Distribution by Chemical Class")
        fig = px.box(df, x='class', y='ic50_uM', log_y=True, color='class')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance(artifacts):
    """Show model performance analysis."""
    st.header("🤖 Model Performance Analysis")
    
    if not artifacts or 'results' not in artifacts:
        st.warning("⚠️ No trained models found. Please run `python train_models.py` first.")
        
        with st.expander("📝 How to train models"):
            st.code("""
# Run the training script
python train_models.py

# This will:
# 1. Load and preprocess the data
# 2. Train 7 different ML models
# 3. Evaluate each model
# 4. Save models and metrics
            """, language="bash")
        return
    
    results = artifacts['results']
    comparison_df = artifacts['comparison']
    
    # Model comparison
    st.subheader("📊 Overall Model Comparison")
    st.plotly_chart(plot_model_comparison(comparison_df), use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📈 Detailed Metrics")
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score', 'ROC-AUC']),
        use_container_width=True
    )
    
    # ROC Curves
    st.subheader("📉 ROC Curves")
    st.plotly_chart(plot_roc_curve(results), use_container_width=True)
    
    # Individual model analysis
    st.markdown("---")
    st.subheader("🔍 Individual Model Analysis")
    
    model_names = list(results.keys())
    selected_model = st.selectbox("Select Model", model_names)
    
    if selected_model:
        model_result = results[selected_model]
        metrics = model_result['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_confusion_matrix(metrics['confusion_matrix'], selected_model),
                use_container_width=True
            )
        
        with col2:
            # Classification Report
            st.subheader("Classification Report")
            report_df = pd.DataFrame(metrics['classification_report']).T
            st.dataframe(report_df, use_container_width=True)
        
        # Feature Importance
        if model_result['feature_importance'] is not None:
            st.subheader("🎯 Feature Importance")
            st.plotly_chart(
                plot_feature_importance(model_result['feature_importance'], selected_model),
                use_container_width=True
            )


def show_predictions(artifacts):
    """Show prediction interface."""
    st.header("🎯 Make Predictions")
    
    if not artifacts or 'results' not in artifacts:
        st.warning("⚠️ No trained models found. Please run `python train_models.py` first.")
        return
    
    st.info("Enter compound properties to predict activity")
    
    # Model selection
    model_names = list(artifacts['results'].keys())
    selected_model = st.selectbox("Select Model", model_names, 
                                  index=model_names.index(artifacts['comparison'].iloc[0]['Model']))
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Compound Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source = st.selectbox("Source", ['plant', 'marine', 'fungi', 'bacteria'])
            chem_class = st.selectbox("Chemical Class", 
                                     ['alkaloid', 'terpene', 'polyketide', 'flavonoid', 'peptide', 'saponin'])
            assay = st.selectbox("Assay Type", ['anticancer', 'antibacterial', 'antimalarial'])
        
        with col2:
            mw = st.number_input("Molecular Weight", min_value=100.0, max_value=1000.0, value=350.0)
            alogp = st.number_input("LogP", min_value=-5.0, max_value=10.0, value=2.5)
            tpsa = st.number_input("TPSA", min_value=0.0, max_value=300.0, value=80.0)
        
        with col3:
            hbd = st.number_input("H-Bond Donors", min_value=0, max_value=20, value=3)
            hba = st.number_input("H-Bond Acceptors", min_value=0, max_value=20, value=5)
            fsp3 = st.number_input("Fsp3", min_value=0.0, max_value=1.0, value=0.4)
        
        col1, col2 = st.columns(2)
        with col1:
            dose_uM = st.number_input("Dose (µM)", min_value=0.01, max_value=1000.0, value=10.0)
        with col2:
            ic50_uM = st.number_input("IC50 (µM)", min_value=0.01, max_value=1000.0, value=150.0)
        
        submit = st.form_submit_button("🔮 Predict Activity", use_container_width=True)
    
    if submit:
        try:
            # Encode categorical features
            le_source = artifacts['label_encoders']['source']
            le_class = artifacts['label_encoders']['class']
            le_assay = artifacts['label_encoders']['assay']
            
            source_encoded = le_source.transform([source])[0]
            class_encoded = le_class.transform([chem_class])[0]
            assay_encoded = le_assay.transform([assay])[0]
            
            # Engineer features
            log_ic50 = np.log10(ic50_uM + 1)
            log_dose = np.log10(dose_uM + 1)
            dose_ic50_ratio = dose_uM / (ic50_uM + 1)
            mw_tpsa_ratio = mw / (tpsa + 1)
            hba_hbd_ratio = hba / (hbd + 1)
            lipinski_violations = sum([mw > 500, alogp > 5, hbd > 5, hba > 10])
            druglikeness_score = sum([
                150 <= mw <= 500,
                -0.4 <= alogp <= 5.6,
                20 <= tpsa <= 130,
                hbd <= 5,
                hba <= 10
            ]) / 5.0
            
            # Create feature vector
            features = np.array([[
                mw, alogp, tpsa, hbd, hba, fsp3,
                dose_uM, ic50_uM, log_ic50, log_dose,
                dose_ic50_ratio, mw_tpsa_ratio, hba_hbd_ratio,
                lipinski_violations, druglikeness_score,
                source_encoded, class_encoded, assay_encoded
            ]])
            
            # Scale features
            features_scaled = artifacts['scaler'].transform(features)
            
            # Make prediction
            model = artifacts['results'][selected_model]['model']
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("### ✅ ACTIVE")
                else:
                    st.error("### ❌ INACTIVE")
            
            if probability is not None:
                with col2:
                    st.metric("Active Probability", f"{probability[1]:.1%}")
                with col3:
                    st.metric("Inactive Probability", f"{probability[0]:.1%}")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={'text': "Activity Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 50], 'color': "#fee2e2"},
                            {'range': [50, 75], 'color': "#fef3c7"},
                            {'range': [75, 100], 'color': "#d1fae5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Compound summary
            with st.expander("📋 Compound Summary"):
                summary_df = pd.DataFrame({
                    'Property': ['Source', 'Chemical Class', 'Assay', 'Molecular Weight', 
                               'LogP', 'TPSA', 'H-Bond Donors', 'H-Bond Acceptors',
                               'Fsp3', 'Dose (µM)', 'IC50 (µM)', 'Lipinski Violations',
                               'Drug-likeness Score'],
                    'Value': [source, chem_class, assay, f"{mw:.1f}", f"{alogp:.2f}",
                            f"{tpsa:.1f}", hbd, hba, f"{fsp3:.2f}", f"{dose_uM:.2f}",
                            f"{ic50_uM:.2f}", lipinski_violations, f"{druglikeness_score:.2f}"]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
