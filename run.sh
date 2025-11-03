#!/bin/bash

# 🧬 Natural Product Compound Activity Predictor - Quick Start Script
# This script will train models and launch the dashboard

echo "=============================================="
echo "🧬 NP Compound Activity Predictor"
echo "=============================================="
echo ""

# Check if models exist
if [ ! -d "models" ] || [ ! -f "models/random_forest.joblib" ]; then
    echo "📚 Training machine learning models..."
    echo "This will take a few minutes..."
    echo ""
    python train_models.py
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Training failed! Please check the error messages above."
        exit 1
    fi
    
    echo ""
    echo "✅ Models trained successfully!"
    echo ""
else
    echo "✅ Models already trained."
    echo ""
fi

# Launch dashboard
echo "🚀 Launching Streamlit dashboard..."
echo ""
echo "The dashboard will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run app.py
