#!/bin/bash
# ============================================================
# Train All Models — Predictive Maintenance Pipeline
# ============================================================
set -e

echo "=========================================="
echo "  PREDICTIVE MAINTENANCE — MODEL TRAINING"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check data exists
if [ ! -f "data/raw/PdM_telemetry.csv" ]; then
    echo "❌ Data not found. Run 'make download-data' first."
    exit 1
fi
echo "✓ Data found"

# Run pipeline
echo ""
echo "Step 1/4: Preprocessing..."
python -m src.data.preprocessor

echo ""
echo "Step 2/4: Feature engineering..."
python -m src.data.feature_engine

echo ""
echo "Step 3/4: Training models..."
python -m src.models.train --model all

echo ""
echo "Step 4/4: Evaluation..."
python -m src.evaluation.metrics
python -m src.evaluation.cost_analysis
python -m src.evaluation.visualization

echo ""
echo "=========================================="
echo "  ✅ ALL MODELS TRAINED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "  Models saved to: models/"
echo "  Reports saved to: reports/figures/"
echo ""
echo "  Next steps:"
echo "    make dashboard   → Launch Streamlit dashboard"
echo "    make api         → Start FastAPI prediction server"
echo "    make test        → Run unit tests"
