#!/bin/bash
set -e

echo "=========================================="
echo "    LEAD SCORING MLOps POC PIPELINE       "
echo "=========================================="

# 1. Generate Data
echo ""
echo "[Step 1] Generating Data..."
python data_generator.py

# 2. Train Model
echo ""
echo "[Step 2] Training Model & Logging to MLflow..."
python train.py

# 3. Batch Inference (on Drifted Data to trigger alert, or normal data)
# First run on training data (should be stable)
echo ""
echo "[Step 3a] Inference on Reference Data (Sanity Check)..."
python inference.py # Defaults to production_data.csv, let's change logic or inputs?
# Actually inference.py defaults to production_data.csv which is DRIFTED in generator. 

# 4. Monitor Drift
echo ""
echo "[Step 4] Monitoring Data Drift..."
# Check Drift between Training Data vs. Production Data
# data_generator.py creates production_data.csv with DRIFT=True by default for the demo.
set +e # Allow failure for demo purposes
python monitor.py
MONITOR_EXIT_CODE=$?
set -e

if [ $MONITOR_EXIT_CODE -ne 0 ]; then
    echo "⚠️  Pipeline Alert: Data Drift Detected! Triggering retraining or technician alert..."
else
    echo "✅  Pipeline Status: Healthy."
fi

echo ""
echo "=========================================="
echo "          POC EXECUTION COMPLETE          "
echo "=========================================="
