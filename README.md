# Lead Scoring MLOps POC

This repository contains a Proof of Concept (POC) demonstrating a robust MLOps architecture for a **Lead Scoring Model**. It simulates a production-grade machine learning lifecycle, including data generation, model training, tracking, and automated drift detection.

## 🚀 Key Features

*   **MLflow Integration**: Automated tracking of experiments, metrics, and model versioning.
*   **Drift Detection**: Implements **Jensen-Shannon Distance** to monitor statistical shifts between training and production data (e.g., detecting if the "Job Title" distribution changes).
*   **Circuit Breaker Pattern**: The pipeline automatically fails if significant data drift is detected, preventing potentially degraded models from serving predictions.
*   **Reproducible Pipeline**: A single script (`pipeline.sh`) orchestrates the entire workflow from data generation to monitoring.

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **MLflow**: For experiment tracking and model registry.
*   **Scikit-Learn**: For the Random Forest classification model.
*   **Pandas & NumPy**: For data manipulation.
*   **SciPy**: For statistical distance calculations (monitoring).

## 📂 Project Structure

```bash
.
├── data_generator.py   # Simulates CRM data (Normal vs Drifted)
├── train.py            # Trains model & logs to MLflow
├── inference.py        # Loads model from MLflow & scores leads
├── monitor.py          # Checks for data drift (Jensen-Shannon)
├── pipeline.sh         # Orchestrates the end-to-end flow
└── requirements.txt    # Python dependencies
```

## ⚡ Getting Started

### 1. Installation

Clone the repository and set up your environment:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the POC

Execute the full MLOps pipeline with a single command:

```bash
chmod +x pipeline.sh
./pipeline.sh
```

**What happens when you run this?**
1.  **Data Generation**: Creates synthetic `training_data.csv` (normal) and `production_data.csv` (drifted).
2.  **Training**: Trains a Random Forest model and registers it to MLflow.
3.  **Inference**: Loads the latest model from the registry and scores the production leads.
4.  **Monitoring**: Compares training vs. production distributions.
    *   *Note*: The pipeline is designed to **FAIL** (Exit 1) at this step to demonstrate the drift detection alert mechanism.

## 📊 Monitoring Logic

The `monitor.py` script ensures model safety by calculating the **Jensen-Shannon Distance** for critical features (like `job_title`).

*   **Threshold**: `0.2`
*   **Status**: `DRIFT DETECTED` if distance > threshold.

This simulates a real-world scenario where a marketing campaign accidentally targets a low-converting demographic (e.g., "Students"), prompting the system to halt and alert the data science team.
