import mlflow.sklearn
import pandas as pd
import sys

def predict(input_path="data/production_data.csv"):
    print(f"Loading predictions for {input_path}...")
    
    # Load Model (Production alias or latest version)
    model_name = "LeadScoringModel"
    # In a real scenario, we'd use 'models:/{model_name}/Production'
    # For this POC, we'll just grab the latest version
    model_uri = f"models:/{model_name}/latest"
    print(f"Loading model from {model_uri}...")
    
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load Data
    df = pd.read_csv(input_path)
    X = df.drop("converted", axis=1) # "converted" wouldn't exist in live inference, but we preserve ground truth for drift/eval later if needed
    
    # Predict
    predictions = model.predict_proba(X)[:, 1] # Probability of class 1
    
    df["score"] = predictions
    
    # Simulate "Sales Qualified Lead" (SQL) definition
    df["is_sql"] = df["score"] > 0.7
    
    output_path = "data/scored_leads.csv"
    df.to_csv(output_path, index=False)
    print(f"Scored leads saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    predict()
