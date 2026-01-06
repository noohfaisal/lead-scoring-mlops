import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train():
    # Load Data
    print("Loading data...")
    df = pd.read_csv("data/training_data.csv")
    
    X = df.drop("converted", axis=1)
    y = df["converted"]
    
    # Preprocessing
    categorical_features = ["job_title"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="passthrough"
    )
    
    # Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow Tracking
    mlflow.set_experiment("Lead_Scoring_Model")
    
    with mlflow.start_run():
        print("Training model...")
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_param("n_estimators", 100)
        
        # Log Model
        mlflow.sklearn.log_model(clf, "model", registered_model_name="LeadScoringModel")
        print("Model logged and registered to MLflow.")

if __name__ == "__main__":
    train()
