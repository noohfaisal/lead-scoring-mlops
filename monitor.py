import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import sys

def check_data_drift(reference_path="data/training_data.csv", production_path="data/production_data.csv", threshold=0.2):
    """
    Compares the distribution of a feature in training (reference) 
    vs. live (production) using Jensen-Shannon Distance.
    """
    print(f"Comparing {reference_path} (Ref) vs {production_path} (Prod)...")
    
    ref_df = pd.read_csv(reference_path)
    prod_df = pd.read_csv(production_path)
    
    # Focus on 'job_title' distribution drift
    # Since JS distance works on probability vectors, we compute relative frequencies.
    
    # Align categories
    all_categories = sorted(list(set(ref_df['job_title'].unique()) | set(prod_df['job_title'].unique())))
    
    ref_counts = ref_df['job_title'].value_counts(normalize=True).reindex(all_categories, fill_value=0)
    prod_counts = prod_df['job_title'].value_counts(normalize=True).reindex(all_categories, fill_value=0)
    
    # Calculate drift score (0 = Identical, 1 = Completely Different)
    # jensenshannon returns distance which is sqrt(JS divergence)
    drift_score = jensenshannon(ref_counts, prod_counts)
    
    status = "DRIFT DETECTED" if drift_score > threshold else "STABLE"
    
    print(f"Feature: job_title | Drift Score: {drift_score:.4f} | Status: {status}")
    
    if drift_score > threshold:
        print("ALERT: Significant shift in job title distribution detected.")
        sys.exit(1) # Return non-zero exit code for pipeline awareness
    else:
        sys.exit(0)

if __name__ == "__main__":
    check_data_drift()
