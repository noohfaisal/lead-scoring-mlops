import pandas as pd
import numpy as np
import random
import os

def generate_leads_data(n=1000, drift=False):
    """
    Generates synthetic leads data.
    If drift=True, shifts distributions to simulate production drift.
    """
    np.random.seed(42 if not drift else 99)
    
    # Features
    job_titles = ['Manager', 'Director', 'Engineer', 'Student', 'Executive']
    weights = [0.3, 0.2, 0.3, 0.1, 0.1]
    
    if drift:
        # Drift: More students (who click but don't buy)
        weights = [0.1, 0.1, 0.2, 0.5, 0.1] 
        print("Generating DRIFTED data (High Student count)...")
    else:
        print("Generating NORMAL training data...")

    jobs = np.random.choice(job_titles, n, p=weights)
    
    # Engagement Score (0-10)
    # Norm: Students have high engagement but low conversion
    engagement = []
    for job in jobs:
        if job == 'Student':
            engagement.append(max(0, min(10, np.random.normal(8, 1))))
        else:
            engagement.append(max(0, min(10, np.random.normal(5, 2))))
            
    engagement = np.array(engagement)
    
    # Target: Converted (0 or 1)
    # Students rarely convert. Managers/Directors convert often.
    conversion_prob = []
    for job, eng in zip(jobs, engagement):
        prob = 0.1 # Base probability
        
        if job in ['Manager', 'Director']:
            prob += 0.4
        elif job == 'Student':
            prob -= 0.05
            
        # Higher engagement increases probability
        prob += (eng / 20)
        
        conversion_prob.append(max(0, min(1, prob)))
        
    converted = np.random.binomial(1, conversion_prob)
    
    df = pd.DataFrame({
        'job_title': jobs,
        'engagement_score': engagement,
        'converted': converted
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # generate training data
    train_df = generate_leads_data(n=1000, drift=False)
    train_df.to_csv("data/training_data.csv", index=False)
    print("Saved data/training_data.csv")
    
    # generate drifted production data
    prod_df = generate_leads_data(n=500, drift=True)
    prod_df.to_csv("data/production_data.csv", index=False)
    print("Saved data/production_data.csv")
