import os
import pandas as pd
import pickle
from decimal import Decimal, ROUND_HALF_UP
from pycaret.anomaly import setup, create_model, assign_model

# Variables
DATA_PATH = '/app/data/Anomalydata.csv'
MODEL_PATH = '/app/model/model.pkl'

ALGORITHM = os.getenv('ALGORITHM', 'abod')  # Default changed to abod for anomaly detection

# Load Dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"Failed to read training data: {e}")
    raise

# Training with anomaly detection approach
try:
    print(f"🚀 Training model with algorithm: {ALGORITHM}")
    s = setup(data=df, verbose=False)
    model = create_model(ALGORITHM)
    result = assign_model(model)
    print("Model created and assigned")

    # Calculate statistics about the model
    scores = result['Anomaly_Score']
    mean_normal = Decimal(float(scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    std_normal = Decimal(float(scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

    model_results = {
        'mean_normal': mean_normal,
        'std_normal': std_normal
    }
    
    print("Training completed. Model statistics:")
    print(model_results)

except Exception as e:
    print(f"Error training model {ALGORITHM}: {e}")
    raise

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_PATH}")