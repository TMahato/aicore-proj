import os
import pandas as pd
import pickle

# Variables
DATA_PATH = '/app/data/irisdataset.csv'
MODEL_PATH = '/app/model/model.pkl'

ALGORITHM = os.getenv('ALGORITHM', 'svm')
CLASS_LABEL = float(os.getenv('CLASS_LABEL', 1.0))
KERNEL = os.getenv('KERNEL', 'rbf')

# Load Dataset
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# Dynamic model initialization
model = None
if ALGORITHM == 'svm':
    from sklearn import svm
    model = svm.SVC(C=CLASS_LABEL, kernel=KERNEL)
elif ALGORITHM == 'randomforest':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=int(CLASS_LABEL))
elif ALGORITHM == 'logistic':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=CLASS_LABEL)
else:
    raise ValueError(f"Unsupported algorithm: {ALGORITHM}")

# Train and evaluate
model.fit(train_x, train_y)
score = model.score(test_x, test_y)
print(f"Test Accuracy: {score}")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
