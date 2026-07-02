import joblib
import sklearn
import sys

print("Python:", sys.executable)
print("scikit-learn:", sklearn.__version__)

print("Loading model...")

model = joblib.load("bank_churn_pipeline.pkl")

print("Loaded successfully!")
print(type(model))