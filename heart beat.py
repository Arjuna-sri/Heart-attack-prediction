import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import pickle

# Generate sample dataset
def generate_sample_data(n_samples=300):
    np.random.seed(42)
    age = np.clip(np.random.normal(55, 15, n_samples), 20, 100)
    sex = np.random.binomial(1, 0.5, n_samples)
    trestbps = np.clip(np.random.normal(130, 20, n_samples), 90, 200)
    chol = np.clip(np.random.normal(240, 50, n_samples), 120, 400)
    fbs = np.random.binomial(1, 0.15, n_samples)
    thalach = np.clip(np.random.normal(150, 25, n_samples), 70, 220)
    
    risk = 0.1 + 0.3 * (age > 55)
    risk += 0.2 * (trestbps > 140)
    risk += 0.2 * (chol > 240)
    risk += 0.1 * fbs
    risk += 0.2 * (thalach < 150)
    risk += 0.1 * sex
    risk = risk / risk.max()
    target = np.random.binomial(1, risk)
    
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'target': target
    })
    return data

# Load data
df = generate_sample_data(300)

# Quick EDA
print("Dataset Preview:")
print(df.head())
print("\nClass Distribution:")
print(df['target'].value_counts())
print(f"Percentage of positive cases: {df['target'].mean() * 100:.2f}%")

# Prepare features
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to build pipeline
def create_model(classifier):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

# Models to test
models = {
    'Logistic Regression': create_model(LogisticRegression(random_state=42)),
    'Random Forest': create_model(RandomForestClassifier(random_state=42))
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc_score,
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print("Classification Report:")
    print(results[name]['report'])

# Cross-validation
print("\nCross-Validation Scores:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Feature importance from Random Forest
rf_model = models['Random Forest'].named_steps['classifier']
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(importance_df)

# Risk prediction function
def predict_heart_attack_risk(model, age, sex, trestbps, chol, fbs, thalach):
    new_patient = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach
    }])
    probability = model.predict_proba(new_patient)[0, 1]
    prediction = model.predict(new_patient)[0]
    return probability, prediction

# Prediction Example
best_model = models['Random Forest']
prob, pred = predict_heart_attack_risk(
    best_model,
    age=65,
    sex=1,
    trestbps=160,
    chol=280,
    fbs=1,
    thalach=120
)
print("\nPrediction Example:")
print(f"Risk Probability: {prob:.4f}")
print(f"Prediction: {'High Risk' if pred == 1 else 'Low Risk'}")

# Save model
with open('heart_attack_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nModel saved as 'heart_attack_prediction_model.pkl'")
