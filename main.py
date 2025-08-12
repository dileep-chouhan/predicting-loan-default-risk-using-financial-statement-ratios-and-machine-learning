import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_samples = 500
# Generate synthetic financial statement data
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_samples),
    'DebtEquityRatio': np.random.uniform(0.1, 2, num_samples),
    'ProfitMargin': np.random.uniform(0.01, 0.2, num_samples),
    'Default': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]) # 80% non-default, 20% default
}
df = pd.DataFrame(data)
# --- 2. Data Preparation ---
X = df[['CurrentRatio', 'DebtEquityRatio', 'ProfitMargin']]
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training ---
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
# --- 5. Visualization ---
# Feature Importance (Illustrative - Logistic Regression doesn't directly provide feature importance like tree-based models)
plt.figure(figsize=(8, 6))
plt.bar(['Current Ratio', 'Debt/Equity Ratio', 'Profit Margin'], model.coef_[0])
plt.title('Feature Coefficients (Illustrative)')
plt.ylabel('Coefficient Magnitude')
plt.xlabel('Financial Ratio')
plt.savefig('feature_coefficients.png')
print("Plot saved to feature_coefficients.png")
#Accuracy visualization (simple bar chart)
plt.figure(figsize=(6,4))
plt.bar(['Accuracy'],[accuracy])
plt.ylim(0,1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.savefig('model_accuracy.png')
print("Plot saved to model_accuracy.png")