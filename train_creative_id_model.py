
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib
import xgboost as xgb

# Load dataset
DATA_PATH = 'creative_id_dataset.csv'
df = pd.read_csv(DATA_PATH)

# Group rare creative IDs into 'other' if they appear less than 5 times
id_counts = df['final_creative_id'].value_counts()
rare_ids = id_counts[id_counts < 5].index.tolist()
df['final_creative_id_grouped'] = df['final_creative_id'].apply(lambda x: x if x not in rare_ids else 'other')

# Feature engineering
features = [
    'initial_creative_id', 'wrapper_count', 'adomain', 'ssai_creative_id', 'wrapper_chain'
]
X = df[features].fillna('')
for col in ['initial_creative_id', 'adomain', 'ssai_creative_id', 'wrapper_chain']:
    X[col] = X[col].astype(str)
    X[col] = X[col].astype('category').cat.codes
y = df['final_creative_id_grouped'].astype(str).astype('category').cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance
import matplotlib.pyplot as plt
importances = clf.feature_importances_
feature_names = X.columns
for name, imp in zip(feature_names, importances):
    print(f"Feature: {name:20s} Importance: {imp:.4f}")
plt.bar(feature_names, importances)
plt.title('Feature Importances')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('feature_importance.png')
print('Feature importance plot saved as feature_importance.png')

# Save model
joblib.dump(clf, 'creative_id_xgb_model.pkl')
print('Model saved as creative_id_xgb_model.pkl')
