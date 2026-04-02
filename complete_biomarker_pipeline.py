import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Simulate multi-omics dataset
np.random.seed(42)
num_samples = 500
num_features = 100
X = np.random.rand(num_samples, num_features)
# Simulating binary target variable for grades
y = np.random.choice([0, 1], size=num_samples)

# Create DataFrame
omics_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
omics_data['target'] = y

# Exploratory Data Analysis (EDA)
print(omics_data.describe())
sns.countplot(x='target', data=omics_data)
plt.title('Target Variable Distribution')
plt.show()

# Feature Selection - LASSO
X = omics_data.drop('target', axis=1)
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)
lasso_features = np.where(np.abs(lasso.coef_) > 0)[0]

# Feature Selection - Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_scaled, y)
importance = rf.feature_importances_
important_features = np.argsort(importance)[-10:]

# Combine selected features
selected_features = set(lasso_features) | set(important_features)
X_selected = X.iloc[:, list(selected_features)]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(model, X_selected, y, cv=skf)
print(f'Cross-validation scores: {cross_val_scores}')

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# SHAP Analysis
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_selected.columns)

# Visualizations
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Output results
output_df = pd.DataFrame({
    'feature': X_selected.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
output_df.to_csv('biomarker_panels.csv', index=False)
print('Biomarker panels saved to biomarker_panels.csv')
