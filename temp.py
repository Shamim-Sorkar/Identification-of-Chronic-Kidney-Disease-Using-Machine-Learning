import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error
)

# 📥 Load dataset
data = pd.read_csv('Chronic_kidney_Disease_Shuffled.csv')
data['class'] = data['class'].map({'ckd': 1, 'notckd': 0})
X = data.drop('class', axis=1)
y = data['class']

# 🔧 Preprocessing
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_num = num_pipe.fit_transform(X[numerical_cols]) if len(numerical_cols) > 0 else np.empty((X.shape[0], 0))

X_cat = X[categorical_cols].copy()
label_encoders = {}
for col in categorical_cols:
    mode = X_cat[col].mode()[0] if len(X_cat[col].mode()) > 0 else 'missing'
    X_cat[col] = X_cat[col].fillna(mode).astype(str)
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])
    label_encoders[col] = le
X_cat = X_cat.values if len(categorical_cols) > 0 else np.empty((X.shape[0], 0))

X_all_processed = np.hstack([X_num, X_cat])
feature_names = list(numerical_cols) + list(categorical_cols)

# 💾 Save full preprocessed dataset
df_all_processed = pd.DataFrame(X_all_processed, columns=feature_names)
df_all_processed['class'] = y
df_all_processed.to_csv('CKD_Preprocessed_AllFeatures.csv', index=False)

# 🔍 Feature Importance (Top 10)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_all_processed, y)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot top 10
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 CKD Predictive Features', fontsize=14)
plt.tight_layout()
plt.show()

top_features = feature_importance_df['Feature'].head(10).tolist()
selected_indices = [feature_names.index(f) for f in top_features]
X_top10 = X_all_processed[:, selected_indices]

# 💾 Save top 10 feature dataset
df_top10_processed = pd.DataFrame(X_top10, columns=top_features)
df_top10_processed['class'] = y
df_top10_processed.to_csv('CKD_Preprocessed_Top10Features.csv', index=False)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance')
}

# Evaluation Function
def evaluate_models(X_data, label):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    roc_curves = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name} on {label}")
        
        if hasattr(model, 'predict_proba'):
            y_prob = cross_val_predict(model, X_data, y, cv=kf, method='predict_proba')[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = cross_val_predict(model, X_data, y, cv=kf, method='predict')
            y_prob = None

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob) if y_prob is not None else np.nan
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        results[name] = {
            'Feature Set': label,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'MAE': mae,
            'MSE': mse
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(4,4))
        plt.title(f'Confusion Matrix - {name} ({label})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        classes = ['Not CKD (0)', 'CKD (1)']
        plt.xticks([0,1], classes)
        plt.yticks([0,1], classes)
        plt.grid(False)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, color='black')
        plt.gca().set_xticks(np.arange(-.5, 2, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, 2, 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=1)
        plt.gca().tick_params(which='minor', bottom=False, left=False)
        plt.tight_layout()
        plt.show()

        # ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_curves.append((fpr, tpr, name, label, roc_auc))
            plt.figure(figsize=(6,4))
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', color='darkorange')
            plt.plot([0,1], [0,1], 'k--')
            plt.title(f'ROC Curve - {name} ({label})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.tight_layout()
            plt.show()
    return pd.DataFrame(results).T, roc_curves

# 🧪 Evaluate
df_all, roc_all = evaluate_models(X_all_processed, 'All Features')
df_top, roc_top = evaluate_models(X_top10, 'Top 10 Features')

# 📈 Combined Comparison
combined_df = pd.concat([df_all, df_top])
combined_df = combined_df[['Feature Set', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'MAE', 'MSE']]
print("\n🔍 Performance Comparison (All Features vs Top 10):")
print(combined_df.round(4))

# 📊 Grouped Bar Charts
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_df.reset_index(), x='index', y=metric, hue='Feature Set', palette='viridis')
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.legend(title='Feature Set')
    plt.tight_layout()
    plt.show()

# 🧵 Combined ROC Curves
plt.figure(figsize=(8,6))
for fpr, tpr, name, label, auc in roc_all + roc_top:
    plt.plot(fpr, tpr, label=f'{name} ({label}) AUC={auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Combined ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()
