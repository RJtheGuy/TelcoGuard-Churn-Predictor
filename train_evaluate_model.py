


import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans


def load_model(model_path="models/churn_model.joblib"):
    """Load the trained model artifacts with error handling"""
    try:
        if os.path.exists(model_path):
            artifacts = joblib.load(model_path)
            st.session_state.model_artifacts = artifacts
            return artifacts
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



def train_and_save_churn_model(X, y, project_name="churn", save_path="models"):
    """
    Production-ready training with all original models and proper data handling
    """
    os.makedirs(save_path, exist_ok=True)
    
    try:
        def safe_convert(col):
            try:
                return pd.to_numeric(col)
            except:
                return col
        
        X = X.apply(safe_convert)
        non_numeric = X.select_dtypes(include=['object', 'category']).columns
        
        if len(non_numeric) > 0:
            raise ValueError(
                f"Non-numeric columns detected after conversion: {list(non_numeric)}\n"
                "Please ensure your preprocessing converts all features to numeric."
            )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                enable_categorical=False
            ),
            'SVC': SVC(probability=True, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        }
        
        results = []
        best_models = {}
        
        for name, clf in classifiers.items():
            try:
                print(f"\n=== Training {name} ===")
                
                if name == 'XGBoost':
                    X_train_res_xgb = X_train_res.astype(np.float32)
                    X_test_xgb = X_test.astype(np.float32)
                    clf.fit(X_train_res_xgb, y_train_res)
                    y_pred = clf.predict(X_test_xgb)
                    y_proba = clf.predict_proba(X_test_xgb)[:, 1]
                else:
                    clf.fit(X_train_res, y_train_res)
                    y_pred = clf.predict(X_test)
                    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else [np.nan]*len(X_test)
                
                metrics = {
                    'model': name,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_proba) if hasattr(clf, 'predict_proba') else np.nan
                }
                results.append(metrics)
                best_models[name] = clf
                
            except Exception as e:
                print(f"Warning: {name} failed - {str(e)}")
                continue
        
        if not results:
            raise RuntimeError("All model training attempts failed")
        
        results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
        best_model_name = results_df.iloc[0]['model']
        best_model = best_models[best_model_name]
        
        artifacts = {
            'model': best_model,
            'metrics': results_df.to_dict('records'),
            'feature_names': list(X.columns),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_model': best_model_name
        }
        
        joblib.dump(artifacts, f"{save_path}/{project_name}_model.joblib")
        return artifacts
        
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        raise


def cluster_churn_reasons(df):
    """Enhanced churn segmentation with more features and better visualization"""
    churned_customers = df[df['Churn'] == 1].copy()
    
    feature_cols = [
        'MonthlyCharges', 'TotalCharges', 'tenure',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'total_services', 'has_internet', 'senior_with_partner'
    ]
    
    for col in feature_cols:
        if col not in churned_customers.columns:
            churned_customers[col] = 0
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    churned_customers['Churn Segment'] = kmeans.fit_predict(churned_customers[feature_cols])
    
    segment_stats = churned_customers.groupby('Churn Segment')[['MonthlyCharges', 'TotalCharges', 'tenure', 'total_services']].mean()
    churned_customers['Segment Profile'] = churned_customers['Churn Segment'].map({
        0: "Price-sensitive (low tenure, few services)",
        1: "High-value (many services, high charges)",
        2: "Long-term (high tenure, medium usage)"
    })
    
    return churned_customers[['Churn Segment', 'Segment Profile', 'MonthlyCharges', 
                            'TotalCharges', 'tenure', 'total_services']]
