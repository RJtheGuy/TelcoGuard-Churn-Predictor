import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

def preprocess_data(df):
    
    data = df.copy()
    

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    if data['TotalCharges'].isnull().sum() > 0:
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
        
    data = data.drop('customerID', axis=1)
    
    binary_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_vars:
        data[col] = data[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    categorical_vars = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod']
    
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
    
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
     
    data['tenure_group'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                  labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])
    data = pd.get_dummies(data, columns=['tenure_group'], drop_first=True)
    
    data['avg_monthly_charges'] = data['TotalCharges'] / (data['tenure'] + 0.1)  
    data['charges_to_tenure_ratio'] = data['MonthlyCharges'] / (data['tenure'] + 0.1)
    
    service_columns = ['PhoneService', 'MultipleLines_Yes', 'OnlineSecurity_Yes', 
                       'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 
                       'StreamingTV_Yes', 'StreamingMovies_Yes']
    
    for col in service_columns:
        if col not in data.columns and col.split('_')[0] + '_No' in data.columns:
            service_col = col.split('_')[0]
            data[col] = data[f'{service_col}_No'].apply(lambda x: 0 if x == 1 else 1)
    
    service_columns = [col for col in service_columns if col in data.columns]
    data['total_services'] = data[service_columns].sum(axis=1)
    
    if 'InternetService_No' in data.columns:
        data['has_internet'] = data['InternetService_No'].apply(lambda x: 0 if x == 1 else 1)
    else:
        internet_cols = [col for col in data.columns if col.startswith('InternetService_')]
        if internet_cols:
            data['has_internet'] = data[internet_cols].max(axis=1)
    
    
    security_columns = [col for col in ['OnlineSecurity_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes'] 
                        if col in data.columns]
    data['has_security_service'] = (data[security_columns].sum(axis=1) > 0).astype(int)
    
    if 'OnlineBackup_Yes' in data.columns:
        data['has_backup'] = data['OnlineBackup_Yes']
    
    streaming_columns = [col for col in ['StreamingTV_Yes', 'StreamingMovies_Yes'] 
                         if col in data.columns]
    data['has_streaming'] = (data[streaming_columns].sum(axis=1) > 0).astype(int)
    
    contract_risk = {'Contract_Month-to-month': 2, 'Contract_One year': 1, 'Contract_Two year': 0}
    data['contract_risk'] = 2
    for contract, risk in contract_risk.items():
        if contract in data.columns:
            data.loc[data[contract] == 1, 'contract_risk'] = risk
    
    if 'PaymentMethod_Electronic check' in data.columns:
        data['payment_risk'] = data['PaymentMethod_Electronic check']
    
    data['senior_with_partner'] = (data['SeniorCitizen'] & data['Partner']).astype(int)
    
    data['has_family'] = ((data['Partner'] == 1) | (data['Dependents'] == 1)).astype(int)
    
    data['senior_internet_user'] = 0
    if 'has_internet' in data.columns:
        data['senior_internet_user'] = (data['SeniorCitizen'] & data['has_internet']).astype(int)
    
    data['long_term_customer'] = (data['tenure'] > 24).astype(int)
    
    data['high_value'] = (data['MonthlyCharges'] > data['MonthlyCharges'].quantile(0.75)).astype(int)
    
    data['price_sensitivity'] = data['total_services'] / (data['MonthlyCharges'] + 0.1)
    
    
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges', 
                      'charges_to_tenure_ratio', 'price_sensitivity']
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, data

