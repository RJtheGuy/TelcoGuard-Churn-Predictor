import pytest
import pandas as pd
import numpy as np
from preprocessing import clean_data, encode_features, create_features, scale_features

# --------------------------------------------
# Fixtures (Reusable Test Data)
# --------------------------------------------

@pytest.fixture
def raw_data():
    """Sample raw input data with edge cases"""
    return pd.DataFrame({
        'customerID': ['1', '2', '3'],
        'TotalCharges': ['100', 'foo', '200.50'],
        'gender': ['Male', 'Female', 'Male'],
        'Partner': ['Yes', 'No', 'No'],
        'Contract': ['Month-to-month', 'Two year', 'One year'],
        'Churn': ['Yes', 'No', 'No'],
        'tenure': [5, 25, 0],  # Includes zero tenure
        'MonthlyCharges': [20, 50, 100],
        'PhoneService': ['Yes', 'Yes', 'No']
    })

@pytest.fixture 
def cleaned_data(raw_data):
    """Pre-cleaned data for downstream tests"""
    return clean_data(raw_data)

# --------------------------------------------
# Unit Tests
# --------------------------------------------

class TestDataCleaning:
    def test_drops_customer_id(self, cleaned_data):
        assert 'customerID' not in cleaned_data.columns
        
    def test_handles_invalid_charges(self, cleaned_data):
        assert pd.api.types.is_numeric_dtype(cleaned_data['TotalCharges'])
        assert cleaned_data['TotalCharges'].isna().sum() == 0

class TestFeatureEncoding:
    def test_binary_encoding(self, cleaned_data):
        encoded = encode_features(cleaned_data)
        assert set(encoded['gender'].unique()) == {0, 1}
        
    def test_one_hot_encoding(self, cleaned_data):
        encoded = encode_features(cleaned_data)
        assert 'Contract_Two year' in encoded.columns
        assert encoded['Churn'].isin([0, 1]).all()

class TestFeatureEngineering:
    def test_creates_derived_features(self, cleaned_data):
        encoded = encode_features(cleaned_data)
        engineered = create_features(encoded)
        
        # Test binning
        assert 'tenure_group_12-24' in engineered.columns
        
        # Test ratios
        assert np.isclose(
            engineered.loc[0, 'avg_monthly_charges'],
            engineered.loc[0, 'TotalCharges'] / 5.1,  # tenure + 0.1
            rtol=0.01
        )
        
        # Test service flags
        assert 'total_services' in engineered.columns

class TestScaling:
    def test_scales_numerical_features(self, cleaned_data):
        encoded = encode_features(cleaned_data)
        engineered = create_features(encoded)
        
        numerical_cols = ['MonthlyCharges', 'TotalCharges']
        scaled, scaler = scale_features(engineered, numerical_cols)
        
        for col in numerical_cols:
            assert np.isclose(scaled[col].mean(), 0, atol=1e-6)
            assert np.isclose(scaled[col].std(), 1, atol=1e-6)

# --------------------------------------------
# Integration Test (Full Pipeline)
# --------------------------------------------

def test_full_pipeline(raw_data):
    """End-to-end test"""
    cleaned = clean_data(raw_data)
    encoded = encode_features(cleaned)
    engineered = create_features(encoded)
    scaled, _ = scale_features(engineered, ['MonthlyCharges', 'TotalCharges'])
    
    # Verify final output shape
    assert scaled.shape[0] == 3
    assert 'Churn' in scaled.columns
    assert 'MonthlyCharges' in scaled.columns
