import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from data_processing_function import preprocess_data
from train_evaluate_model import train_and_save_churn_model, cluster_churn_reasons,load_model


def main():
    st.title("TelcoGuard Churn Prediction Dashboard")
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head())
            
            st.write("### Processing Data...")
            X, y, processed_df = preprocess_data(df)
            X_scaled = X  

            artifacts = load_model()
            if artifacts:
                st.success(f"Pre-trained model loaded ({artifacts['best_model']}, trained on {artifacts['training_date']})")
                model = artifacts['model']
            else:
                with st.spinner('Training model... This may take a few minutes'):
                    artifacts = train_and_save_churn_model(X_scaled, y)
                    model = artifacts['model']
                    st.success("Model trained successfully!")
                    st.write("### Model Performance")
                    st.dataframe(pd.DataFrame(artifacts['metrics']))
                    
                    y_pred = model.predict(X_scaled)
                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
            
            st.write("### Making Predictions...")
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            df['Churn Probability'] = probabilities
            df['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            
            st.write("Prediction Distribution:")
            st.write(df['Churn Prediction'].value_counts())
            
            st.download_button(
                "Download Predictions",
                df.to_csv(index=False),
                "churn_predictions.csv",
                "text/csv"
            )
            
            st.write("### Churn Segmentation Analysis")
            churn_segments = cluster_churn_reasons(processed_df)
            
            tab1, tab2 = st.tabs(["Data View", "Visual Analysis"])
            
            with tab1:
                st.dataframe(churn_segments.head(10))
            
            with tab2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(x='Churn Segment', y='MonthlyCharges', data=churn_segments, ax=ax1)
                sns.scatterplot(x='tenure', y='TotalCharges', hue='Segment Profile', 
                               data=churn_segments, ax=ax2, palette='viridis')
                st.pyplot(fig)
            
            st.write("### Suggested Retention Strategies")
            st.markdown("""
                **Segment 0 - Price-sensitive customers:**  
                • Discounted bundle packages  
                • Budget-friendly service tiers  
                • Loyalty reward programs  
                
                **Segment 1 - High-value customers:**  
                • Premium support services  
                • Exclusive feature access  
                • Personalized account management  
                
                **Segment 2 - Long-term customers:**  
                • Contract renewal incentives  
                • Legacy customer benefits  
                • Tenure-based upgrades
            """)
            
            if st.checkbox("Show High-Risk Customers"):
                high_risk = df[df['Churn Probability'] > 0.7].sort_values('Churn Probability', ascending=False)
                st.write(f"Found {len(high_risk)} high-risk customers:")
                st.dataframe(high_risk.head(10))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again")

if __name__ == "__main__":
    main()