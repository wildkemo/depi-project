import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure the page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
This interactive dashboard predicts heart disease risk using machine learning models.
""")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("heart.csv")
    
    # Preprocessing (same as your original code)
    data = df.copy()
    label_encoder_sex = LabelEncoder()
    label_encoder_exercise = LabelEncoder()
    data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])
    data['ExerciseAngina'] = label_encoder_exercise.fit_transform(data['ExerciseAngina'])
    data = pd.get_dummies(data, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True, dtype=int)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    # Split data
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder_sex, label_encoder_exercise, X.columns

# Train models
@st.cache_resource
def train_models(X_train, y_train):
    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return logistic_model, rf_model

# Main dashboard function
def main():
    # Load data and train models
    X_train, X_test, y_train, y_test, scaler, le_sex, le_exercise, feature_names = load_and_preprocess_data()
    logistic_model, rf_model = train_models(X_train, y_train)
    
    # Calculate accuracies
    logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test)) * 100
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100
    
    # Sidebar - User input
    st.sidebar.header("Patient Information")
    
    with st.sidebar.form("patient_form"):
        st.subheader("Demographic Information")
        age = st.slider("Age", 18, 100, 50)
        sex = st.radio("Sex", list(le_sex.classes_))
        
        st.subheader("Medical History")
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 70, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        
        st.subheader("Exercise Test Results")
        max_hr = st.slider("Maximum Heart Rate Achieved", 60, 202, 150)
        exercise_angina = st.radio("Exercise Induced Angina", list(le_exercise.classes_))
        oldpeak = st.slider("ST Depression Induced by Exercise", -2.6, 6.2, 0.0, 0.1)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Up", "Flat", "Down"])
        
        submitted = st.form_submit_button("Predict Heart Disease Risk")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Patient Risk Assessment")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Sex': sex,
                'ChestPainType': chest_pain,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': int(fasting_bs),
                'RestingECG': resting_ecg,
                'MaxHR': max_hr,
                'ExerciseAngina': exercise_angina,
                'Oldpeak': oldpeak,
                'ST_Slope': st_slope
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            input_df['Sex'] = le_sex.transform(input_df['Sex'])
            input_df['ExerciseAngina'] = le_exercise.transform(input_df['ExerciseAngina'])
            input_df = pd.get_dummies(input_df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)
            
            # Ensure all columns are present
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]
            
            # Scale numerical features
            columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
            
            # Make predictions
            logistic_pred = logistic_model.predict(input_df)[0]
            rf_pred = rf_model.predict(input_df)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Logistic Regression", 
                         "High Risk" if logistic_pred == 1 else "Low Risk",
                         f"Accuracy: {logistic_accuracy:.1f}%")
            
            with col2:
                st.metric("Random Forest", 
                         "High Risk" if rf_pred == 1 else "Low Risk",
                         f"Accuracy: {rf_accuracy:.1f}%")
            
            # Visual indicators
            risk_level = (logistic_pred + rf_pred) / 2
            st.progress(int(risk_level * 100))
            st.caption(f"Overall risk assessment: {risk_level*100:.0f}%")
            
            # Feature importance
            st.subheader("Key Risk Factors")
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(5)
                
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Reds_r', ax=ax)
                ax.set_title('Top Contributing Factors')
                st.pyplot(fig)
    
    with col2:
        st.header("Model Performance")
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Logistic Regression CM
        y_pred_log = logistic_model.predict(X_test)
        cm_log = confusion_matrix(y_test, y_pred_log)
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Logistic Regression')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Random Forest CM
        y_pred_rf = rf_model.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title('Random Forest')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        st.pyplot(fig)
        
        # Model metrics
        st.subheader("Evaluation Metrics")
        st.markdown(f"""
        - **Logistic Regression Accuracy**: {logistic_accuracy:.1f}%
        - **Random Forest Accuracy**: {rf_accuracy:.1f}%
        """)

if __name__ == "__main__":
    main()