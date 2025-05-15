import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    /* Main content area */
    .main {
        background-color: #f9f9f9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Cards */
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Titles */
    .title {
        color: #333333;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Risk indicators */
    .high-risk {
        color: #d32f2f;
        font-weight: 600;
    }
    
    .low-risk {
        color: #388e3c;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4a6baf;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
    }
    
    /* Progress bars */
    .stProgress>div>div>div>div {
        background-color: #4a6baf;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a6baf;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("heart.csv")
    
    # Preprocessing
    data = df.copy()
    label_encoder_sex = LabelEncoder()
    label_encoder_exercise = LabelEncoder()
    data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])
    data['ExerciseAngina'] = label_encoder_exercise.fit_transform(data['ExerciseAngina'])
    data = pd.get_dummies(data, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True, dtype=int)
    
    scaler = MinMaxScaler()
    columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return df, X_train, X_test, y_train, y_test, scaler, label_encoder_sex, label_encoder_exercise, X.columns

@st.cache_resource
def train_models(X_train, y_train):
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return logistic_model, rf_model

def main():
    # Load data and models
    df, X_train, X_test, y_train, y_test, scaler, le_sex, le_exercise, feature_names = load_and_preprocess_data()
    logistic_model, rf_model = train_models(X_train, y_train)
    
    # Calculate metrics
    logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test)) * 100
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100
    
    # Sidebar - Patient Input
    with st.sidebar:
        st.markdown("## Patient Information")
        
        with st.form("patient_form"):
            st.markdown("### Demographic")
            age = st.slider("Age", 18, 100, 50)
            sex = st.radio("Sex", list(le_sex.classes_))
            
            st.markdown("### Medical History")
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
            resting_bp = st.slider("Resting BP (mm Hg)", 70, 200, 120)
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
            fasting_bs = st.radio("Fasting BS > 120 mg/dL?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            
            st.markdown("### Exercise Test")
            max_hr = st.slider("Max Heart Rate", 60, 202, 150)
            exercise_angina = st.radio("Exercise Angina", list(le_exercise.classes_))
            oldpeak = st.slider("ST Depression", -2.6, 6.2, 0.0, 0.1)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
            
            submitted = st.form_submit_button("Predict Risk")

    # Main Content
    st.markdown("# Heart Disease Prediction Dashboard")
    
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
        
        # Preprocess input
        input_df = pd.DataFrame([input_data])
        input_df['Sex'] = le_sex.transform(input_df['Sex'])
        input_df['ExerciseAngina'] = le_exercise.transform(input_df['ExerciseAngina'])
        input_df = pd.get_dummies(input_df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)
        
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]
        
        columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
        
        # Make predictions
        logistic_pred = logistic_model.predict(input_df)[0]
        rf_pred = rf_model.predict(input_df)[0]
        logistic_prob = logistic_model.predict_proba(input_df)[0][1] * 100
        rf_prob = rf_model.predict_proba(input_df)[0][1] * 100
        
        # Results Cards
        st.markdown("## Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.markdown("### Logistic Regression")
                st.markdown(f"Accuracy: {logistic_accuracy:.1f}%")
                if logistic_pred == 1:
                    st.markdown(f"<p class='high-risk'>High Risk ({logistic_prob:.1f}%)</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='low-risk'>Low Risk ({logistic_prob:.1f}%)</p>", unsafe_allow_html=True)
                st.progress(int(logistic_prob))
        
        with col2:
            with st.container():
                st.markdown("### Random Forest")
                st.markdown(f"Accuracy: {rf_accuracy:.1f}%")
                if rf_pred == 1:
                    st.markdown(f"<p class='high-risk'>High Risk ({rf_prob:.1f}%)</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='low-risk'>Low Risk ({rf_prob:.1f}%)</p>", unsafe_allow_html=True)
                st.progress(int(rf_prob))
        
        # Combined risk assessment
        with st.container():
            st.markdown("### Combined Risk Assessment")
            avg_risk = (logistic_prob + rf_prob) / 2
            st.progress(int(avg_risk))
            st.markdown(f"Average Risk Score: {avg_risk:.1f}%")
            if avg_risk > 50:
                st.markdown("<p class='high-risk'>Recommendation: Further cardiac evaluation recommended</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='low-risk'>Recommendation: Low risk - maintain healthy lifestyle</p>", unsafe_allow_html=True)
        
        # Feature Importance
        st.markdown("## Key Risk Factors")
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                         title='Top 10 Most Important Features',
                         color='Importance',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance
        st.markdown("## Model Performance")
        
        tab1, tab2 = st.tabs(["Confusion Matrices", "Classification Report"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Logistic Regression")
                cm_log = confusion_matrix(y_test, logistic_model.predict(X_test))
                fig = px.imshow(cm_log, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Healthy', 'Disease'], y=['Healthy', 'Disease'],
                               color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Random Forest")
                cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
                fig = px.imshow(cm_rf, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Healthy', 'Disease'], y=['Healthy', 'Disease'],
                               color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Logistic Regression")
                report = classification_report(y_test, logistic_model.predict(X_test), output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))
            
            with col2:
                st.markdown("#### Random Forest")
                report = classification_report(y_test, rf_model.predict(X_test), output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))
    
    # Data Overview (always visible)
    st.markdown("## Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("#### Heart Disease Distribution")
            fig = px.pie(df, names='HeartDisease', 
                        title='Percentage of Patients with Heart Disease',
                        color='HeartDisease',
                        color_discrete_map={0: '#4CAF50', 1: '#F44336'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.container():
            st.markdown("#### Age Distribution")
            fig = px.histogram(df, x='Age', color='HeartDisease',
                             nbins=20,
                             color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                             title='Age Distribution by Heart Disease Status')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()