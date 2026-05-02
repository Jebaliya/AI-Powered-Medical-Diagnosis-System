import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import pickle
import json
from datetime import datetime
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLES ====================
st.markdown("""
    <style>
    .main-title { font-size: 3em; font-weight: bold; text-align: center; color: #1f77b4; }
    .main-subheader { font-size: 1.3em; text-align: center; color: #666; }
    .prediction-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 20px; border-radius: 10px; color: white; margin: 20px 0;
    }
    .prediction-title { font-size: 1.5em; font-weight: bold; }
    .prediction-value { font-size: 2.5em; font-weight: bold; margin-top: 10px; }
    .confidence { font-size: 1.2em; color: #90EE90; margin-top: 10px; }
    .disclaimer { 
        background: #fff3cd; padding: 15px; border-radius: 8px; 
        border-left: 5px solid #ffc107; color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== PRECAUTIONS DICTIONARY ====================
PRECAUTIONS = {
    'Flu': ['Rest well', 'Stay hydrated', 'Avoid contact with others', 'Take antiviral medications'],
    'Common Cold': ['Use saline nasal drops', 'Rest', 'Gargle with warm water', 'Consume vitamin C'],
    'Allergies': ['Avoid allergens', 'Use antihistamines', 'Keep windows closed', 'Clean air filters'],
    'Asthma': ['Use prescribed inhaler', 'Avoid triggers', 'Monitor peak flow', 'Seek medical help if symptoms worsen'],
    'Pneumonia': ['Complete antibiotic course', 'Rest', 'Stay hydrated', 'Seek immediate medical care'],
    'Bronchitis': ['Use expectorant cough syrups', 'Stay hydrated', 'Use humidifier', 'Rest'],
    'Migraine': ['Rest in dark room', 'Apply cold compress', 'Take prescribed medication', 'Avoid triggers'],
    'Hypertension': ['Reduce salt intake', 'Exercise regularly', 'Reduce stress', 'Take medications as prescribed'],
    'Diabetes': ['Monitor blood sugar', 'Healthy diet', 'Regular exercise', 'Take insulin/medications as prescribed'],
    'Arthritis': ['Apply heat/cold', 'Gentle exercise', 'Anti-inflammatory medications', 'Physical therapy'],
    'Skin Infection': ['Keep area clean', 'Apply topical antibiotic', 'Use clean bandages', 'Avoid scratching'],
    'Anxiety': ['Deep breathing exercises', 'Meditation', 'Regular exercise', 'Consult therapist'],
    'Gastritis': ['Avoid spicy food', 'Take antacids', 'Eat smaller meals', 'Avoid alcohol'],
    'Insomnia': ['Maintain sleep schedule', 'Avoid caffeine', 'Relaxation techniques', 'Consult doctor if persistent'],
    'Anemia': ['Iron-rich diet', 'Take supplements', 'Regular checkups', 'Increase protein intake'],
    'Thyroid': ['Take thyroid medications', 'Regular TSH tests', 'Adequate iodine intake', 'Manage stress'],
    'Infection': ['Take antibiotics', 'Keep wound clean', 'Rest', 'Drink fluids'],
    'Weakness': ['Adequate sleep', 'Nutritious diet', 'Gradual exercise', 'Medical checkup'],
    'Fatigue': ['Rest adequately', 'Balanced diet', 'Stay hydrated', 'Consult doctor'],
    'Acne': ['Keep skin clean', 'Use non-comedogenic products', 'Avoid touching face', 'Consult dermatologist']
}

# ==================== SYMPTOMS LIST ====================
ALL_SYMPTOMS = [
    'Fever', 'Cough', 'Sore Throat', 'Headache', 'Body Aches',
    'Fatigue', 'Chills', 'Shortness of Breath', 'Chest Pain', 'Runny Nose',
    'Sneezing', 'Congestion', 'Wheezing', 'Skin Rash', 'Itching',
    'Nausea', 'Vomiting', 'Diarrhea', 'Abdominal Pain', 'Loss of Appetite',
    'Dizziness', 'Tremors', 'Palpitations', 'High Blood Pressure', 'Low Blood Pressure',
    'Anxiety', 'Depression', 'Insomnia', 'Muscle Pain', 'Joint Pain',
    'Swelling', 'Inflammation', 'Redness', 'Discharge', 'Weakness'
]

# ==================== DISEASE LIST ====================
DISEASES = [
    'Flu', 'Common Cold', 'Allergies', 'Asthma', 'Pneumonia',
    'Bronchitis', 'Migraine', 'Hypertension', 'Diabetes', 'Arthritis',
    'Skin Infection', 'Anxiety', 'Gastritis', 'Insomnia', 'Anemia',
    'Thyroid', 'Infection', 'Weakness', 'Fatigue', 'Acne'
]

# ==================== SYNTHETIC DATASET GENERATION ====================
def generate_synthetic_dataset():
    """Generate synthetic dataset for training"""
    np.random.seed(42)
    
    symptom_disease_mapping = {
        'Flu': ['Fever', 'Cough', 'Sore Throat', 'Body Aches', 'Fatigue', 'Chills'],
        'Common Cold': ['Runny Nose', 'Sneezing', 'Cough', 'Sore Throat', 'Congestion'],
        'Allergies': ['Sneezing', 'Runny Nose', 'Itching', 'Rash'],
        'Asthma': ['Wheezing', 'Shortness of Breath', 'Chest Pain', 'Cough'],
        'Pneumonia': ['Fever', 'Cough', 'Shortness of Breath', 'Chest Pain', 'Fatigue'],
        'Bronchitis': ['Cough', 'Chest Pain', 'Shortness of Breath', 'Wheezing'],
        'Migraine': ['Headache', 'Nausea', 'Vomiting', 'Dizziness'],
        'Hypertension': ['Headache', 'Dizziness', 'Chest Pain'],
        'Diabetes': ['Fatigue', 'Loss of Appetite', 'Weakness'],
        'Arthritis': ['Joint Pain', 'Muscle Pain', 'Swelling', 'Inflammation'],
        'Skin Infection': ['Redness', 'Itching', 'Swelling', 'Discharge'],
        'Anxiety': ['Palpitations', 'Tremors', 'Anxiety', 'Dizziness'],
        'Gastritis': ['Abdominal Pain', 'Nausea', 'Loss of Appetite'],
        'Insomnia': ['Insomnia', 'Fatigue', 'Anxiety'],
        'Anemia': ['Weakness', 'Fatigue', 'Dizziness'],
        'Thyroid': ['Fatigue', 'Weakness', 'Weight Changes'],
        'Infection': ['Fever', 'Chills', 'Weakness'],
        'Weakness': ['Weakness', 'Fatigue', 'Loss of Appetite'],
        'Fatigue': ['Fatigue', 'Weakness'],
        'Acne': ['Skin Rash', 'Redness', 'Itching']
    }
    
    X = []
    y = []
    
    for disease, symptoms in symptom_disease_mapping.items():
        for _ in range(5):  # 5 samples per disease
            selected_symptoms = symptoms.copy()
            if np.random.random() > 0.5 and len(ALL_SYMPTOMS) > len(symptoms):
                extra = np.random.choice([s for s in ALL_SYMPTOMS if s not in symptoms], 
                                        size=min(2, len([s for s in ALL_SYMPTOMS if s not in symptoms])), 
                                        replace=False)
                selected_symptoms.extend(extra)
            X.append(selected_symptoms)
            y.append(disease)
    
    return X, y

# ==================== MODEL TRAINING & LOADING ====================
def train_model():
    """Train the model and save it"""
    X_train, y_train = generate_synthetic_dataset()
    
    # Initialize encoders
    mlb = MultiLabelBinarizer()
    X_encoded = mlb.fit_transform(X_train)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_encoded, y_encoded)
    
    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'mlb': mlb, 'le': le}, f)
    
    return model, mlb, le

def load_model():
    """Load model from pickle file"""
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['mlb'], data['le']
    else:
        return train_model()

# ==================== PREDICTION FUNCTION ====================
def predict_disease(symptoms):
    """Predict disease based on selected symptoms"""
    if not symptoms:
        return None, None
    
    model, mlb, le = load_model()
    
    # Encode symptoms
    X_test = mlb.transform([symptoms])
    
    # Get probabilities
    probabilities = model.predict_proba(X_test)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = le.classes_[top_indices]
    top_probs = probabilities[top_indices]
    
    results = list(zip(top_diseases, top_probs))
    return results, model, mlb, le

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-title">🏥 Medical Diagnosis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">AI-Powered Preliminary Health Assessment Tool</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Patient Information")
        
        # Symptoms selection
        st.subheader("Select Your Symptoms")
        selected_symptoms = st.multiselect(
            "Choose all applicable symptoms:",
            options=ALL_SYMPTOMS,
            default=[],
            help="Select one or more symptoms you're experiencing"
        )
        
        st.divider()
        
        # File upload (optional)
        st.subheader("Upload Medical Report (Optional)")
        uploaded_file = st.file_uploader(
            "Upload PDF report:",
            type=["pdf"],
            help="You can optionally upload a medical report for reference"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.divider()
        
        # Prediction button
        predict_button = st.button(
            "🔍 Predict Disease",
            use_container_width=True,
            type="primary"
        )
    
    # Main panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            if not selected_symptoms:
                st.error("❌ Please select at least one symptom!")
            else:
                with st.spinner("🔄 Analyzing symptoms..."):
                    results, model, mlb, le = predict_disease(selected_symptoms)
                    
                    if results:
                        # Display top prediction
                        st.markdown("""
                            <div class="prediction-box">
                                <div class="prediction-title">🎯 Most Likely Condition</div>
                                <div class="prediction-value">{}</div>
                                <div class="confidence">Confidence: {:.1f}%</div>
                            </div>
                        """.format(results[0][0], results[0][1] * 100), unsafe_allow_html=True)
                        
                        # Top 3 predictions table
                        st.subheader("📊 Top 3 Predictions")
                        
                        pred_df = pd.DataFrame({
                            'Rank': ['1st', '2nd', '3rd'],
                            'Disease': [r[0] for r in results],
                            'Probability': [f"{r[1]*100:.1f}%" for r in results]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        st.subheader("📈 Probability Distribution")
                        chart_data = pd.DataFrame({
                            'Disease': [r[0] for r in results],
                            'Probability': [r[1]*100 for r in results]
                        })
                        st.bar_chart(chart_data.set_index('Disease'), height=300)
                        
                        # Precautions
                        st.subheader("💊 Recommended Precautions")
                        top_disease = results[0][0]
                        
                        if top_disease in PRECAUTIONS:
                            for i, precaution in enumerate(PRECAUTIONS[top_disease], 1):
                                st.info(f"**{i}.** {precaution}")
                        
                        # Download report
                        st.divider()
                        st.subheader("📥 Generate Report")
                        
                        report_text = f"""
MEDICAL DIAGNOSIS SYSTEM - REPORT
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYMPTOMS SELECTED:
{chr(10).join(f'• {s}' for s in selected_symptoms)}

PREDICTIONS:
{chr(10).join(f'{i+1}. {r[0]}: {r[1]*100:.1f}%' for i, r in enumerate(results))}

RECOMMENDED PRECAUTIONS FOR: {top_disease}
{chr(10).join(f'• {p}' for p in PRECAUTIONS.get(top_disease, []))}

DISCLAIMER:
This is not a medical diagnosis. This system provides a preliminary assessment only.
Please consult a qualified healthcare professional for accurate diagnosis and treatment.

====================================
"""
                        st.download_button(
                            label="📄 Download Report",
                            data=report_text,
                            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    with col2:
        st.subheader("📌 Quick Info")
        st.info(f"""
        **Symptoms Selected:** {len(selected_symptoms)}
        
        **Total Symptoms:** {len(ALL_SYMPTOMS)}
        
        **Diseases:** {len(DISEASES)}
        """)
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Disclaimer:</strong> This is not a medical diagnosis. Consult a doctor.</p>
        <p>Developed for educational purposes | 2024</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
