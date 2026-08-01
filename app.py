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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FORCE NATIVE DARK THEME ====================
# Native Streamlit components (dataframe grid, bar chart, etc.) render inside
# sandboxed iframes and only respect Streamlit's own theme engine, not page CSS.
# Setting these options at runtime forces every built-in widget to render dark,
# without requiring a separate .streamlit/config.toml file.
try:
    st._config.set_option("theme.base", "dark")
    st._config.set_option("theme.primaryColor", "#4f8bff")
    st._config.set_option("theme.backgroundColor", "#0b0c0e")
    st._config.set_option("theme.secondaryBackgroundColor", "#17181c")
    st._config.set_option("theme.textColor", "#f2f2f3")
    st._config.set_option("theme.font", "sans serif")
except Exception:
    pass

# ==================== GLOBAL STYLES (DARK THEME) ====================
st.markdown("""
    <style>

    /* ---------- Design Tokens ---------- */
    :root {
        --bg-primary: #0b0c0e;
        --bg-secondary: #121316;
        --bg-tertiary: #17181c;
        --bg-elevated: #1c1d22;
        --border-color: #2a2b30;
        --border-color-hover: #3a3b42;
        --text-primary: #f2f2f3;
        --text-secondary: #a3a5ab;
        --text-muted: #6b6d75;
        --accent: #4f8bff;
        --accent-hover: #6c9dff;
        --accent-soft: rgba(79, 139, 255, 0.12);
        --success: #3ecf8e;
        --warning: #e8b339;
        --danger: #ef5b5b;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.35);
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ---------- Global App Background ---------- */
    html, body, [data-testid="stApp"], .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-main);
    }

    [data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
        border-bottom: 1px solid var(--border-color);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: var(--font-main);
        letter-spacing: -0.015em;
    }

    p, span, label, div {
        font-family: var(--font-main);
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: var(--border-color) !important;
        margin: 1.5rem 0 !important;
    }

    /* ---------- Cards / Containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--border-color-hover) !important;
    }

    /* ---------- Prediction Hero Card ---------- */
    .prediction-card {
        background: linear-gradient(160deg, #1a1d2b 0%, #14151b 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 28px 32px;
        margin: 4px 0 20px 0;
        box-shadow: var(--shadow-md);
    }

    .prediction-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 10px;
    }

    .prediction-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .prediction-confidence {
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--success);
        background: rgba(62, 207, 142, 0.12);
        border: 1px solid rgba(62, 207, 142, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* ---------- Disclaimer Banner ---------- */
    .disclaimer-banner {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--warning);
        border-radius: var(--radius-md);
        padding: 14px 18px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.5;
    }

    .disclaimer-banner strong {
        color: var(--text-primary);
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        color: var(--text-muted);
        padding: 24px 0 8px 0;
        font-size: 0.8rem;
        line-height: 1.6;
        border-top: 1px solid var(--border-color);
        margin-top: 1.5rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1rem !important;
        transition: background-color 0.18s ease, transform 0.06s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: var(--accent-hover) !important;
    }

    .stButton > button:active, .stDownloadButton > button:active {
        transform: scale(0.98);
    }

    .stDownloadButton > button {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background-color: var(--bg-elevated) !important;
    }

    /* ---------- Inputs / Multiselect / File Uploader ---------- */
    [data-baseweb="select"] > div, .stMultiSelect > div > div {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }

    [data-baseweb="select"] > div:hover {
        border-color: var(--border-color-hover) !important;
    }

    [data-baseweb="select"] input {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }

    [data-baseweb="select"] svg {
        fill: var(--text-secondary) !important;
    }

    [data-baseweb="tag"] {
        background-color: var(--accent-soft) !important;
        border: 1px solid rgba(79, 139, 255, 0.35) !important;
        color: var(--accent-hover) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ---------- Multiselect Dropdown Popover ---------- */
    /* Cap the options list height so an open dropdown stays compact and
       scrollable instead of expanding over the button and sections below it. */
    div[data-baseweb="popover"] {
        z-index: 9999 !important;
    }

    div[data-baseweb="popover"] ul[role="listbox"],
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] ul {
        max-height: 220px !important;
        overflow-y: auto !important;
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li {
        color: var(--text-primary) !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: var(--accent-soft) !important;
    }

    [data-baseweb="tag"] * {
        color: var(--accent-hover) !important;
        fill: var(--accent-hover) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-tertiary) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: var(--radius-md) !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
    }

    section[data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }

    section[data-testid="stFileUploader"] svg {
        fill: var(--text-secondary) !important;
    }

    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploader"] button {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploader"] button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ---------- Tables / DataFrames ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    /* ---------- Alerts / Status Messages ---------- */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    /* ---------- Metric-style info tile ---------- */
    .info-tile {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    .info-tile-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
    }

    .info-tile-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ---------- Precaution rows ---------- */
    .precaution-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 10px 14px;
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        margin-bottom: 8px;
        transition: border-color 0.18s ease;
    }

    .precaution-row:hover {
        border-color: var(--border-color-hover);
    }

    .precaution-index {
        min-width: 22px;
        height: 22px;
        border-radius: 50%;
        background-color: var(--accent-soft);
        color: var(--accent-hover);
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }

    .precaution-text {
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.5;
    }

    /* ---------- Spinner text ---------- */
    .stSpinner > div > div {
        color: var(--text-secondary) !important;
    }

    /* ---------- Bar chart container spacing ---------- */
    [data-testid="stVegaLiteChart"] {
        margin-top: 4px;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-elevated);
        border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-color-hover);
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
    st.markdown('<div class="app-title">Medical Diagnosis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">AI-powered preliminary health assessment tool</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-banner"><strong>Notice.</strong> '
        'This tool does not provide a medical diagnosis. It offers a preliminary, '
        'model-generated assessment only. Please consult a qualified healthcare '
        'professional for accurate diagnosis and treatment.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Sidebar for inputs
    with st.sidebar:
        st.header("Patient Information")

        st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)
        selected_symptoms = st.multiselect(
            "Select all applicable symptoms",
            options=ALL_SYMPTOMS,
            default=[],
            help="Select one or more symptoms you are experiencing",
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown('<div class="section-label">Medical Report</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload PDF report (optional)",
            type=["pdf"],
            help="You can optionally upload a medical report for reference",
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
        

        st.divider()
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")

        predict_button = st.button(
            "Predict Disease",
            use_container_width=True,
            type="primary"
        )

    # Main panel
    col1, col2 = st.columns([2, 1])

    with col1:
        if predict_button:
            if not selected_symptoms:
                st.error("Please select at least one symptom before continuing.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    results, model, mlb, le = predict_disease(selected_symptoms)

                    if results:
                        # Display top prediction
                        st.markdown("""
                            <div class="prediction-card">
                                <div class="prediction-eyebrow">Most Likely Condition</div>
                                <div class="prediction-value">{}</div>
                                <div class="prediction-confidence">Confidence {:.1f}%</div>
                            </div>
                        """.format(results[0][0], results[0][1] * 100), unsafe_allow_html=True)

                        # Top 3 predictions table
                        with st.container(border=True):
                            st.markdown('<div class="section-label">Top 3 Predictions</div>', unsafe_allow_html=True)

                            pred_df = pd.DataFrame({
                                'Rank': ['1st', '2nd', '3rd'],
                                'Disease': [r[0] for r in results],
                                'Probability': [f"{r[1]*100:.1f}%" for r in results]
                            })

                            st.dataframe(pred_df, use_container_width=True, hide_index=True)

                        st.write("")

                        # Bar chart
                        with st.container(border=True):
                            st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
                            chart_data = pd.DataFrame({
                                'Disease': [r[0] for r in results],
                                'Probability': [r[1]*100 for r in results]
                            })
                            st.bar_chart(chart_data.set_index('Disease'), height=280)

                        st.write("")

                        # Precautions
                        with st.container(border=True):
                            st.markdown('<div class="section-label">Recommended Precautions</div>', unsafe_allow_html=True)
                            top_disease = results[0][0]

                            if top_disease in PRECAUTIONS:
                                precautions_html = ""
                                for i, precaution in enumerate(PRECAUTIONS[top_disease], 1):
                                    precautions_html += (
                                        '<div class="precaution-row">'
                                        f'<div class="precaution-index">{i}</div>'
                                        f'<div class="precaution-text">{precaution}</div>'
                                        '</div>'
                                    )
                                st.markdown(precautions_html, unsafe_allow_html=True)

                        st.write("")

                        # Download report
                        with st.container(border=True):
                            st.markdown('<div class="section-label">Generate Report</div>', unsafe_allow_html=True)

                            report_text = f"""
MEDICAL DIAGNOSIS SYSTEM - REPORT
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYMPTOMS SELECTED:
{chr(10).join(f'- {s}' for s in selected_symptoms)}

PREDICTIONS:
{chr(10).join(f'{i+1}. {r[0]}: {r[1]*100:.1f}%' for i, r in enumerate(results))}

RECOMMENDED PRECAUTIONS FOR: {top_disease}
{chr(10).join(f'- {p}' for p in PRECAUTIONS.get(top_disease, []))}

DISCLAIMER:
This is not a medical diagnosis. This system provides a preliminary assessment only.
Please consult a qualified healthcare professional for accurate diagnosis and treatment.

====================================
"""
                            st.download_button(
                                label="Download Report",
                                data=report_text,
                                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
        else:
            with st.container(border=True):
                st.markdown('<div class="section-label">Getting Started</div>', unsafe_allow_html=True)
                st.write(
                    "Select your symptoms from the sidebar, then click "
                    "**Predict Disease** to generate a preliminary assessment, "
                    "a probability breakdown, and recommended precautions."
                )

    with col2:
        st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"""
                <div class="info-tile">
                    <div class="info-tile-label">Symptoms Selected</div>
                    <div class="info-tile-value">{len(selected_symptoms)}</div>
                </div>
                <div class="info-tile">
                    <div class="info-tile-label">Total Symptoms Tracked</div>
                    <div class="info-tile-value">{len(ALL_SYMPTOMS)}</div>
                </div>
                <div class="info-tile" style="margin-bottom:0;">
                    <div class="info-tile-label">Conditions Modeled</div>
                    <div class="info-tile-value">{len(DISEASES)}</div>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="app-footer">
            This is not a medical diagnosis. Consult a doctor.<br>
            Developed for learning purposes | 2026
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
