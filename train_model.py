"""
Model Training Script for Medical Diagnosis System
Run this once to generate and save the pre-trained model
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

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

# ==================== MODEL TRAINING ====================
def train_and_save_model():
    """Train the model and save it to pickle file"""
    print("🔄 Generating synthetic dataset...")
    X_train, y_train = generate_synthetic_dataset()
    print(f"✅ Dataset created with {len(X_train)} samples")
    
    print("\n🔄 Encoding data...")
    # Initialize encoders
    mlb = MultiLabelBinarizer()
    X_encoded = mlb.fit_transform(X_train)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    print(f"✅ Encoding complete. Features: {X_encoded.shape[1]}")
    
    print("\n🔄 Training Random Forest model...")
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_encoded, y_encoded)
    print("✅ Model training complete")
    
    print("\n🔄 Saving model to pickle file...")
    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'mlb': mlb, 'le': le}, f)
    print("✅ Model saved as 'model.pkl'")
    
    print("\n" + "="*50)
    print("📊 Model Information:")
    print(f"   Total diseases: {len(le.classes_)}")
    print(f"   Diseases: {', '.join(le.classes_)}")
    print(f"   Total symptoms: {len(mlb.classes_)}")
    print("="*50)

if __name__ == "__main__":
    train_and_save_model()
    print("\n✨ Ready to run: streamlit run app.py")
