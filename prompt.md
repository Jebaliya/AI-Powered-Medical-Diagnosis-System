Create a complete AI-Powered Medical Diagnosis System using Python and Streamlit.

The application must be fully functional, clean, and professional, and ready to deploy on Streamlit Cloud without errors.

---

## 🎯 PROJECT REQUIREMENTS

Build a web app where users can:

1. Select multiple symptoms from a predefined list
2. Upload a medical report (PDF optional)
3. Click a "Predict Disease" button
4. Get:

   * Top 3 predicted diseases
   * Probability scores
   * Suggested precautions
   * Simple visualization (bar chart)

---

## 🧠 MACHINE LEARNING REQUIREMENTS

* Use a lightweight ML model:

  * Random Forest OR Naive Bayes

* Train using a small synthetic dataset (you can generate inside code)

* Use Label Encoding for diseases

* Use MultiLabelBinarizer for symptoms

* Save and load model using pickle

* Ensure:

  * Fast prediction
  * Small model size (<50MB)
  * No external API dependency

---

## 🎨 UI / UX REQUIREMENTS (VERY IMPORTANT)

Design a modern, clean, professional UI:

* Use Streamlit layout features:

  * Sidebar for inputs
  * Main panel for results

* Add:

  * Title with emoji 🏥
  * Subheader description
  * Section dividers
  * Cards/containers for output

* Input UI:

  * Multiselect dropdown for symptoms
  * File uploader (PDF only)
  * Predict button

* Output UI:

  * Highlight predicted disease (large font)
  * Show top 3 predictions with percentages
  * Bar chart for probabilities
  * Expandable section for precautions

* Add:

  * Loading spinner during prediction
  * Success / warning messages
  * Footer with developer name

---

## 📁 FILE STRUCTURE

Generate all required files:

1. app.py (main Streamlit app)
2. model.pkl (trained model)
3. symptoms list inside code or CSV
4. precautions dictionary (JSON or Python dict)
5. requirements.txt

---

## 📦 DEPLOYMENT REQUIREMENTS

* Ensure compatibility with Streamlit Cloud

* Avoid heavy libraries

* Do NOT use:

  * OpenAI API
  * Large deep learning models
  * GPU dependencies

* requirements.txt should include only:
  streamlit
  pandas
  numpy
  scikit-learn
  PyPDF2

---

## 💡 EXTRA FEATURES (OPTIONAL BUT PREFERRED)

* Add download report button (text format)
* Show uploaded file name
* Add simple disclaimer:
  "This is not a medical diagnosis. Consult a doctor."

---

## 🧾 CODE QUALITY

* Write clean, well-structured code
* Use functions for:

  * model loading
  * prediction
  * UI sections
* Add comments for understanding
* Ensure no errors when running:
  streamlit run app.py

---

## 🎯 OUTPUT FORMAT

Provide:

1. Full app.py code
2. requirements.txt
3. Any additional files content

Make sure everything works directly when copied into VS Code and deployed.
