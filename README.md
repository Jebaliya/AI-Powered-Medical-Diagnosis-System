# Medical Diagnosis System

An AI-powered preliminary health assessment tool built with Streamlit. Users select the symptoms they are experiencing, and a Random Forest classifier predicts the most likely conditions along with confidence scores and recommended precautions.

> **Disclaimer:** This tool does not provide a medical diagnosis. It offers a preliminary, model-generated assessment only. Always consult a qualified healthcare professional for accurate diagnosis and treatment.

## Features

- **Symptom-based prediction** — select one or more symptoms from a curated list and get a ranked prediction of the most likely condition.
- **Top 3 predictions** — view the top three matching conditions with their probability scores in a clean table.
- **Probability distribution chart** — a bar chart visualizing confidence across the top predictions.
- **Recommended precautions** — condition-specific guidance displayed for the top predicted result.
- **Optional medical report upload** — attach a PDF report for reference alongside the assessment.
- **Downloadable report** — generate and download a plain-text summary of symptoms, predictions, and precautions.
- **Modern dark UI** — a fully dark, enterprise-style interface built entirely with Streamlit and injected CSS (no external theme files required).

## Tech Stack

| Component        | Technology                          |
|-------------------|--------------------------------------|
| UI Framework      | Streamlit                           |
| Machine Learning  | scikit-learn (RandomForestClassifier)|
| Data Handling     | pandas, NumPy                       |
| Model Persistence | pickle                              |

## Project Structure

```
.
├── app.py          # Main application file (UI, model training/loading, prediction logic)
├── model.pkl        # Auto-generated on first run (trained model + encoders)
└── README.md         # Project documentation
```

## How It Works

1. On first run, the app generates a synthetic training dataset mapping symptoms to conditions and trains a `RandomForestClassifier`. The trained model and encoders are saved to `model.pkl`.
2. On subsequent runs, the app loads the saved model directly instead of retraining.
3. When a user selects symptoms and clicks **Predict Disease**, the symptoms are encoded with a `MultiLabelBinarizer` and passed to the model, which returns probabilities for each condition.
4. The top 3 predictions are displayed with their probabilities, along with a bar chart and precautions for the top result.
5. Users can download a text report summarizing the session.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
pip install streamlit pandas numpy scikit-learn
```

### Running the App

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

## Usage

1. Open the app and expand the sidebar (open by default).
2. Under **Symptoms**, select all symptoms you are experiencing.
3. (Optional) Upload a PDF medical report under **Medical Report** for reference.
4. Click **Predict Disease**.
5. Review the most likely condition, the top 3 predictions table, the probability chart, and the recommended precautions.
6. Click **Download Report** to save a text summary of the session.

## Notes

- The underlying model is trained on a small **synthetic** dataset for demonstration purposes and is not clinically validated.
- Predictions and precautions are illustrative only and must not be used as a substitute for professional medical advice.
- The dark theme is enforced both via injected CSS and Streamlit's runtime theme configuration (`st._config.set_option`), so native components (tables, charts) render consistently dark without requiring a separate `.streamlit/config.toml` file.

## License

This project was developed for learning purposes.
