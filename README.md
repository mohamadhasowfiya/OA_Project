# AI-Assisted Osteoarthritis (OA) Screening System

## Project Overview

This project is an AI-assisted screening system designed to identify potential knee osteoarthritis (OA) severity levels using X-ray images and patient health information.

The system combines Deep Learning and Machine Learning techniques to support early screening and risk assessment.

## Technologies Used

- Python
- TensorFlow
- MobileNetV2
- LightGBM
- Flask
- HTML and CSS
- SQLite

## Key Features

- Knee X-ray image classification.
- OA severity classification into five categories.
- Patient health questionnaire.
- AI-assisted risk prediction.
- Patient screening history.
- Screening reports.
- Risk assessment dashboard.

## Model Information

**Deep Learning:** MobileNetV2 for X-ray image classification.

**Machine Learning:** LightGBM for patient risk assessment using health and questionnaire features.

## Dataset

The project uses a knee osteoarthritis X-ray dataset with five Kellgren-Lawrence severity classes.

The original dataset is not included in this repository.

Dataset source: [Kaggle – Knee Osteoarthritis Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

## Project Structure

```text
OA_Project/
├── web_app/
│   ├── templates/
│   ├── app.py
│   ├── database.py
│   ├── model_predict.py
│   └── risk_predict.py
├── mobilenetv2_oa_model.keras
├── mobilenetv2_best.keras
├── lightgbm_oa_risk_model.pkl
├── requirements.txt
├── sample_dataset/
└── README.md
```

## Installation

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
cd web_app
python app.py
```

4. Open your browser at:

```text
http://127.0.0.1:5000
```

## Disclaimer

This project is an AI-assisted screening and risk-assessment prototype. It is not a final medical diagnosis. Clinical decisions should be made by qualified healthcare professionals.
