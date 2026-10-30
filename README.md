# AI-Assisted Osteoarthritis (OA) Screening System

## 📌 Project Overview

This project is an AI-assisted screening system designed to identify potential knee osteoarthritis (OA) severity levels using X-ray images and patient health information.

The system combines Deep Learning and Machine Learning techniques to support early screening and risk assessment.

It aims to provide an accessible screening support tool that can help users understand potential OA severity and risk levels.

## 🎯 Objectives

- Support early screening of knee osteoarthritis.
- Classify knee X-ray images into different severity categories.
- Assess patient risk using health questionnaire information.
- Generate screening reports for patient reference.
- Provide an accessible web-based screening interface.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, MobileNetV2
- **Machine Learning:** LightGBM
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Database:** SQLite
- **Data Processing:** NumPy, Pandas

## ✨ Key Features

- Knee X-ray image upload and classification.
- OA severity classification into five categories.
- Patient registration and health questionnaire.
- AI-assisted risk prediction.
- Patient screening history.
- Screening report generation.
- Risk assessment dashboard.

## 🧠 Model Information

### 1. Deep Learning Model – MobileNetV2

MobileNetV2 is used for knee X-ray image classification.

The model classifies images into five categories:

1. Healthy
2. Doubtful
3. Minimal
4. Moderate
5. Severe

### 2. Machine Learning Model – LightGBM

LightGBM is used for patient risk assessment based on health questionnaire information and available input features.

The model provides predicted risk levels and confidence information.

## 📂 Dataset

The project uses a knee osteoarthritis X-ray dataset containing five severity classes based on the Kellgren-Lawrence grading system.

The original complete dataset is not included in this repository.

**Dataset source:** [Kaggle – Knee Osteoarthritis Dataset with Severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

## 📁 Project Structure

```text
OA_Project/
│
├── web_app/
│   ├── templates/
│   │   ├── history.html
│   │   ├── index.html
│   │   ├── patient_report.html
│   │   ├── patients.html
│   │   ├── reports.html
│   │   ├── screening.html
│   │   └── sensor.html
│   │
│   ├── app.py
│   ├── database.py
│   ├── model_predict.py
│   └── risk_predict.py
│
├── mobilenet_model.py
├── mobilenetv2_best.keras
├── mobilenetv2_oa_model.keras
├── lightgbm_oa_risk_model.pkl
├── generate_risk_dataset.py
├── train_lightgbm.py
├── oa_risk_training_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohamadhasowfiya/OA_Project.git
```

### Step 2: Navigate to the Project Folder

```bash
cd OA_Project
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask Application

```bash
cd web_app
python app.py
```

### Step 5: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 🔮 Future Enhancements

- Improve model classification performance.
- Integrate real-time sensor data.
- Add multilingual support.
- Improve accessibility for rural and remote healthcare settings.
- Enhance screening reports and risk assessment features.

## ⚠️ Disclaimer

This project is an AI-assisted screening and risk-assessment prototype. It is not a final medical diagnosis.

The predictions are intended only for screening support. Clinical decisions should be made by qualified healthcare professionals using appropriate clinical evaluation.

## 👩‍💻 Project Repository

GitHub: https://github.com/mohamadhasowfiya/OA_Project
