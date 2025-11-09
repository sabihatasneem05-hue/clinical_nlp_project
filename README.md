# 🧠 Data-Centric AI for Clinical NLP
**End-to-End Weak Supervision, Active Learning, and Fairness Analysis for Healthcare Text**

## 📋 Overview
This project implements a Data-Centric AI (DCAI) approach to Clinical NLP — combining weak supervision, active learning, calibration, and fairness analysis.

It demonstrates:
- Automated ICD-10 coding from discharge summaries
- Adverse Drug Event (ADE) detection
- Fairness and bias analysis across patient subgroups
- A beautiful Streamlit dashboard for interactive execution

## 🖼️ UI Preview
![Streamlit Clinical NLP UI Preview](https://via.placeholder.com/1200x650.png?text=Streamlit+Clinical+NLP+Dashboard)

## 🧩 Features
✅ ICD Coding Automation — Suggests ICD-10 diagnostic codes  
💊 ADE Detection — Detects drug-induced reactions  
⚖️ Fairness Analysis — Evaluates subgroup disparities  
🔍 Negation & Uncertainty Detection — Handles linguistic nuance  
🧪 Active Learning — Selects uncertain samples for human review  
📊 Comprehensive Metrics — Accuracy, calibration, and fairness  

## 🗂️ Project Structure
```
clinical_nlp_project/
├── dcai_clinical_nlp.py
├── app_ui.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation
### 1️⃣ Clone the Repository
```
git clone https://github.com/<your-username>/clinical_nlp_project.git
cd clinical_nlp_project
```
### 2️⃣ Create a Virtual Environment
```
python -m venv venv
.env\Scriptsctivate
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

## 🚀 Run the Application
### ▶️ Streamlit Dashboard
```
streamlit run app_ui.py
```
### ▶️ Command-Line Demonstration
```
python dcai_clinical_nlp.py
```

## 🧾 Requirements
```
numpy
pandas
scikit-learn
scipy
streamlit
```

## 🧑‍💻 Author
**Sabi**  
Senior Python Developer | Clinical NLP & Data-Centric AI
