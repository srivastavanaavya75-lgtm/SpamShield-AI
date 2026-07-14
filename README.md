# 🛡️ SpamShield AI

<div align="center">

# Machine Learning Powered Email Spam Detection Platform

Detect spam emails instantly using Machine Learning, TF-IDF Vectorization, and Multinomial Naive Bayes Classification.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)]
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)]
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)]
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-blue?style=for-the-badge&logo=plotly)]
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]

### 🌐 Live Application

**🔗 https://spamshield-ai-cr4aj62tev39xzfjincyupr.streamlit.app**

</div>

---

## 📌 Overview

SpamShield AI is an intelligent email spam detection platform built using Machine Learning and deployed with Streamlit.

The system analyzes email content in real time using a TF-IDF Vectorizer and a Multinomial Naive Bayes classifier to determine whether an email is legitimate or spam.

Beyond basic classification, the application provides confidence scores, risk assessment, analytics dashboards, prediction history tracking, downloadable reports, and CSV exports through a modern interactive interface.

---

## ✨ Key Features

### 🔍 Real-Time Spam Detection
Analyze email content instantly using a trained machine learning model.

### 🤖 Machine Learning Classification
Powered by:

- TF-IDF Vectorization
- Multinomial Naive Bayes

### 📊 Confidence Scoring
Displays model confidence for every prediction.

### 🚨 Risk Assessment
Automatically categorizes emails into risk levels.

### 📈 Analytics Dashboard
Interactive visualizations powered by Plotly.

### 🕒 Prediction History
Tracks all scans performed during the session.

### 📤 Export Results
Download:

- CSV Reports
- Prediction Logs
- Session Analytics

### 🎨 Premium User Interface
Modern dashboard inspired by SaaS applications with custom styling.

### 📱 Responsive Design
Works across desktop and laptop devices.

---

# 📊 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 96.50% |
| Precision | 97.75% |
| Recall | 94.57% |
| F1 Score | 96.13% |

---

# 🧠 Machine Learning Pipeline

```text
Email Input
     │
     ▼
Text Preprocessing
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Multinomial Naive Bayes
     │
     ▼
Spam / Safe Prediction
     │
     ▼
Confidence Score
     │
     ▼
Analytics Dashboard
```

---

# 🛠️ Tech Stack

| Layer | Technology |
|---------|---------|
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Algorithm | Multinomial Naive Bayes |
| Feature Engineering | TF-IDF Vectorizer |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Model Storage | Joblib |
| Language | Python |

---

# 📂 Project Structure

```text
SpamShield-AI/
│
├── app.py
├── train_model.py
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── model_metrics.json
├── requirements.txt
├── logo.png
├── README.md
│
└── screenshots/
    ├── dashboard.png
    ├── spam-result.png
    └── analytics.png
```

---

# 📸 Application Screenshots

## Dashboard

<img width="1905" height="917" alt="image" src="https://github.com/user-attachments/assets/d63b1f67-916b-422b-81af-45dc012f1559" />


---

## Spam Detection Result

<img width="1917" height="1077" alt="image" src="https://github.com/user-attachments/assets/b13e3a1e-3f2c-4f83-ae73-5ef2b5cbd282" />


---

## Analytics Dashboard

<img width="1917" height="1077" alt="image" src="https://github.com/user-attachments/assets/00f19647-325c-4ffb-be6d-6e1bd21d005a" />


---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/srivastavanaavya75-lgtm/SpamShield-AI.git
cd SpamShield-AI
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Train the Model

```bash
python train_model.py
```

This generates:

```text
model.pkl
vectorizer.pkl
model_metrics.json
```

---

## 5️⃣ Run the Application

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

# 🚀 How To Use

### Step 1

Paste email content into the input box.

### Step 2

Click:

```text
Analyze Email
```

### Step 3

The application converts text into TF-IDF features.

### Step 4

The trained machine learning model predicts:

- 🚨 Spam
- ✅ Safe

### Step 5

Review:

- Confidence Score
- Spam Probability
- Risk Level
- Analytics Dashboard
- Prediction History

---

# 📈 Analytics Features

### Spam vs Safe Distribution

Visual representation of classification outcomes.

### Confidence Trend

Track confidence scores across scans.

### Session Statistics

Analyze spam detection activity.

### Historical Predictions

View previous classifications.

### CSV Export

Export complete prediction history.

### Report Download

Generate downloadable analysis reports.

---

# 📁 Dataset Information

The model is trained on labeled email samples covering multiple spam categories.

### Spam Categories

- Lottery Scams
- Prize Scams
- Phishing Emails
- Banking Fraud
- Fake Offers
- Promotional Spam

### Legitimate Categories

- Academic Communication
- Professional Emails
- General Correspondence
- Business Messages

---

# 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

### Live Demo

🔗 **https://spamshield-ai-cr4aj62tev39xzfjincyupr.streamlit.app**

---

# 🚀 Future Enhancements

- Email File Upload Support (.eml)
- Deep Learning Models (BERT / DistilBERT)
- Multi-Language Spam Detection
- User Authentication
- Database Integration
- Explainable AI Features
- Cloud Storage
- Real Email Inbox Integration
- Docker Support
- AWS Deployment

---

# 👩‍💻 Author

### Naavya Srivastava

B.Tech CSE (Data Science)

### GitHub

https://github.com/srivastavanaavya75-lgtm

### LinkedIn

https://www.linkedin.com/in/naavya-srivastava-661bb03ba/

---

# ⭐ Support

If you found this project useful:

⭐ Star the Repository

🍴 Fork the Project

💡 Share Feedback

🚀 Connect and Collaborate

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### Built with Python, Streamlit, Scikit-Learn and Plotly

🛡️ SpamShield AI • Intelligent Email Protection Through Machine Learning

</div>
