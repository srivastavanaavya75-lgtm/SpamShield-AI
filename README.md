# 🛡️ SpamShield AI

### Machine Learning Powered Email Spam Detection Platform

SpamShield AI is a machine learning-based email spam detection platform built using Python, Streamlit, Scikit-Learn, and Plotly.

The application analyzes email content in real time using a TF-IDF Vectorizer and Multinomial Naive Bayes classifier to identify spam and legitimate emails. It provides confidence scoring, session analytics, prediction history, interactive visualizations, CSV export, and downloadable reports through a modern dashboard interface.

---

## ✨ Features

- 🔍 Real-Time Email Spam Detection
- 🤖 TF-IDF + Multinomial Naive Bayes Classification
- 📊 Confidence Score & Spam Probability
- 🚨 Risk Level Assessment
- 📈 Interactive Analytics Dashboard
- 🕒 Prediction History Tracking
- 📋 Session-Based Scan Analytics
- 📤 Export Results as CSV
- 📄 Download Prediction Reports
- 🎨 Modern Dashboard UI with Custom Styling
- 📱 Responsive Layout

---

## 📊 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 96.5% |
| Precision | 97.75% |
| Recall | 94.57% |
| F1 Score | 96.13% |

---

## 🧰 Tech Stack

| Layer | Technology |
|---------|-------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Algorithm | Multinomial Naive Bayes |
| Feature Engineering | TF-IDF Vectorizer |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Storage | Joblib |
| Programming Language | Python |

---

## 📂 Project Structure

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

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/srivastavanaavya75-lgtm/SpamShield-AI.git
cd SpamShield-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model

```bash
python train_model.py
```

This generates:

- model.pkl
- vectorizer.pkl
- model_metrics.json

---

### 5️⃣ Run Application

```bash
streamlit run app.py
```

Application will open at:

```text
http://localhost:8501
```

---

## 🚀 How It Works

### Step 1

Paste email content into the input area.

### Step 2

Click **Analyze Email**.

### Step 3

The system converts text into TF-IDF features.

### Step 4

The trained Multinomial Naive Bayes model predicts whether the email is:

- 🚨 Spam
- ✅ Safe

### Step 5

View:

- Confidence Score
- Spam Probability
- Risk Level
- Prediction History
- Session Analytics

---

## 📸 Application Preview

### Dashboard

> Add screenshot here

```text
screenshots/dashboard.png
```

### Spam Detection Result

> Add screenshot here

```text
screenshots/spam-result.png
```

### Analytics Dashboard

> Add screenshot here

```text
screenshots/analytics.png
```

---

## 📊 Analytics Features

- Spam vs Safe Distribution Chart
- Prediction Confidence Trend
- Session Statistics
- Historical Prediction Tracking
- CSV Export
- Report Download

---

## 🧠 Machine Learning Pipeline

```text
Email Text
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
Confidence & Analytics
```

---

## 📁 Dataset Information

The project uses a labeled dataset containing spam and legitimate email samples.

Categories include:

- Lottery Scams
- Prize Scams
- Banking Fraud
- Phishing Attempts
- Promotional Spam
- Academic Emails
- Professional Communication
- General Correspondence

---

## 🚀 Future Enhancements

- Email File (.eml) Support
- Deep Learning Models (BERT / DistilBERT)
- Multi-Language Spam Detection
- User Authentication System
- Cloud Deployment
- Database Integration
- Explainable AI for Spam Keywords
- Real Email Inbox Integration

---

## 🌐 Deployment

Run locally using:

```bash
streamlit run app.py
```

Future deployment options:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Docker
- AWS

---

## 👩‍💻 Author

**Naavya Srivastava**

B.Tech CSE (Data Science)

GitHub:
https://github.com/srivastavanaavya75-lgtm

LinkedIn:
https://www.linkedin.com/in/naavya-srivastava-661bb03ba/

---

## ⭐ Support

If you found this project useful:

- Star the repository ⭐
- Fork the project 🍴
- Share feedback 💡

---

## 📄 License

This project is licensed under the MIT License.

---

### Built with ❤️ using Python, Streamlit, Scikit-Learn and Plotly
