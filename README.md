# 🛡️ SpamShield AI

**AI-Powered Email Spam Detection System**

SpamShield AI is a premium, portfolio-quality web application that detects spam
emails in real time using a TF-IDF + Multinomial Naive Bayes machine learning
pipeline, wrapped in a modern glassmorphic SaaS dashboard built with Streamlit.

---

## ✨ Features

- 🔍 **Real-time spam detection** — paste any email text and get an instant prediction
- 📊 **Confidence score, risk level & prediction probability** for every scan
- 🎯 **Model performance metrics** (accuracy, precision, recall) shown live in the sidebar
- 📈 **Interactive analytics** — spam vs. safe distribution, confidence gauge, prediction history trend
- 🕓 **Prediction history table** with per-session tracking
- ⬇️ **Export results as CSV** and **download a full text report**
- 🎨 **Modern glassmorphic UI** inspired by Stripe, Notion, Linear and Perplexity
- 📱 **Fully responsive** layout

---

## 🧰 Tech Stack

| Layer            | Technology                     |
|-------------------|--------------------------------|
| UI / Frontend      | Streamlit + Custom CSS         |
| ML Model           | Scikit-Learn (Multinomial NB)  |
| Feature Extraction | TF-IDF Vectorizer              |
| Data Handling      | Pandas, NumPy                  |
| Visualization      | Plotly                         |
| Model Persistence  | Joblib                         |

---

## 📁 Project Structure

```
spamshield-ai/
├── app.py                 # Streamlit dashboard (UI + inference)
├── train_model.py         # Dataset generation + model training script
├── model.pkl              # Trained Multinomial Naive Bayes model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── model_metrics.json     # Saved accuracy / precision / recall metrics
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── assets/
│   └── logo.png           # (optional) app logo
└── data/
    └── spam.csv           # Labeled email dataset
```

---

## ⚙️ Installation

**1. Clone or download the project**

```bash
git clone https://github.com/<your-username>/spamshield-ai.git
cd spamshield-ai
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the model**

```bash
python train_model.py
```

This generates `data/spam.csv` (if not already present), trains the TF-IDF +
Naive Bayes classifier, and saves `model.pkl`, `vectorizer.pkl`, and
`model_metrics.json`.

**5. Run the app**

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🚀 Usage

1. Paste any email content into the **"Paste your email content here..."** box.
2. Click **Analyze Email**.
3. View the result — **🚨 Spam Detected** or **✅ Safe Email** — along with
   confidence score, risk/trust level, and prediction probability.
4. Check the **Analytics Dashboard** for spam vs. safe distribution and a
   confidence trend across your session.
5. Export your scan history as **CSV** or download a full **prediction report**.

---

## 📸 Screenshots

> Add screenshots of the hero section, analysis panel, and analytics dashboard here
> after running the app locally.

```
assets/screenshot-hero.png
assets/screenshot-results.png
assets/screenshot-analytics.png
```

---

## 🧪 Model Details

- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (unigrams + bigrams, top 3000 features)
- **Training data:** Balanced synthetic dataset of spam and legitimate (ham)
  emails covering common categories — promotional scams, phishing, fake
  lottery/prize offers, loan/finance scams, and everyday academic/work
  correspondence.

> 📝 **Note:** The bundled `data/spam.csv` is a synthetically generated demo
> dataset so the project runs out-of-the-box with no external downloads. For
> production use, replace it with a real-world labeled dataset (e.g. the
> SMS Spam Collection or Enron-Spam dataset) and re-run `train_model.py` —
> the pipeline will retrain automatically on any CSV with `text` and `label`
> columns.

---

## 🔮 Future Improvements

- [ ] Add support for `.eml` file uploads
- [ ] Integrate deep learning models (LSTM / DistilBERT) for comparison
- [ ] Add multi-language spam detection
- [ ] User authentication and persistent (database-backed) history
- [ ] Deploy to Streamlit Community Cloud / Docker container
- [ ] Add explainability (highlight spam-triggering words in the email)

---

## 👤 Author

**SpamShield AI** — built as a portfolio / academic machine learning project
demonstrating end-to-end ML app development: data preparation, model
training, and production-style deployment with Streamlit.

Feel free to fork, star ⭐, and customize this project for your own use.

---

## 📄 License

This project is open-source and available under the MIT License.
