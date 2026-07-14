"""
SpamShield AI
--------------
AI-Powered Email Spam Detection System built with Streamlit, Scikit-Learn
(TF-IDF + Multinomial Naive Bayes) and Plotly.

Run:
    streamlit run app.py
"""

import html
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SpamShield AI | Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CONSTANTS / PALETTE
# ----------------------------------------------------------------------
VANILLA_CUSTARD = "#FFF9EB"
PISTACHIO_FROST = "#C5E384"
FOREST_GREEN = "#2E7D32"
MIDNIGHT_ESPRESSO = "#200F07"
BORDER_COLOR = "#D8E5B6"
DANGER = "#C62828"
TEXT_COLOR = "#200F07"
SECONDARY_TEXT = "#5E4B43"

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
METRICS_PATH = "model_metrics.json"


def styled_card(key):
    """
    st.container(key=...) requires Streamlit >= 1.34. On older installs
    this raises TypeError; fall back to a plain (unstyled) container so
    the app still runs instead of crashing. Upgrading Streamlit
    (see requirements.txt) restores the full glass-card styling.
    """
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()

# ----------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {TEXT_COLOR};
    }}

    .stApp {{
        background: linear-gradient(135deg, {VANILLA_CUSTARD} 0%, #F6F8E8 30%, #DDEDB3 70%, {PISTACHIO_FROST} 100%);
        background-attachment: fixed;
    }}

    #MainMenu, footer, header {{visibility: hidden;}}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    /* ---------------- TOP NAV ---------------- */
    .ss-navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {BORDER_COLOR};
        border-radius: 18px;
        padding: 14px 26px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.08);
    }}
    .ss-navbar-left {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 20px;
        color: {MIDNIGHT_ESPRESSO};
    }}
    .ss-badge {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(46, 125, 50, 0.12);
        border: 1px solid rgba(46, 125, 50, 0.35);
        color: {FOREST_GREEN};
        font-weight: 600;
        font-size: 13px;
        padding: 7px 14px;
        border-radius: 999px;
    }}
    .ss-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {FOREST_GREEN};
        box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.6);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.5); }}
        70% {{ box-shadow: 0 0 0 8px rgba(46, 125, 50, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(46, 125, 50, 0); }}
    }}

    /* ---------------- HERO ---------------- */
    .ss-hero {{
        text-align: center;
        padding: 18px 10px 6px 10px;
    }}
    .ss-hero h1 {{
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 46px;
        letter-spacing: -1px;
        margin-bottom: 6px;
        background: linear-gradient(90deg, {MIDNIGHT_ESPRESSO}, {FOREST_GREEN});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .ss-hero p {{
        font-size: 17px;
        color: {SECONDARY_TEXT};
        max-width: 620px;
        margin: 0 auto 26px auto;
        line-height: 1.5;
    }}

    /* ---------------- GLASS CARD ---------------- */
    .glass-card,
    .st-key-analyze_card,
    .st-key-pie_card,
    .st-key-line_card,
    .st-key-history_card {{
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {BORDER_COLOR};
        border-radius: 20px;
        padding: 26px 28px;
        box-shadow: 0 10px 34px rgba(46, 125, 50, 0.08);
        margin-bottom: 22px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .glass-card:hover,
    .st-key-analyze_card:hover,
    .st-key-pie_card:hover,
    .st-key-line_card:hover,
    .st-key-history_card:hover {{
        box-shadow: 0 14px 40px rgba(46, 125, 50, 0.14);
    }}

    /* ---------------- METRIC CARDS ---------------- */
    .metric-card {{
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 16px 34px rgba(46, 125, 50, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.7);
        border-color: rgba(46, 125, 50, 0.4);
    }}
    .metric-value {{
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: {FOREST_GREEN};
        margin-bottom: 2px;
    }}
    .metric-label {{
        font-size: 12.5px;
        font-weight: 600;
        color: {SECONDARY_TEXT};
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}

    /* ---------------- SECTION TITLE ---------------- */
    .ss-section-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 22px;
        color: {MIDNIGHT_ESPRESSO};
        margin: 8px 0 14px 2px;
    }}

    /* ---------------- TEXT AREA ---------------- */
    .stTextArea textarea {{
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1.5px solid {BORDER_COLOR} !important;
        border-radius: 14px !important;
        font-size: 15px !important;
        padding: 14px !important;
        color: {TEXT_COLOR} !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {FOREST_GREEN} !important;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.15) !important;
    }}

    /* ---------------- BUTTONS ---------------- */
    .stButton > button {{
        border-radius: 12px;
        font-weight: 700;
        font-size: 14.5px;
        padding: 10px 22px;
        border: none;
        transition: all 0.2s ease;
    }}
    div[data-testid="column"]:nth-of-type(1) .stButton > button {{
        background: linear-gradient(135deg, {FOREST_GREEN}, #3fa047);
        color: white;
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.35);
    }}
    div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(46, 125, 50, 0.45);
    }}
    div[data-testid="column"]:nth-of-type(2) .stButton > button {{
        background: rgba(200, 60, 60, 0.08);
        color: {DANGER};
        border: 1.5px solid rgba(198, 40, 40, 0.35);
    }}
    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {{
        background: rgba(198, 40, 40, 0.14);
        transform: translateY(-2px);
    }}

    /* ---------------- RESULT CARDS ---------------- */
    .result-spam {{
        background: linear-gradient(135deg, rgba(198, 40, 40, 0.14), rgba(198, 40, 40, 0.05));
        border: 2.5px solid {DANGER};
        border-radius: 18px;
        padding: 22px 26px;
        margin-top: 18px;
        animation: fadeIn 0.4s ease;
        box-shadow: 0 10px 30px rgba(198, 40, 40, 0.16);
        color: {TEXT_COLOR} !important;
    }}
    .result-safe {{
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.14), rgba(46, 125, 50, 0.05));
        border: 2.5px solid {FOREST_GREEN};
        border-radius: 18px;
        padding: 22px 26px;
        margin-top: 18px;
        animation: fadeIn 0.4s ease;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.16);
        color: {TEXT_COLOR} !important;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .result-title-spam {{
        font-family: 'Sora', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: {DANGER} !important;
        margin-bottom: 4px;
    }}
    .result-title-safe {{
        font-family: 'Sora', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: {FOREST_GREEN} !important;
        margin-bottom: 4px;
    }}
    .result-sub {{
        font-size: 14px;
        color: {SECONDARY_TEXT} !important;
        margin-bottom: 14px;
    }}

    .stat-chip {{
        display: inline-block;
        background: rgba(255, 255, 255, 0.85);
        border: 1.5px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 8px 14px;
        margin-right: 10px;
        margin-bottom: 8px;
        font-size: 13.5px;
        font-weight: 700;
        color: {TEXT_COLOR} !important;
    }}
    .result-spam .stat-chip {{
        border-color: rgba(198, 40, 40, 0.4);
    }}
    .result-safe .stat-chip {{
        border-color: rgba(46, 125, 50, 0.4);
    }}

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MIDNIGHT_ESPRESSO} 0%, #2b1a0f 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: {VANILLA_CUSTARD} !important;
    }}
    section[data-testid="stSidebar"] .glass-card {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 249, 235, 0.15);
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255, 249, 235, 0.15);
    }}

    /* ---------------- DOWNLOAD BUTTON ---------------- */
    .stDownloadButton > button {{
        border-radius: 12px;
        font-weight: 700;
        background: rgba(46, 125, 50, 0.1);
        color: {FOREST_GREEN};
        border: 1.5px solid rgba(46, 125, 50, 0.35);
    }}
    .stDownloadButton > button:hover {{
        background: rgba(46, 125, 50, 0.18);
    }}

    /* ---------------- CUSTOM HISTORY TABLE ---------------- */
    .ss-table-wrapper {{
        overflow-x: auto;
        border-radius: 14px;
        border: 1px solid {BORDER_COLOR};
    }}
    table.ss-history-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        background: rgba(255, 255, 255, 0.5);
    }}
    table.ss-history-table thead th {{
        background: linear-gradient(135deg, {FOREST_GREEN}, #3fa047);
        color: {VANILLA_CUSTARD} !important;
        font-weight: 700;
        text-align: left;
        padding: 12px 16px;
        position: sticky;
        top: 0;
    }}
    table.ss-history-table tbody td {{
        padding: 10px 16px;
        color: {TEXT_COLOR} !important;
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    table.ss-history-table tbody tr:nth-child(even) {{
        background: rgba(197, 227, 132, 0.18);
    }}
    table.ss-history-table tbody tr:nth-child(odd) {{
        background: rgba(255, 255, 255, 0.55);
    }}
    table.ss-history-table tbody tr:hover {{
        background: rgba(46, 125, 50, 0.14);
        transition: background 0.2s ease;
    }}
    .tag-spam {{
        color: {DANGER} !important;
        font-weight: 700;
    }}
    .tag-safe {{
        color: {FOREST_GREEN} !important;
        font-weight: 700;
    }}

    /* ---------------- RESPONSIVENESS ---------------- */
    @media (max-width: 992px) {{
        .ss-hero h1 {{ font-size: 38px; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    @media (max-width: 768px) {{
        .ss-navbar {{
            flex-direction: column;
            gap: 10px;
            text-align: center;
            padding: 14px 16px;
        }}
        .ss-hero h1 {{ font-size: 30px; }}
        .ss-hero p {{ font-size: 15px; }}
        .metric-value {{ font-size: 21px; }}
        .glass-card {{ padding: 18px 16px; }}
        .result-title-spam, .result-title-safe {{ font-size: 19px; }}
        .stat-chip {{ font-size: 12.5px; padding: 6px 10px; }}
        table.ss-history-table {{ font-size: 12px; }}
        table.ss-history-table thead th,
        table.ss-history-table tbody td {{ padding: 8px 10px; }}
    }}
    @media (max-width: 480px) {{
        .ss-hero h1 {{ font-size: 25px; }}
        .metric-card {{ padding: 14px 10px; }}
        .metric-value {{ font-size: 18px; }}
        .metric-label {{ font-size: 10.5px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# LOAD MODEL + METRICS (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    return model, vectorizer, metrics


model, vectorizer, metrics = load_artifacts()

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "email_input" not in st.session_state:
    st.session_state.email_input = ""

# ----------------------------------------------------------------------
# TOP NAVBAR
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="ss-navbar">
        <div class="ss-navbar-left">🛡️ SpamShield AI</div>
        <div class="ss-badge"><span class="ss-dot"></span> Model Active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# MODEL MISSING GUARD
# ----------------------------------------------------------------------
if model is None:
    st.error(
        "⚠️ Model artifacts not found. Please run `python train_model.py` first to "
        "generate `model.pkl` and `vectorizer.pkl`, then restart the app."
    )
    st.stop()

# ----------------------------------------------------------------------
# HERO SECTION
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="ss-hero">
        <h1>SpamShield AI</h1>
        <p>Detect spam emails instantly using machine learning-powered text classification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

hc1, hc2, hc3, hc4 = st.columns(4)
hero_stats = [
    (hc1, f"{metrics.get('accuracy', 0)}%", "Accuracy"),
    (hc2, "Naive Bayes", "Model Type"),
    (hc3, f"{metrics.get('dataset_size', 0):,}", "Dataset Size"),
    (hc4, f"{metrics.get('feature_count', 0):,}", "Total Features"),
]
for col, value, label in hero_stats:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div style="margin-bottom: 26px;"></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SpamShield AI")
    st.markdown("---")
    st.markdown("#### 📊 Model Information")
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="margin-bottom:8px;"><b>Algorithm:</b> {metrics.get('algorithm', 'Multinomial Naive Bayes')}</div>
            <div style="margin-bottom:8px;"><b>Vectorizer:</b> {metrics.get('vectorizer', 'TF-IDF')}</div>
            <div style="margin-bottom:8px;"><b>Accuracy:</b> {metrics.get('accuracy', 0)}%</div>
            <div style="margin-bottom:8px;"><b>Precision:</b> {metrics.get('precision', 0)}%</div>
            <div style="margin-bottom:8px;"><b>Recall:</b> {metrics.get('recall', 0)}%</div>
            <div style="margin-bottom:8px;"><b>F1 Score:</b> {metrics.get('f1_score', 0)}%</div>
            <div style="margin-bottom:8px;"><b>Dataset Size:</b> {metrics.get('dataset_size', 0):,} emails</div>
            <div><b>Features Count:</b> {metrics.get('feature_count', 0):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ℹ️ About")
    st.markdown(
        """
        <div class="glass-card">
        SpamShield AI is a machine learning powered spam email detection
        platform built using Scikit-Learn and Streamlit. It analyzes email
        text using TF-IDF features and a Multinomial Naive Bayes classifier
        to flag spam in real time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📈 Session Analytics")
    # Rendered into a placeholder so it can be refreshed the moment a new
    # prediction is added below, instead of showing stale counts from
    # before this run's analysis happened.
    session_analytics_slot = st.empty()

    def render_session_analytics():
        total = len(st.session_state.history)
        spam_count = sum(1 for h in st.session_state.history if h["prediction"] == "Spam")
        safe_count = total - spam_count
        session_analytics_slot.markdown(
            f"""
            <div class="glass-card">
                <div style="margin-bottom:6px;"><b>Total Scanned:</b> {total}</div>
                <div style="margin-bottom:6px;">🚨 <b>Spam Found:</b> {spam_count}</div>
                <div>✅ <b>Safe Emails:</b> {safe_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_session_analytics()

# ----------------------------------------------------------------------
# MAIN ANALYSIS PANEL
# ----------------------------------------------------------------------
st.markdown('<div class="ss-section-title">📩 Analyze an Email</div>', unsafe_allow_html=True)

analyze_card = styled_card("analyze_card")
with analyze_card:
    def clear_email_input():
        st.session_state.email_input = ""

    email_text = st.text_area(
        "Email content",
        height=180,
        placeholder="Paste your email content here...",
        label_visibility="collapsed",
        key="email_input",
    )

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        analyze_clicked = st.button("🔍 Analyze Email", width="stretch")
    with col_b:
        st.button("🗑️ Clear", width="stretch", on_click=clear_email_input)

    result_placeholder = st.container()

    if analyze_clicked:
        text_to_analyze = email_text.strip()
        if not text_to_analyze:
            st.warning("Please paste some email content before analyzing.")
        else:
            with st.spinner("🧠 Analyzing email with SpamShield AI..."):
                vec = vectorizer.transform([text_to_analyze])
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]  # [P(ham), P(spam)]
                spam_prob = proba[1] * 100
                ham_prob = proba[0] * 100

            is_spam = pred == 1
            confidence = spam_prob if is_spam else ham_prob

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "email_snippet": text_to_analyze[:60] + ("..." if len(text_to_analyze) > 60 else ""),
                "prediction": "Spam" if is_spam else "Safe",
                "confidence": round(confidence, 2),
                "spam_probability": round(spam_prob, 2),
                "safe_probability": round(ham_prob, 2),
            }
            st.session_state.history.insert(0, record)
            render_session_analytics()  # refresh sidebar counters immediately

            with result_placeholder:
                if is_spam:
                    risk_level = "High" if spam_prob >= 85 else ("Medium" if spam_prob >= 60 else "Low")
                    st.markdown(
                        f"""
                        <div class="result-spam">
                            <div class="result-title-spam">🚨 Spam Detected</div>
                            <div class="result-sub">This email shows strong characteristics of spam content.</div>
                            <span class="stat-chip">Confidence Score: {confidence:.1f}%</span>
                            <span class="stat-chip">Risk Level: {risk_level}</span>
                            <span class="stat-chip">Spam Probability: {spam_prob:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    trust_level = "High" if ham_prob >= 85 else ("Medium" if ham_prob >= 60 else "Low")
                    st.markdown(
                        f"""
                        <div class="result-safe">
                            <div class="result-title-safe">✅ Safe Email</div>
                            <div class="result-sub">This email looks legitimate based on its content.</div>
                            <span class="stat-chip">Confidence Score: {confidence:.1f}%</span>
                            <span class="stat-chip">Trust Score: {trust_level}</span>
                            <span class="stat-chip">Safe Probability: {ham_prob:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Confidence gauge
                gauge_color = DANGER if is_spam else FOREST_GREEN
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        number={"suffix": "%", "font": {"size": 30, "color": TEXT_COLOR}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR},
                            "bar": {"color": gauge_color},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 60], "color": "rgba(200,200,200,0.25)"},
                                {"range": [60, 85], "color": "rgba(197, 227, 132, 0.5)"},
                                {"range": [85, 100], "color": "rgba(46, 125, 50, 0.3)"},
                            ],
                        },
                        title={"text": "Prediction Confidence", "font": {"size": 15, "color": TEXT_COLOR}},
                    )
                )
                fig_gauge.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "color": TEXT_COLOR},
                    # NOTE: no layout-level "title"/"title_font_color" here - the
                    # indicator above already carries its own title. Setting an
                    # empty layout title previously rendered a stray "undefined"
                    # label above the gauge.
                )
                st.plotly_chart(fig_gauge, width="stretch")

# ----------------------------------------------------------------------
# ANALYTICS SECTION
# ----------------------------------------------------------------------
st.markdown('<div class="ss-section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    ac1, ac2 = st.columns(2)

    with ac1:
        pie_card = styled_card("pie_card")
        with pie_card:
            dist_counts = hist_df["prediction"].value_counts()
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=dist_counts.index,
                        values=dist_counts.values,
                        hole=0.55,
                        marker=dict(
                            colors=[DANGER if l == "Spam" else FOREST_GREEN for l in dist_counts.index],
                            line=dict(color=VANILLA_CUSTARD, width=2),
                        ),
                        textinfo="label+percent",
                        textfont=dict(color=VANILLA_CUSTARD, size=13, family="Inter"),
                        insidetextfont=dict(color=VANILLA_CUSTARD, size=13, family="Inter"),
                        outsidetextfont=dict(color=TEXT_COLOR, size=13, family="Inter"),
                    )
                ]
            )
            fig_pie.update_layout(
                title=dict(text="Spam vs Safe Distribution", font=dict(color=TEXT_COLOR, size=16)),
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": TEXT_COLOR},
                showlegend=True,
                legend=dict(font=dict(color=TEXT_COLOR, size=12)),
            )
            st.plotly_chart(fig_pie, width="stretch")

    with ac2:
        line_card = styled_card("line_card")
        with line_card:
            trend_df = hist_df.iloc[::-1].reset_index(drop=True)
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(
                    x=list(range(1, len(trend_df) + 1)),
                    y=trend_df["confidence"],
                    mode="lines+markers",
                    line=dict(color=FOREST_GREEN, width=3),
                    marker=dict(
                        size=8,
                        color=[DANGER if p == "Spam" else FOREST_GREEN for p in trend_df["prediction"]],
                    ),
                    name="Confidence",
                )
            )
            fig_line.update_layout(
                title=dict(text="Prediction History (Confidence over Time)", font=dict(color=TEXT_COLOR, size=16)),
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": TEXT_COLOR},
                legend=dict(font=dict(color=TEXT_COLOR, size=12)),
                xaxis=dict(
                    title=dict(text="Scan #", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    gridcolor="rgba(32, 15, 7, 0.08)",
                    zerolinecolor="rgba(32, 15, 7, 0.15)",
                ),
                yaxis=dict(
                    title=dict(text="Confidence (%)", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    gridcolor="rgba(32, 15, 7, 0.08)",
                    zerolinecolor="rgba(32, 15, 7, 0.15)",
                    range=[0, 100],
                ),
            )
            st.plotly_chart(fig_line, width="stretch")

    # Prediction history table + export
    history_card = styled_card("history_card")
    with history_card:
        st.markdown("**🕓 Prediction History**")

        table_cols = hist_df[["timestamp", "email_snippet", "prediction", "confidence"]]
        row_html_parts = []
        for _, row in table_cols.iterrows():
            tag_class = "tag-spam" if row["prediction"] == "Spam" else "tag-safe"
            icon = "🚨" if row["prediction"] == "Spam" else "✅"
            ts = html.escape(str(row["timestamp"]))
            snippet = html.escape(str(row["email_snippet"]))
            prediction = html.escape(str(row["prediction"]))
            row_html_parts.append(
                "<tr>"
                f"<td>{ts}</td>"
                f"<td>{snippet}</td>"
                f'<td class="{tag_class}">{icon} {prediction}</td>'
                f"<td>{row['confidence']:.1f}%</td>"
                "</tr>"
            )
        # Built as a single-line string (no embedded newlines/indentation) -
        # a multi-line indented f-string here previously got misread by
        # Streamlit's markdown parser as a code block, showing raw <tr>/<td>
        # tags as plain text instead of rendering as an HTML table.
        table_html = (
            '<div class="ss-table-wrapper"><table class="ss-history-table">'
            "<thead><tr><th>Timestamp</th><th>Email Snippet</th>"
            "<th>Prediction</th><th>Confidence</th></tr></thead>"
            f"<tbody>{''.join(row_html_parts)}</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "⬇️ Export Results CSV",
                data=csv_data,
                file_name=f"spamshield_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
            )
        with dcol2:
            report_text = (
                f"SpamShield AI - Prediction Report\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Scans: {len(hist_df)}\n"
                f"Spam Detected: {(hist_df['prediction'] == 'Spam').sum()}\n"
                f"Safe Emails: {(hist_df['prediction'] == 'Safe').sum()}\n\n"
                + hist_df.to_string(index=False)
            )
            st.download_button(
                "📄 Download Prediction Report",
                data=report_text.encode("utf-8"),
                file_name=f"spamshield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width="stretch",
            )
else:
    st.markdown(
        f"""
        <div class="glass-card" style="text-align:center; color:{SECONDARY_TEXT};">
        No predictions yet. Analyze an email above to see analytics here.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div style="text-align:center; margin-top: 30px; color: {SECONDARY_TEXT}; font-size: 13px;">
        Built with ❤️ using Streamlit, Scikit-Learn &amp; Plotly &nbsp;•&nbsp; SpamShield AI © 2025
    </div>
    """,
    unsafe_allow_html=True,
)