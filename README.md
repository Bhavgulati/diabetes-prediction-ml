<div align="center">

# ⬡ DiabetesAI
### 🧬 Medical Intelligence Platform — Powered by AI, Built for Impact

[![Live Demo](https://img.shields.io/badge/🚀%20LIVE%20DEMO-Click%20Here-00d4ff?style=for-the-badge)](https://diabetes-prediction-ml-2-dt3u.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Bhavgulati-7c3aed?style=for-the-badge&logo=github)](https://github.com/Bhavgulati/diabetes-prediction-ml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge)](https://render.com)

<br/>

> **"Not just a student project. A production-grade Medical AI platform that any hospital developer would be proud to ship."**

<br/>

![DiabetesAI Banner](https://img.shields.io/badge/🩺%20Diabetes%20Risk%20Assessment-AI%20Powered-ff4d6d?style=for-the-badge)
![RAG](https://img.shields.io/badge/🧠%20RAG%20Chatbot-WHO%20+%20ADA%20Guidelines-10b981?style=for-the-badge)
![SHAP](https://img.shields.io/badge/🔬%20SHAP%20Explainability-XAI%20Powered-7c3aed?style=for-the-badge)

</div>

---

## 🔥 What is DiabetesAI?

DiabetesAI is a **full-stack Medical AI platform** that predicts diabetes risk using **Random Forest ML**, explains results using **SHAP (Explainable AI)**, and provides personalised medical guidance through a **RAG-powered chatbot grounded in WHO and ADA clinical guidelines**.

This isn't a Jupyter notebook. This is a **live, deployed, production application** with a REST API, clinical PDF reports, Google OAuth, rate limiting, model versioning, and data export — the kind of engineering you'd see at a real healthcare company.

---

## ⚡ Features That Will Blow Your Mind

### 🧠 Machine Learning Core
| Feature | Description |
|---|---|
| 🎯 **Risk Scoring** | `predict_proba()` gives exact probability — not just "diabetic/not diabetic" |
| 🔬 **SHAP Explainability** | Tells the user *why* they got their risk score — personalised per prediction |
| 🤖 **Model Comparison** | Random Forest vs Logistic Regression vs XGBoost — side by side accuracy metrics |
| 📊 **Feature Importance** | Visual chart of which biomarkers matter most |

### 🏥 Medical AI Chatbot (RAG-Powered)
| Feature | Description |
|---|---|
| 📚 **RAG Architecture** | Retrieves from WHO + ADA clinical guidelines before answering |
| 🎯 **Context-Aware** | Knows patient's exact risk score and biomarkers |
| 💬 **Auto-Explanation** | Automatically explains SHAP results in plain English on load |
| 🌍 **Language** | English 🇬🇧 |

### 📄 Clinical PDF Report
| Feature | Description |
|---|---|
| 🏥 **Hospital-Grade Design** | Navy header, risk stripe, patient info, biomarker table with NORMAL/HIGH/PRE-DIABETIC labels |
| 📊 **SHAP Analysis Table** | ▲ INCREASES RISK / ▼ REDUCES RISK per biomarker |
| 📋 **Clinical Recommendations** | Colour-coded by category: IMMEDIATE, MEDICAL, DIETARY, LIFESTYLE |
| ⚠️ **Urgency Banner** | Red/Amber/Green alert based on risk category |
| 🆔 **Unique Report ID** | `DAI-20260322` format — like a real hospital report |

### 🔧 Backend Engineering (Production-Grade)
| Feature | Description |
|---|---|
| 🛡️ **Rate Limiting** | 20 predictions/min · 30 chats/min · 10 API calls/min per IP |
| 📝 **Structured Logging** | Every prediction logged: `user=x glucose=180 risk=78% time=143ms` |
| 🔄 **Model Versioning** | Multiple model versions, `/api/predict?model_version=v1`, instant rollback |
| 📥 **Data Export** | `GET /api/export/csv` and `GET /api/export/json` — full prediction history |
| 🔌 **REST API** | Full API with `/api/predict`, `/api/health`, `/api/models`, `/api/docs` |
| 📡 **API Documentation** | Interactive docs at `/api/docs` |

### 🎨 Frontend Features
| Feature | Description |
|---|---|
| ❤️ **Health Score 0-100** | Circular gauge like Apple Health — holistic score across all biomarkers |
| 📈 **Comparison Mode** | Previous vs current risk — shows ▼ 33% improvement |
| 📊 **Progress Bar** | Shows form completion percentage in real time |
| 💡 **Tooltip Guidance** | Every field has medical guidance — what is normal, why it matters |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DiabetesAI Platform                   │
├─────────────────┬───────────────────┬───────────────────┤
│   Frontend      │    Flask Backend   │   ML Pipeline     │
│                 │                   │                   │
│ • Real-time     │ • Google OAuth    │ • Random Forest   │
│   risk preview  │ • Rate Limiting   │ • SHAP Engine     │
│ • Voice Input   │ • Struct Logging  │ • Model Versioning│
│ • Health Score  │ • REST API        │ • XGBoost         │
│ • Multi-lang    │ • Data Export     │ • Logistic Reg    │
├─────────────────┼───────────────────┼───────────────────┤
│   RAG Engine    │   Database        │   Infrastructure  │
│                 │                   │                   │
│ • WHO Guidelines│ • SQLite          │ • Docker          │
│ • ADA Standards │ • User History    │ • Gunicorn        │
│ • Keyword Search│ • Predictions     │ • Render Deploy   │
│ • Claude AI     │ • Google OAuth    │ • GitHub CI/CD    │
└─────────────────┴───────────────────┴───────────────────┘
```

---

## 🚀 Tech Stack

```python
tech_stack = {
    "ML":          ["scikit-learn", "Random Forest", "SHAP", "XGBoost"],
    "AI":          ["Claude (Anthropic)", "RAG", "WHO + ADA Guidelines"],
    "Backend":     ["Flask", "SQLAlchemy", "Flask-Login", "Authlib"],
    "Database":    ["SQLite", "Google OAuth"],
    "PDF":         ["ReportLab"],
    "DevOps":      ["Docker", "Gunicorn", "Render", "GitHub Actions"],
    "Testing":     ["pytest", "8 unit tests"],
    "Frontend":    ["Chart.js", "Web Speech API", "Vanilla JS"],
}
```

---

## 📡 REST API

```bash
# Health check
GET /api/health
→ {"status": "healthy", "rag": {"status": "ready", "total_chunks": 15}}

# Predict diabetes risk
POST /api/predict
{
  "pregnancies": 2, "glucose": 180, "bloodpressure": 80,
  "skinthickness": 35, "insulin": 100, "bmi": 34.0,
  "dpf": 0.8, "age": 45
}
→ {"risk_percent": 78.5, "risk_category": "High", "prediction": 1, "model_version": "v1"}

# Export history
GET /api/export/csv   → Downloads full prediction history as CSV
GET /api/export/json  → Downloads full prediction history as JSON

# Model versions
GET /api/models
→ {"models": [{"version": "v1", "accuracy": 98.7, "is_latest": true}]}
```

Full interactive docs at: **[/api/docs](https://diabetes-prediction-ml-2-dt3u.onrender.com/api/docs)**

---

## 🧬 How RAG Works

```
User: "Can I eat rice?"
         ↓
DiabetesAI searches WHO + ADA medical guidelines
         ↓
Finds: ADA Nutrition Therapy 2024 → "Patients with diabetes
       should choose whole grain rice, limit portion to 1/3 cup"
         ↓
Passes context to Claude AI
         ↓
Claude: "According to ADA guidelines, you can eat rice in
         moderation — choose brown rice and keep portions small.
         White rice has a glycaemic index of 73 which can spike
         blood sugar quickly. Always consult your doctor! 🩺"
```

**This is Retrieval Augmented Generation (RAG)** — one of the hottest techniques in production AI. The chatbot doesn't just use general knowledge — it retrieves from verified medical sources first.

---

## 🏃 Run Locally

```bash
# Clone
git clone https://github.com/Bhavgulati/diabetes-prediction-ml.git
cd "diabetes-prediction-ml/Diabetes-prediction deployed"

# Install
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add: ANTHROPIC_API_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SECRET_KEY

# Run
python app.py
# → http://127.0.0.1:5001
```

### 🐳 Docker
```bash
docker-compose up --build
# → http://localhost:5001
```

### 🧪 Tests
```bash
pytest test_app.py -v
# → 8 tests passing ✅
```

---

## 📊 Model Performance on Expanded Dataset

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 🏆 **Random Forest (Tuned, threshold=0.6)** | **93.8%** | **60.4%** | **84.6%** | **70.4%** |
| Logistic Regression (cleaned, threshold=0.6) | 89.3% | 44.0% | 82.8% | 57.5% |
| Random Forest + SMOTE | 92.4% | 54.0% | 85.2% | 66.2% |

> Tuned Random Forest delivered the best overall screening performance on the expanded 100,768-record dataset, achieving the strongest balance between recall, precision, and F1-score after threshold calibration.
## 🗂️ Project Structure

```
diabetes-prediction-ml/
├── Diabetes-prediction deployed/
│   ├── app.py                    # Main Flask application
│   ├── rag/
│   │   ├── rag_engine.py         # Lightweight RAG search engine
│   │   ├── medical_knowledge.py  # WHO + ADA medical documents
│   │   └── setup_rag.py          # One-time setup script
│   ├── templates/
│   │   ├── index.html            # Assessment form (real-time risk)
│   │   ├── result.html           # Results + chatbot + health score
│   │   ├── dashboard.html        # User history + trend graph
│   │   ├── login.html            # Google OAuth
│   │   └── api_docs.html         # REST API documentation
│   ├── diabetes-prediction-rfc-model.pkl  # Trained Random Forest
│   ├── test_app.py               # 8 pytest unit tests
│   ├── Dockerfile                # Docker configuration
│   ├── docker-compose.yml        # Docker Compose
│   ├── gunicorn.conf.py          # Production server config
│   └── requirements.txt
└── diabetes.csv                  # Pima Indian Diabetes Dataset
```

---

## 🌟 What Makes This Different

Most ML projects are Jupyter notebooks with a `model.predict()` call.

**DiabetesAI is a production system:**

- 🔐 **Authentication** — Google OAuth, not just a public form
- 🛡️ **Security** — Rate limiting prevents API abuse
- 📝 **Observability** — Every prediction logged with response time
- 🔄 **MLOps** — Model versioning with instant rollback
- 🌍 **Accessibility** — Voice input + multi-language for ALL users
- 🏥 **Medical Grade** — Clinical PDF reports with WHO/ADA citations
- 📡 **API-First** — Any developer can integrate via REST API

---

## 👨‍💻 Built By

<div align="center">

**Bhavishya Gulati**

[![GitHub](https://img.shields.io/badge/GitHub-Bhavgulati-181717?style=for-the-badge&logo=github)](https://github.com/Bhavgulati)

*Built with ❤️, lots of ☕, and Claude AI 🤖*

</div>

---

## ⚠️ Medical Disclaimer

This application is for **educational and informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional before making any health decisions.

---

<div align="center">

**⭐ If this project impressed you, give it a star! ⭐**

[![Star](https://img.shields.io/github/stars/Bhavgulati/diabetes-prediction-ml?style=for-the-badge&color=f59e0b)](https://github.com/Bhavgulati/diabetes-prediction-ml)

*Made in India 🇮🇳 · Deployed Worldwide 🌍*

</div>