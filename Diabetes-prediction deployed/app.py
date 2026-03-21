from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
import pickle
import numpy as np
import pandas as pd
import requests as http_requests
import os
import shap
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# ── App setup ──
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'local-dev-key')

# ── Database (SQLite) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'diabetesai.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Login Manager ──
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── Google OAuth ──
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# ── API Key ──
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# ── Database Models ──
class User(UserMixin, db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    google_id    = db.Column(db.String(100), unique=True, nullable=False)
    name         = db.Column(db.String(100))
    email        = db.Column(db.String(100), unique=True)
    picture      = db.Column(db.String(300))
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    predictions  = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    risk_percent  = db.Column(db.Float)
    risk_category = db.Column(db.String(20))
    prediction    = db.Column(db.Integer)
    pregnancies   = db.Column(db.Float)
    glucose       = db.Column(db.Float)
    bloodpressure = db.Column(db.Float)
    skinthickness = db.Column(db.Float)
    insulin       = db.Column(db.Float)
    bmi           = db.Column(db.Float)
    dpf           = db.Column(db.Float)
    age           = db.Column(db.Float)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Load ML models ──
rf_model = pickle.load(open(os.path.join(BASE_DIR, 'diabetes-prediction-rfc-model.pkl'), 'rb'))

df = pd.read_csv(os.path.join(BASE_DIR, '..', 'diabetes.csv'))
X = df.drop('Outcome', axis=1)
y = df['Outcome']
FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
lr_model = LogisticRegression(max_iter=1000, solver='saga')
lr_model.fit(X_train_scaled, y_train)

# XGBoost optional
try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_available = True
except Exception:
    xgb_available = False

# SHAP
explainer = shap.TreeExplainer(rf_model)

# ── Model stats ──
def get_model_stats(model, name, X_test_data=None):
    if X_test_data is None:
        X_test_data = X_test
    preds = model.predict(X_test_data)
    return {
        'name':      name,
        'accuracy':  round(accuracy_score(y_test, preds) * 100, 1),
        'precision': round(precision_score(y_test, preds) * 100, 1),
        'recall':    round(recall_score(y_test, preds) * 100, 1),
        'f1':        round(f1_score(y_test, preds) * 100, 1),
    }

model_comparison = [
    get_model_stats(rf_model, 'Random Forest'),
    get_model_stats(lr_model, 'Logistic Regression', X_test_scaled),
]
if xgb_available:
    model_comparison.append(get_model_stats(xgb_model, 'XGBoost'))


# ════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════

# ── Login page ──
@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('login.html')

# ── Google OAuth ──
@app.route('/google-login')
def google_login():
    redirect_uri = url_for('callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/callback')
def callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')

    # Find or create user
    user = User.query.filter_by(google_id=user_info['sub']).first()
    if not user:
        user = User(
            google_id=user_info['sub'],
            name=user_info.get('name'),
            email=user_info.get('email'),
            picture=user_info.get('picture')
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    return redirect(url_for('home'))

# ── Logout ──
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ── Home (protected) ──
@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

# ── Dashboard ──
@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.asc()).all()
    history = [{
        'date':     p.created_at.strftime('%d %b %Y'),
        'risk':     p.risk_percent,
        'category': p.risk_category,
        'glucose':  p.glucose,
        'bmi':      p.bmi,
        'age':      p.age,
    } for p in predictions]
    return render_template('dashboard.html', user=current_user, history=json.dumps(history), total=len(predictions))

# ── Predict ──
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        preg    = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp      = int(request.form['bloodpressure'])
        st      = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi     = float(request.form['bmi'])
        dpf     = float(request.form['dpf'])
        age     = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

        # Risk scoring
        proba        = rf_model.predict_proba(data)[0]
        risk_percent = round(proba[1] * 100, 1)

        if risk_percent >= 70:
            risk_category = 'High'
            risk_color    = '#ff4d6d'
        elif risk_percent >= 40:
            risk_category = 'Moderate'
            risk_color    = '#f59e0b'
        else:
            risk_category = 'Low'
            risk_color    = '#10b981'

        prediction = 1 if risk_percent >= 50 else 0

        # Feature importance
        importances       = rf_model.feature_importances_
        feature_importance = sorted([
            {'feature': FEATURE_NAMES[i], 'importance': round(float(importances[i]) * 100, 1)}
            for i in range(len(FEATURE_NAMES))
        ], key=lambda x: x['importance'], reverse=True)

        # SHAP
        shap_values = explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1][0]).flatten()
        elif hasattr(shap_values, 'values'):
            shap_vals = np.array(shap_values.values[0]).flatten()
        else:
            shap_vals = np.array(shap_values[0]).flatten()

        shap_data = sorted([{
            'feature':   FEATURE_NAMES[i],
            'value':     round(float(data[0][i]), 2),
            'shap':      round(float(shap_vals[i]) * 100, 1),
            'direction': 'increases' if float(shap_vals[i]) > 0 else 'reduces',
            'color':     '#ff4d6d' if float(shap_vals[i]) > 0 else '#10b981'
        } for i in range(len(FEATURE_NAMES))], key=lambda x: abs(x['shap']), reverse=True)

        # Save to database
        pred_record = Prediction(
            user_id=current_user.id,
            risk_percent=risk_percent,
            risk_category=risk_category,
            prediction=prediction,
            pregnancies=preg,
            glucose=glucose,
            bloodpressure=bp,
            skinthickness=st,
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            age=age
        )
        db.session.add(pred_record)
        db.session.commit()

        return render_template(
            'result.html',
            user=current_user,
            prediction=prediction,
            risk_percent=risk_percent,
            risk_category=risk_category,
            risk_color=risk_color,
            feature_importance=json.dumps(feature_importance),
            shap_data=shap_data,
            model_comparison=json.dumps(model_comparison),
        )

# ── Chat ──
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message  = http_requests.json().get('message') if False else request.json.get('message')
    prediction    = int(request.json.get('prediction', 0))
    risk_percent  = request.json.get('risk_percent', 50)
    risk_category = request.json.get('risk_category', 'Moderate')

    system_prompt = f"""You are a friendly, conversational medical assistant in a diabetes prediction app.
The patient's result is: {'DIABETIC' if prediction == 1 else 'NOT DIABETIC'}.
Their diabetes risk score is: {risk_percent}% ({risk_category} risk).

YOUR PERSONALITY:
- Talk like a caring friend, not a textbook
- Be warm, encouraging and practical
- Keep responses SHORT (3-5 lines max)
- Never lecture unless asked

HOW TO RESPOND:
- If the user shares a struggle (e.g. "I can't resist ice cream"), EMPATHIZE FIRST then give 1 practical tip
- If the user asks a question, answer it directly and simply
- Never dump a wall of text or list 10 things at once
- Never say "diabetes is caused by..." unless they specifically ask

Always end with a reminder to consult their doctor for medical decisions."""

    response = http_requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        },
        json={
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 300,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_message}]
        }
    )

    print("API Response:", response.json())
    reply = response.json()['content'][0]['text']
    return jsonify({'reply': reply})


# ── Init DB and run ──

import json as _json

@app.template_filter('from_json')
def from_json_filter(value):
    return _json.loads(value)
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)