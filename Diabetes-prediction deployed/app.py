from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
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
import io
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# ── PDF ──
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Email ──
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# ── App setup ──
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'local-dev-key')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'diabetesai.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
SENDGRID_API_KEY  = os.environ.get('SENDGRID_API_KEY', '')
GOOGLE_MAPS_KEY   = os.environ.get('GOOGLE_MAPS_API_KEY', '')

class User(UserMixin, db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    google_id   = db.Column(db.String(100), unique=True, nullable=False)
    name        = db.Column(db.String(100))
    email       = db.Column(db.String(100), unique=True)
    picture     = db.Column(db.String(300))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

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

rf_model = pickle.load(open(os.path.join(BASE_DIR, 'diabetes-prediction-rfc-model.pkl'), 'rb'))
df = pd.read_csv(os.path.join(BASE_DIR, '..', 'diabetes.csv'))
X = df.drop('Outcome', axis=1)
y = df['Outcome']
FEATURE_NAMES = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
lr_model = LogisticRegression(max_iter=1000, solver='saga')
lr_model.fit(X_train_scaled, y_train)

try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_available = True
except Exception:
    xgb_available = False

explainer = shap.TreeExplainer(rf_model)

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
#  INPUT VALIDATION
# ════════════════════════════════════════
def validate_inputs(data):
    errors = []
    rules = {
        'pregnancies':   (0,   20,   'Pregnancies must be between 0 and 20'),
        'glucose':       (1,   600,  'Glucose must be between 1 and 600 mg/dL'),
        'bloodpressure': (1,   200,  'Blood pressure must be between 1 and 200 mmHg'),
        'skinthickness': (0,   100,  'Skin thickness must be between 0 and 100 mm'),
        'insulin':       (0,   900,  'Insulin must be between 0 and 900 IU/mL'),
        'bmi':           (1,   80,   'BMI must be between 1 and 80 kg/m²'),
        'dpf':           (0,   3,    'Diabetes pedigree must be between 0 and 3'),
        'age':           (1,   120,  'Age must be between 1 and 120 years'),
    }
    for field, (min_val, max_val, msg) in rules.items():
        val = data.get(field)
        if val is None:
            errors.append(f'{field} is required')
        else:
            try:
                if not (min_val <= float(val) <= max_val):
                    errors.append(msg)
            except (ValueError, TypeError):
                errors.append(f'{field} must be a valid number')
    return errors


# ════════════════════════════════════════
#  CLINICAL PDF REPORT
# ════════════════════════════════════════
def generate_pdf_report(user, risk_percent, risk_category, risk_color, shap_data, inputs):
    buffer = io.BytesIO()

    # Colours
    NAVY       = HexColor('#0a2342')
    DARK_NAVY  = HexColor('#061529')
    MID_BLUE   = HexColor('#1a4a7a')
    LIGHT_BLUE = HexColor('#e8f4fd')
    RED        = HexColor('#c0392b')
    AMBER      = HexColor('#d68910')
    GREEN      = HexColor('#1e8449')
    LIGHT_GRAY = HexColor('#f2f3f4')
    MID_GRAY   = HexColor('#aab7b8')
    DARK_GRAY  = HexColor('#2c3e50')
    WHITE      = white

    risk_color_map = {'High': RED, 'Moderate': AMBER, 'Low': GREEN}
    RISK_COLOR = risk_color_map.get(risk_category, AMBER)

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=0.4*inch, bottomMargin=0.6*inch,
        leftMargin=0.6*inch, rightMargin=0.6*inch
    )

    story = []
    W = A4[0] - 1.2*inch  # usable width

    # ── Styles ──
    header_title = ParagraphStyle('ht', fontSize=11, fontName='Helvetica-Bold',
                                   textColor=WHITE, leading=14)
    header_sub   = ParagraphStyle('hs', fontSize=8, fontName='Helvetica',
                                   textColor=HexColor('#a8c8e8'), leading=11)
    section_head = ParagraphStyle('sh', fontSize=10, fontName='Helvetica-Bold',
                                   textColor=WHITE, leading=13)
    body_sm      = ParagraphStyle('bs', fontSize=8.5, fontName='Helvetica',
                                   textColor=DARK_GRAY, leading=13)
    body_bold    = ParagraphStyle('bb', fontSize=8.5, fontName='Helvetica-Bold',
                                   textColor=DARK_GRAY, leading=13)
    disclaimer_s = ParagraphStyle('ds', fontSize=7.5, fontName='Helvetica',
                                   textColor=HexColor('#7f8c8d'), leading=11,
                                   alignment=TA_CENTER)
    label_s      = ParagraphStyle('ls', fontSize=7.5, fontName='Helvetica-Bold',
                                   textColor=MID_GRAY, leading=10,
                                   spaceAfter=1)
    value_s      = ParagraphStyle('vs', fontSize=9, fontName='Helvetica-Bold',
                                   textColor=DARK_GRAY, leading=11)

    report_id = f"DAI-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    now_str   = datetime.now().strftime('%d %B %Y, %I:%M %p')

    # ════════════════════════
    # HEADER BANNER
    # ════════════════════════
    header_data = [[
        Paragraph('<b>DiabetesAI</b><br/>Medical Intelligence Platform', header_title),
        Paragraph(
            f'<b>DIABETES RISK ASSESSMENT REPORT</b><br/>'
            f'Report ID: {report_id}<br/>'
            f'Generated: {now_str}',
            ParagraphStyle('hr', fontSize=8, fontName='Helvetica',
                           textColor=WHITE, leading=12, alignment=TA_RIGHT)
        )
    ]]
    header_table = Table(header_data, colWidths=[W*0.5, W*0.5])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), NAVY),
        ('TOPPADDING',   (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0), (-1,-1), 14),
        ('LEFTPADDING',  (0,0), (0,-1),  14),
        ('RIGHTPADDING', (-1,0),(-1,-1), 14),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(header_table)

    # ── Red/Amber/Green risk stripe ──
    stripe_color = RISK_COLOR
    stripe = Table([['', '', '']], colWidths=[W*0.33, W*0.34, W*0.33])
    stripe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), stripe_color),
        ('TOPPADDING', (0,0),(-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
    ]))
    story.append(stripe)
    story.append(Spacer(1, 10))

    # ════════════════════════
    # PATIENT INFO + RISK SCORE (side by side)
    # ════════════════════════
    patient_data = [
        [Paragraph('PATIENT INFORMATION', label_s), Paragraph('AI RISK ASSESSMENT', label_s)],
        [
            Table([
                [Paragraph('Full Name', label_s),      Paragraph(user.name, value_s)],
                [Paragraph('Email', label_s),           Paragraph(user.email or '—', value_s)],
                [Paragraph('Report Date', label_s),     Paragraph(datetime.now().strftime('%d %B %Y'), value_s)],
                [Paragraph('Assessment ID', label_s),   Paragraph(report_id, value_s)],
                [Paragraph('Powered By', label_s),      Paragraph('Random Forest ML Model', value_s)],
            ], colWidths=[W*0.18, W*0.28],
               style=[('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                      ('LINEBELOW',(0,0),(-1,-2),0.3,HexColor('#dce1e7'))]),

            Table([
                [Paragraph(f'{risk_percent}%',
                    ParagraphStyle('rp', fontSize=42, fontName='Helvetica-Bold',
                                   textColor=RISK_COLOR, leading=46, alignment=TA_CENTER))],
                [Paragraph(f'{risk_category.upper()} RISK',
                    ParagraphStyle('rc', fontSize=13, fontName='Helvetica-Bold',
                                   textColor=RISK_COLOR, leading=16, alignment=TA_CENTER))],
                [Paragraph(
                    'See a Doctor Immediately' if risk_category == 'High' else
                    'Monitor Health Closely' if risk_category == 'Moderate' else
                    'Maintain Healthy Habits',
                    ParagraphStyle('rr', fontSize=8, fontName='Helvetica',
                                   textColor=DARK_GRAY, leading=11, alignment=TA_CENTER))],
            ], colWidths=[W*0.46],
               style=[('ALIGN',(0,0),(-1,-1),'CENTER'),
                      ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                      ('BOX',(0,0),(-1,-1),1.5,RISK_COLOR),
                      ('BACKGROUND',(0,0),(-1,-1),HexColor('#fdfefe')),
                      ('ROUNDEDCORNERS',[6])])
        ]
    ]
    outer = Table(patient_data, colWidths=[W*0.46+W*0.18, W*0.46])
    outer.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 6),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ('LEFTPADDING',  (0,0),(-1,-1), 8),
        ('RIGHTPADDING', (0,0),(-1,-1), 8),
        ('VALIGN',       (0,0),(-1,-1), 'TOP'),
        ('BACKGROUND',   (0,0),(-1,-1), LIGHT_GRAY),
        ('BOX',          (0,0),(-1,-1), 0.5, HexColor('#dce1e7')),
    ]))
    story.append(outer)
    story.append(Spacer(1, 10))

    # ════════════════════════
    # SECTION HEADER helper
    # ════════════════════════
    def section_header(title, icon=''):
        t = Table([[Paragraph(f'{icon}  {title}', section_head)]],
                  colWidths=[W])
        t.setStyle(TableStyle([
            ('BACKGROUND',   (0,0),(-1,-1), MID_BLUE),
            ('TOPPADDING',   (0,0),(-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
            ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ]))
        return t

    # ════════════════════════
    # BIOMARKER RESULTS TABLE
    # ════════════════════════
    story.append(section_header('BIOMARKER RESULTS', '🔬'))
    story.append(Spacer(1, 4))

    def status_label(metric, value):
        thresholds = {
            'Glucose':        [(70,99,'NORMAL'),(100,125,'PRE-DIABETIC'),(126,600,'HIGH')],
            'BMI':            [(18.5,24.9,'NORMAL'),(25,29.9,'OVERWEIGHT'),(30,80,'OBESE')],
            'Blood Pressure': [(60,80,'NORMAL'),(81,89,'ELEVATED'),(90,200,'HIGH')],
            'Insulin':        [(16,166,'NORMAL'),(0,15,'LOW'),(167,900,'HIGH')],
        }
        ranges = thresholds.get(metric)
        if not ranges:
            return ''
        try:
            v = float(str(value).split()[0])
            for lo, hi, label in ranges:
                if lo <= v <= hi:
                    colors_map = {
                        'NORMAL': GREEN, 'PRE-DIABETIC': AMBER,
                        'HIGH': RED, 'ELEVATED': AMBER,
                        'OVERWEIGHT': AMBER, 'OBESE': RED,
                        'LOW': AMBER
                    }
                    c = colors_map.get(label, MID_GRAY)
                    return label, c
        except:
            pass
        return '', MID_GRAY

    bio_headers = [
        Paragraph('<b>Biomarker</b>', ParagraphStyle('bh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE)),
        Paragraph('<b>Your Value</b>', ParagraphStyle('bh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
        Paragraph('<b>Reference Range</b>', ParagraphStyle('bh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
        Paragraph('<b>Status</b>', ParagraphStyle('bh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
    ]

    bio_rows = [
        ['Fasting Glucose',      f"{inputs.get('glucose','—')} mg/dL",    '70 – 99 mg/dL',   'Glucose'],
        ['Body Mass Index (BMI)',f"{inputs.get('bmi','—')} kg/m²",         '18.5 – 24.9',     'BMI'],
        ['Diastolic Blood Pres.',f"{inputs.get('bloodpressure','—')} mmHg",'60 – 80 mmHg',    'Blood Pressure'],
        ['Serum Insulin',        f"{inputs.get('insulin','—')} IU/mL",     '16 – 166 IU/mL',  'Insulin'],
        ['Tricep Skin Thickness',f"{inputs.get('skinthickness','—')} mm",  '10 – 30 mm',      ''],
        ['Pregnancies',          str(inputs.get('pregnancies','—')),        '0 – 17',          ''],
        ['Diabetes Pedigree Fn.',str(inputs.get('dpf','—')),               '0.078 – 2.42',    ''],
        ['Age',                  f"{inputs.get('age','—')} years",          '—',               ''],
    ]

    table_data = [bio_headers]
    status_styles = []

    for i, (metric, val, ref, key) in enumerate(bio_rows):
        row_num = i + 1
        s_result = status_label(key, val) if key else ('', MID_GRAY)
        if s_result and s_result[0]:
            s_text, s_color = s_result
            status_cell = Paragraph(
                f'<b>{s_text}</b>',
                ParagraphStyle('st', fontSize=7.5, fontName='Helvetica-Bold',
                               textColor=s_color, alignment=TA_CENTER)
            )
        else:
            status_cell = Paragraph('—', ParagraphStyle('st', fontSize=8, fontName='Helvetica',
                                    textColor=MID_GRAY, alignment=TA_CENTER))

        table_data.append([
            Paragraph(metric, body_sm),
            Paragraph(f'<b>{val}</b>', ParagraphStyle('bv', fontSize=9, fontName='Helvetica-Bold',
                                                       textColor=DARK_GRAY, alignment=TA_CENTER)),
            Paragraph(ref, ParagraphStyle('br', fontSize=8.5, fontName='Helvetica',
                                          textColor=HexColor('#7f8c8d'), alignment=TA_CENTER)),
            status_cell
        ])

    bio_table = Table(table_data, colWidths=[W*0.32, W*0.22, W*0.26, W*0.20])
    bio_styles = [
        ('BACKGROUND',    (0,0), (-1,0), NAVY),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('GRID',          (0,0), (-1,-1), 0.4, HexColor('#dce1e7')),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW',     (0,0), (-1,0), 2, MID_BLUE),
    ]
    bio_table.setStyle(TableStyle(bio_styles))
    story.append(bio_table)
    story.append(Spacer(1, 10))

    # ════════════════════════
    # SHAP ANALYSIS
    # ════════════════════════
    story.append(section_header('AI RISK FACTOR ANALYSIS (SHAP)', '📊'))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        'The following analysis shows how each biomarker contributed to your personalised risk prediction:',
        ParagraphStyle('note', fontSize=8, fontName='Helvetica-Oblique',
                       textColor=HexColor('#7f8c8d'), leading=11, spaceAfter=6)
    ))

    shap_headers = [
        Paragraph('<b>Risk Factor</b>',   ParagraphStyle('sh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE)),
        Paragraph('<b>Your Value</b>',    ParagraphStyle('sh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
        Paragraph('<b>Contribution</b>',  ParagraphStyle('sh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
        Paragraph('<b>Effect</b>',        ParagraphStyle('sh', fontSize=8.5, fontName='Helvetica-Bold', textColor=WHITE, alignment=TA_CENTER)),
    ]
    shap_rows = [shap_headers]
    for item in shap_data[:6]:
        effect_color = RED if item['direction'] == 'increases' else GREEN
        effect_text  = '▲ INCREASES RISK' if item['direction'] == 'increases' else '▼ REDUCES RISK'
        shap_rows.append([
            Paragraph(item['feature'], body_sm),
            Paragraph(str(item['value']),
                      ParagraphStyle('sv', fontSize=9, fontName='Helvetica-Bold',
                                     textColor=DARK_GRAY, alignment=TA_CENTER)),
            Paragraph(f"{abs(item['shap'])}%",
                      ParagraphStyle('sc', fontSize=9, fontName='Helvetica-Bold',
                                     textColor=effect_color, alignment=TA_CENTER)),
            Paragraph(effect_text,
                      ParagraphStyle('se', fontSize=7.5, fontName='Helvetica-Bold',
                                     textColor=effect_color, alignment=TA_CENTER)),
        ])

    shap_table = Table(shap_rows, colWidths=[W*0.30, W*0.20, W*0.22, W*0.28])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), NAVY),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('GRID',          (0,0), (-1,-1), 0.4, HexColor('#dce1e7')),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW',     (0,0), (-1,0), 2, MID_BLUE),
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 10))

    # ════════════════════════
    # CLINICAL RECOMMENDATIONS
    # ════════════════════════
    story.append(section_header('CLINICAL RECOMMENDATIONS', '📋'))
    story.append(Spacer(1, 4))

    if risk_category == 'High':
        urgency = Table([[Paragraph(
            '⚠️  URGENT: This report indicates HIGH diabetes risk. Immediate medical consultation is strongly advised.',
            ParagraphStyle('urg', fontSize=9, fontName='Helvetica-Bold',
                           textColor=WHITE, leading=13)
        )]], colWidths=[W])
        urgency.setStyle(TableStyle([
            ('BACKGROUND',   (0,0),(-1,-1), RED),
            ('TOPPADDING',   (0,0),(-1,-1), 10),
            ('BOTTOMPADDING',(0,0),(-1,-1), 10),
            ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ]))
        story.append(urgency)
        story.append(Spacer(1, 6))
        recs = [
            ('IMMEDIATE', 'Consult an endocrinologist or diabetologist within 1–2 weeks.'),
            ('IMMEDIATE', 'Request HbA1c, fasting glucose, and oral glucose tolerance test.'),
            ('DIETARY',   'Eliminate sugar, refined carbohydrates, white rice, and processed foods.'),
            ('DIETARY',   'Adopt a low-glycaemic index diet under dietitian supervision.'),
            ('LIFESTYLE', 'Begin 30 minutes of moderate exercise daily — walking, swimming, or cycling.'),
            ('LIFESTYLE', 'Monitor fasting blood glucose daily and maintain a log.'),
            ('MEDICAL',   'Discuss Metformin or other medications with your doctor if confirmed diabetic.'),
            ('MEDICAL',   'Schedule HbA1c re-evaluation every 3 months.'),
        ]
    elif risk_category == 'Moderate':
        urgency = Table([[Paragraph(
            '⚡  ADVISORY: Elevated diabetes risk detected. Medical review recommended within 4–6 weeks.',
            ParagraphStyle('urg', fontSize=9, fontName='Helvetica-Bold',
                           textColor=WHITE, leading=13)
        )]], colWidths=[W])
        urgency.setStyle(TableStyle([
            ('BACKGROUND',   (0,0),(-1,-1), AMBER),
            ('TOPPADDING',   (0,0),(-1,-1), 10),
            ('BOTTOMPADDING',(0,0),(-1,-1), 10),
            ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ]))
        story.append(urgency)
        story.append(Spacer(1, 6))
        recs = [
            ('MEDICAL',   'Schedule a diabetes screening appointment within 4–6 weeks.'),
            ('MEDICAL',   'Request fasting blood glucose and HbA1c tests.'),
            ('DIETARY',   'Reduce sugar, white bread, and processed food intake significantly.'),
            ('DIETARY',   'Increase fibre, vegetables, lean proteins, and whole grains.'),
            ('LIFESTYLE', 'Target 150 minutes of moderate exercise per week.'),
            ('LIFESTYLE', 'Achieve 5–7% body weight reduction if overweight.'),
            ('MONITORING','Check fasting blood glucose every 3–6 months.'),
            ('MONITORING','Track BMI, blood pressure, and glucose levels regularly.'),
        ]
    else:
        urgency = Table([[Paragraph(
            '✅  LOW RISK: No immediate concerns. Maintain healthy habits and schedule annual screening.',
            ParagraphStyle('urg', fontSize=9, fontName='Helvetica-Bold',
                           textColor=WHITE, leading=13)
        )]], colWidths=[W])
        urgency.setStyle(TableStyle([
            ('BACKGROUND',   (0,0),(-1,-1), GREEN),
            ('TOPPADDING',   (0,0),(-1,-1), 10),
            ('BOTTOMPADDING',(0,0),(-1,-1), 10),
            ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ]))
        story.append(urgency)
        story.append(Spacer(1, 6))
        recs = [
            ('DIETARY',   'Maintain balanced diet: vegetables, whole grains, lean proteins, healthy fats.'),
            ('DIETARY',   'Limit sugar, refined carbohydrates, and processed food intake.'),
            ('LIFESTYLE', 'Exercise at least 30 minutes daily — any moderate activity counts.'),
            ('LIFESTYLE', 'Maintain healthy BMI (18.5–24.9) and avoid sedentary behaviour.'),
            ('MONITORING','Schedule annual fasting blood glucose and HbA1c screening.'),
            ('MONITORING','Monitor blood pressure and maintain below 120/80 mmHg.'),
            ('PREVENTIVE','Avoid smoking and limit alcohol to reduce long-term diabetes risk.'),
            ('PREVENTIVE','Manage stress — chronic stress raises cortisol and blood glucose levels.'),
        ]

    # Category colours
    cat_colors = {
        'IMMEDIATE': RED, 'MEDICAL': MID_BLUE, 'DIETARY': GREEN,
        'LIFESTYLE': HexColor('#1a5276'), 'MONITORING': HexColor('#6c3483'),
        'PREVENTIVE': HexColor('#1a5276')
    }

    rec_rows = []
    for cat, text in recs:
        c = cat_colors.get(cat, NAVY)
        rec_rows.append([
            Paragraph(f'<b>{cat}</b>',
                      ParagraphStyle('rc', fontSize=7, fontName='Helvetica-Bold',
                                     textColor=WHITE, alignment=TA_CENTER)),
            Paragraph(text, body_sm)
        ])

    rec_table = Table(rec_rows, colWidths=[W*0.14, W*0.86])
    rec_styles = [('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
                  ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8),
                  ('GRID',(0,0),(-1,-1),0.4,HexColor('#dce1e7')),
                  ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                  ('ROWBACKGROUNDS',(0,0),(-1,-1),[WHITE, LIGHT_GRAY])]
    for i, (cat, _) in enumerate(recs):
        c = cat_colors.get(cat, NAVY)
        rec_styles.append(('BACKGROUND', (0,i), (0,i), c))
    rec_table.setStyle(TableStyle(rec_styles))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # ════════════════════════
    # FOOTER DISCLAIMER
    # ════════════════════════
    footer = Table([[Paragraph(
        f'<b>MEDICAL DISCLAIMER:</b> This report is generated by DiabetesAI, an artificial intelligence system, '
        f'for informational and screening purposes only. It does NOT constitute a medical diagnosis, '
        f'clinical opinion, or professional medical advice. The risk score is based on statistical modelling '
        f'of the Pima Indian Diabetes Dataset and may not reflect all individual health factors. '
        f'Always consult a qualified, registered medical practitioner before making any health decisions. '
        f'Report ID: {report_id} | Generated: {now_str} | DiabetesAI v1.0',
        disclaimer_s
    )]], colWidths=[W])
    footer.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), LIGHT_GRAY),
        ('TOPPADDING',   (0,0),(-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1), 10),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('BOX',          (0,0),(-1,-1), 0.5, HexColor('#dce1e7')),
    ]))
    story.append(footer)

    doc.build(story)
    buffer.seek(0)
    return buffer


# ════════════════════════════════════════
#  EMAIL
# ════════════════════════════════════════
def send_email_report(user_email, user_name, risk_percent, risk_category):
    if not SENDGRID_AVAILABLE or not SENDGRID_API_KEY:
        return False
    try:
        color = {'High': '#c0392b', 'Moderate': '#d68910', 'Low': '#1e8449'}.get(risk_category, '#1a4a7a')
        rec   = ('Please consult a doctor immediately.' if risk_category == 'High' else
                 'Monitor your health closely.' if risk_category == 'Moderate' else
                 'Keep up your healthy habits!')
        html  = f"""
        <div style="font-family:sans-serif;max-width:600px;margin:0 auto;background:#0a2342;color:#f1f5f9;padding:40px;border-radius:8px">
          <h1 style="color:#a8c8e8;font-size:22px;margin-bottom:4px">DiabetesAI</h1>
          <p style="color:#5d8aa8;font-size:12px;margin-bottom:24px">Medical Intelligence Platform</p>
          <h2 style="font-size:18px">Your Health Report is Ready, {user_name}</h2>
          <div style="background:#061529;border:2px solid {color};border-radius:8px;padding:24px;margin:24px 0;text-align:center">
            <div style="font-size:52px;font-weight:800;color:{color}">{risk_percent}%</div>
            <div style="font-size:16px;font-weight:700;color:{color};margin-top:6px;letter-spacing:2px">{risk_category.upper()} RISK</div>
          </div>
          <p style="color:#aab7b8;font-size:14px">{rec}</p>
          <p style="color:#7f8c8d;font-size:11px;margin-top:32px;border-top:1px solid #1a4a7a;padding-top:16px">
            DISCLAIMER: This is not a medical diagnosis. Always consult a qualified healthcare professional.
          </p>
        </div>"""
        message = Mail(
            from_email='noreply@diabetesai.com',
            to_emails=user_email,
            subject=f'DiabetesAI Report — {risk_category} Risk ({risk_percent}%)',
            html_content=html
        )
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        sg.client.mail.send.post(request_body=message.get())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


# ════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════

@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/google-login')
def google_login():
    redirect_uri = url_for('callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/callback')
def callback():
    token     = google.authorize_access_token()
    user_info = token.get('userinfo')
    user      = User.query.filter_by(google_id=user_info['sub']).first()
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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

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
    return render_template('dashboard.html', user=current_user,
                           history=json.dumps(history), total=len(predictions))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        raw = {
            'pregnancies':   request.form.get('pregnancies'),
            'glucose':       request.form.get('glucose'),
            'bloodpressure': request.form.get('bloodpressure'),
            'skinthickness': request.form.get('skinthickness'),
            'insulin':       request.form.get('insulin'),
            'bmi':           request.form.get('bmi'),
            'dpf':           request.form.get('dpf'),
            'age':           request.form.get('age'),
        }
        errors = validate_inputs(raw)
        if errors:
            return render_template('index.html', user=current_user, errors=errors)

        preg    = int(raw['pregnancies'])
        glucose = int(raw['glucose'])
        bp      = int(raw['bloodpressure'])
        st      = int(raw['skinthickness'])
        insulin = int(raw['insulin'])
        bmi     = float(raw['bmi'])
        dpf     = float(raw['dpf'])
        age     = int(raw['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        proba        = rf_model.predict_proba(data)[0]
        risk_percent = round(proba[1] * 100, 1)

        if risk_percent >= 70:
            risk_category, risk_color = 'High',     '#ff4d6d'
        elif risk_percent >= 40:
            risk_category, risk_color = 'Moderate', '#f59e0b'
        else:
            risk_category, risk_color = 'Low',      '#10b981'

        prediction = 1 if risk_percent >= 50 else 0

        importances       = rf_model.feature_importances_
        feature_importance = sorted([
            {'feature': FEATURE_NAMES[i], 'importance': round(float(importances[i]) * 100, 1)}
            for i in range(len(FEATURE_NAMES))
        ], key=lambda x: x['importance'], reverse=True)

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

        pred_record = Prediction(
            user_id=current_user.id,
            risk_percent=risk_percent, risk_category=risk_category,
            prediction=prediction, pregnancies=preg, glucose=glucose,
            bloodpressure=bp, skinthickness=st, insulin=insulin,
            bmi=bmi, dpf=dpf, age=age
        )
        db.session.add(pred_record)
        db.session.commit()

        try:
            send_email_report(current_user.email, current_user.name, risk_percent, risk_category)
        except Exception:
            pass

        inputs = {'pregnancies': preg, 'glucose': glucose, 'bloodpressure': bp,
                  'skinthickness': st, 'insulin': insulin, 'bmi': bmi, 'dpf': dpf, 'age': age}

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
            inputs=json.dumps(inputs),
            google_maps_key=GOOGLE_MAPS_KEY,
        )


@app.route('/download-report', methods=['POST'])
@login_required
def download_report():
    if not PDF_AVAILABLE:
        return jsonify({'error': 'PDF not available'}), 500
    data   = request.json
    buffer = generate_pdf_report(
        user=current_user,
        risk_percent=data.get('risk_percent'),
        risk_category=data.get('risk_category'),
        risk_color=data.get('risk_color', '#00d4ff'),
        shap_data=data.get('shap_data', []),
        inputs=data.get('inputs', {})
    )
    return send_file(
        buffer, as_attachment=True,
        download_name=f'DiabetesAI_Clinical_Report_{current_user.name.replace(" ","_")}_{datetime.now().strftime("%Y%m%d")}.pdf',
        mimetype='application/pdf'
    )


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message  = request.json.get('message')
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
- If the user shares a struggle, EMPATHIZE FIRST then give 1 practical tip
- If the user asks a question, answer it directly and simply
- Never dump a wall of text or list 10 things at once

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
    reply = response.json()['content'][0]['text']
    return jsonify({'reply': reply})


# REST API
@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status':    'healthy',
        'service':   'DiabetesAI API',
        'version':   '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    errors = validate_inputs(data)
    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 422
    try:
        arr = np.array([[
            float(data['pregnancies']), float(data['glucose']),
            float(data['bloodpressure']), float(data['skinthickness']),
            float(data['insulin']), float(data['bmi']),
            float(data['dpf']), float(data['age'])
        ]])
        proba         = rf_model.predict_proba(arr)[0]
        risk_percent  = round(proba[1] * 100, 1)
        risk_category = ('High' if risk_percent >= 70 else
                         'Moderate' if risk_percent >= 40 else 'Low')
        return jsonify({
            'risk_percent':  risk_percent,
            'risk_category': risk_category,
            'prediction':    1 if risk_percent >= 50 else 0,
            'confidence':    round(max(proba) * 100, 1),
            'message':       f'{risk_category} diabetes risk detected ({risk_percent}%)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/docs', methods=['GET'])
def api_docs():
    return render_template('api_docs.html')


import json as _json

@app.template_filter('from_json')
def from_json_filter(value):
    return _json.loads(value)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)