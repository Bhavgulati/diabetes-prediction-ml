import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, validate_inputs

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        yield client

# ── Validation Tests ──

def test_valid_inputs_pass():
    data = {'pregnancies':2,'glucose':120,'bloodpressure':70,
            'skinthickness':20,'insulin':80,'bmi':25.0,'dpf':0.5,'age':35}
    errors = validate_inputs(data)
    assert len(errors) == 0

def test_negative_glucose_fails():
    data = {'pregnancies':2,'glucose':-1,'bloodpressure':70,
            'skinthickness':20,'insulin':80,'bmi':25.0,'dpf':0.5,'age':35}
    errors = validate_inputs(data)
    assert any('Glucose' in e for e in errors)

def test_zero_bmi_fails():
    data = {'pregnancies':2,'glucose':120,'bloodpressure':70,
            'skinthickness':20,'insulin':80,'bmi':0,'dpf':0.5,'age':35}
    errors = validate_inputs(data)
    assert any('BMI' in e for e in errors)

def test_invalid_age_fails():
    data = {'pregnancies':2,'glucose':120,'bloodpressure':70,
            'skinthickness':20,'insulin':80,'bmi':25.0,'dpf':0.5,'age':200}
    errors = validate_inputs(data)
    assert any('Age' in e for e in errors)

def test_missing_field_fails():
    data = {'glucose':120,'bloodpressure':70,'skinthickness':20,
            'insulin':80,'bmi':25.0,'dpf':0.5,'age':35}
    errors = validate_inputs(data)
    assert len(errors) > 0

# ── API Tests ──

def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'DiabetesAI API'

def test_api_predict_valid(client):
    payload = {'pregnancies':2,'glucose':180,'bloodpressure':80,
               'skinthickness':35,'insulin':100,'bmi':34.0,'dpf':0.8,'age':45}
    response = client.post('/api/predict',
                           json=payload,
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'risk_percent' in data
    assert 'risk_category' in data
    assert data['risk_category'] in ['Low', 'Moderate', 'High']

def test_api_predict_invalid_input(client):
    payload = {'pregnancies':2,'glucose':-50,'bloodpressure':80,
               'skinthickness':35,'insulin':100,'bmi':34.0,'dpf':0.8,'age':45}
    response = client.post('/api/predict',
                           json=payload,
                           content_type='application/json')
    assert response.status_code == 422

def test_api_predict_missing_fields(client):
    payload = {'glucose': 120}
    response = client.post('/api/predict',
                           json=payload,
                           content_type='application/json')
    assert response.status_code == 422

def test_high_glucose_predicts_higher_risk(client):
    low_payload  = {'pregnancies':0,'glucose':85, 'bloodpressure':70,
                    'skinthickness':20,'insulin':80,'bmi':22.0,'dpf':0.2,'age':25}
    high_payload = {'pregnancies':5,'glucose':200,'bloodpressure':90,
                    'skinthickness':35,'insulin':200,'bmi':38.0,'dpf':1.2,'age':55}
    low_res  = client.post('/api/predict', json=low_payload,  content_type='application/json')
    high_res = client.post('/api/predict', json=high_payload, content_type='application/json')
    low_risk  = low_res.get_json()['risk_percent']
    high_risk = high_res.get_json()['risk_percent']
    assert high_risk > low_risk

def test_api_predict_no_data(client):
    response = client.post('/api/predict', content_type='application/json')
    assert response.status_code == 400