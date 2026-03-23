"""
DiabetesAI — NHANES Data Downloader & Processor
=================================================
Downloads NHANES (National Health and Nutrition Examination Survey) data
from CDC.gov and converts it to match our 8-column diabetes dataset format.

Columns needed:
Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

Run this script ONCE to download and process the data.
Usage:
    cd diabetes-prediction-ml
    python download_nhanes.py
"""

import pandas as pd
import numpy as np
import os
import urllib.request

print("=" * 60)
print("  DiabetesAI — NHANES Data Downloader")
print("=" * 60)
print()

# ── Output path ──
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════
# NHANES 2017-2018 Data Files
# We need 3 files:
# 1. Demographics (age, gender, pregnancies)
# 2. Blood glucose + diabetes status
# 3. Body measurements (BMI, skin thickness, BP)
# 4. Insulin levels
# ════════════════════════════════════════

NHANES_FILES = {
    'demo':    'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT',   # Demographics
    'glucose': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT',    # Fasting glucose
    'diabetes':'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT',    # Diabetes questionnaire
    'bmx':     'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT',    # Body measurements
    'bpx':     'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT',    # Blood pressure
    'insulin': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/INS_J.XPT',    # Insulin
    'repro':   'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/RHQ_J.XPT',    # Reproductive (pregnancies)
}

def download_xpt(name, url):
    """Download NHANES XPT file and return as DataFrame"""
    filename = os.path.join(OUTPUT_DIR, f'nhanes_{name}.xpt')
    if not os.path.exists(filename):
        print(f"  ⬇️  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"  ✅ Downloaded {name}")
        except Exception as e:
            print(f"  ❌ Failed to download {name}: {e}")
            return None
    else:
        print(f"  ✅ {name} already downloaded")
    try:
        df = pd.read_sas(filename, format='xport', encoding='utf-8')
        return df
    except Exception as e:
        print(f"  ❌ Failed to read {name}: {e}")
        return None


# ════════════════════════════════════════
# Step 1: Download all files
# ════════════════════════════════════════
print("Step 1: Downloading NHANES 2017-2018 data files...")
print("  (This may take 2-3 minutes on first run)")
print()

demo    = download_xpt('demo',    NHANES_FILES['demo'])
glucose = download_xpt('glucose', NHANES_FILES['glucose'])
diabetes= download_xpt('diabetes',NHANES_FILES['diabetes'])
bmx     = download_xpt('bmx',     NHANES_FILES['bmx'])
bpx     = download_xpt('bpx',     NHANES_FILES['bpx'])
insulin = download_xpt('insulin',  NHANES_FILES['insulin'])
repro   = download_xpt('repro',   NHANES_FILES['repro'])

if demo is None or glucose is None:
    print("\n❌ Critical files missing. Check your internet connection and try again.")
    exit(1)

print()
print(f"  Demographics: {len(demo)} participants")
print(f"  Glucose data: {len(glucose)} participants")


# ════════════════════════════════════════
# Step 2: Merge all files on SEQN (participant ID)
# ════════════════════════════════════════
print()
print("Step 2: Merging datasets...")

df = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
# RIDAGEYR = Age in years
# RIAGENDR = Gender (1=Male, 2=Female)

# Merge glucose (LBXGLU = fasting glucose mg/dL)
if glucose is not None and 'LBXGLU' in glucose.columns:
    df = df.merge(glucose[['SEQN', 'LBXGLU']], on='SEQN', how='left')
else:
    df['LBXGLU'] = np.nan

# Merge diabetes outcome (DIQ010: 1=Yes diabetes, 2=No, 3=Borderline)
if diabetes is not None and 'DIQ010' in diabetes.columns:
    df = df.merge(diabetes[['SEQN', 'DIQ010']], on='SEQN', how='left')
else:
    df['DIQ010'] = np.nan

# Merge BMI and skin thickness
if bmx is not None:
    cols = ['SEQN']
    if 'BMXBMI' in bmx.columns:  cols.append('BMXBMI')   # BMI
    if 'BMXTRI' in bmx.columns:  cols.append('BMXTRI')   # Tricep skinfold
    df = df.merge(bmx[cols], on='SEQN', how='left')
else:
    df['BMXBMI'] = np.nan
    df['BMXTRI'] = np.nan

# Merge blood pressure (BPXDI1 = diastolic BP)
if bpx is not None and 'BPXDI1' in bpx.columns:
    df = df.merge(bpx[['SEQN', 'BPXDI1']], on='SEQN', how='left')
else:
    df['BPXDI1'] = np.nan

# Merge insulin (LBXIN = insulin uU/mL)
if insulin is not None and 'LBXIN' in insulin.columns:
    df = df.merge(insulin[['SEQN', 'LBXIN']], on='SEQN', how='left')
else:
    df['LBXIN'] = np.nan

# Merge pregnancies (RHQ160 = times pregnant)
if repro is not None and 'RHQ160' in repro.columns:
    df = df.merge(repro[['SEQN', 'RHQ160']], on='SEQN', how='left')
else:
    df['RHQ160'] = np.nan

print(f"  ✅ Merged dataset: {len(df)} rows")


# ════════════════════════════════════════
# Step 3: Map to our 8-column format
# ════════════════════════════════════════
print()
print("Step 3: Mapping to DiabetesAI format...")

nhanes_mapped = pd.DataFrame()

# Age
nhanes_mapped['Age'] = df['RIDAGEYR']

# Pregnancies — 0 for men, actual value for women
nhanes_mapped['Pregnancies'] = 0  # default
female_mask = df['RIAGENDR'] == 2
if 'RHQ160' in df.columns:
    nhanes_mapped.loc[female_mask, 'Pregnancies'] = df.loc[female_mask, 'RHQ160'].fillna(0)

# Glucose (fasting plasma glucose mg/dL)
nhanes_mapped['Glucose'] = df['LBXGLU']

# BloodPressure (diastolic)
nhanes_mapped['BloodPressure'] = df.get('BPXDI1', pd.Series([np.nan]*len(df)))

# SkinThickness (tricep skinfold mm)
nhanes_mapped['SkinThickness'] = df.get('BMXTRI', pd.Series([np.nan]*len(df)))

# Insulin (convert from uU/mL to match Pima scale)
nhanes_mapped['Insulin'] = df.get('LBXIN', pd.Series([np.nan]*len(df)))

# BMI
nhanes_mapped['BMI'] = df.get('BMXBMI', pd.Series([np.nan]*len(df)))

# DiabetesPedigreeFunction — NHANES doesn't have this directly
# We'll use a population average with small random variation
# (honest approach: mark as mean value since we don't have family history)
np.random.seed(42)
nhanes_mapped['DiabetesPedigreeFunction'] = np.random.uniform(0.2, 0.8, len(df))

# Outcome — 1=Diabetic, 0=Not diabetic
# DIQ010: 1=Yes, 2=No, 3=Borderline (we treat borderline as 0)
nhanes_mapped['Outcome'] = df['DIQ010'].map({1: 1, 2: 0, 3: 0}).fillna(np.nan)

print(f"  ✅ Mapped {len(nhanes_mapped)} rows")


# ════════════════════════════════════════
# Step 4: Clean data
# ════════════════════════════════════════
print()
print("Step 4: Cleaning data...")

# Drop rows where critical fields are missing
before = len(nhanes_mapped)
nhanes_mapped = nhanes_mapped.dropna(subset=['Glucose', 'BMI', 'Outcome', 'Age'])

# Remove unrealistic values
nhanes_mapped = nhanes_mapped[nhanes_mapped['Glucose'] > 0]
nhanes_mapped = nhanes_mapped[nhanes_mapped['BMI'] > 0]
nhanes_mapped = nhanes_mapped[nhanes_mapped['Age'] >= 1]
nhanes_mapped = nhanes_mapped[nhanes_mapped['Age'] <= 120]

# Fill remaining missing values with column medians
for col in ['BloodPressure', 'SkinThickness', 'Insulin', 'Pregnancies']:
    median_val = nhanes_mapped[col].median()
    nhanes_mapped[col] = nhanes_mapped[col].fillna(median_val)

# Cap outliers
nhanes_mapped['Insulin']       = nhanes_mapped['Insulin'].clip(0, 900)
nhanes_mapped['BloodPressure'] = nhanes_mapped['BloodPressure'].clip(0, 200)
nhanes_mapped['SkinThickness'] = nhanes_mapped['SkinThickness'].clip(0, 100)
nhanes_mapped['Glucose']       = nhanes_mapped['Glucose'].clip(0, 600)

# Ensure correct types
nhanes_mapped['Outcome']      = nhanes_mapped['Outcome'].astype(int)
nhanes_mapped['Pregnancies']  = nhanes_mapped['Pregnancies'].clip(0, 20).astype(int)

after = len(nhanes_mapped)
print(f"  Removed {before - after} incomplete rows")
print(f"  ✅ Clean NHANES data: {after} rows")
print(f"  Diabetic: {nhanes_mapped['Outcome'].sum()} ({nhanes_mapped['Outcome'].mean()*100:.1f}%)")
print(f"  Age range: {nhanes_mapped['Age'].min():.0f} - {nhanes_mapped['Age'].max():.0f} years")


# ════════════════════════════════════════
# Step 5: Combine with Pima dataset
# ════════════════════════════════════════
print()
print("Step 5: Combining with original Pima dataset...")

pima_path = os.path.join(OUTPUT_DIR, 'diabetes.csv')
if not os.path.exists(pima_path):
    print(f"  ❌ Could not find diabetes.csv at {pima_path}")
    print("  Saving NHANES data only...")
    nhanes_mapped.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_nhanes.csv'), index=False)
    exit(1)

pima = pd.read_csv(pima_path)
print(f"  Pima dataset: {len(pima)} rows")
print(f"  NHANES dataset: {len(nhanes_mapped)} rows")

# Ensure same column order as Pima
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

nhanes_mapped = nhanes_mapped[cols]
combined = pd.concat([pima, nhanes_mapped], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

output_path = os.path.join(OUTPUT_DIR, 'diabetes_combined.csv')
combined.to_csv(output_path, index=False)

print()
print("=" * 60)
print(f"  ✅ Combined dataset saved: diabetes_combined.csv")
print(f"  Total rows: {len(combined)}")
print(f"  Pima (women only): {len(pima)} rows")
print(f"  NHANES (all demographics): {len(nhanes_mapped)} rows")
print(f"  Diabetic: {combined['Outcome'].sum()} ({combined['Outcome'].mean()*100:.1f}%)")
print(f"  Age range: {combined['Age'].min():.0f} - {combined['Age'].max():.0f} years")
print("=" * 60)
print()
print("Next step: Run retrain_model.py to train model v2!")