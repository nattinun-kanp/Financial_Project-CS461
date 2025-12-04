import joblib
import pandas as pd
import numpy as np
import io
import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Optional

# --- Configuration ---
MODEL_PATH = 'super_ensemble_model.joblib'
DATA_PATH = 'UCI_Credit_Card.csv'

# --- Global Variables ---
model = None
feature_names = []
df_global = None  # ตัวแปรเก็บข้อมูลสำหรับหน้า Search

# --- Helper Functions ---
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if 'PAY_0' in data.columns: data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    if 'EDUCATION' in data.columns: data['EDUCATION'] = data['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    if 'MARRIAGE' in data.columns: data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})
    
    limit = data['LIMIT_BAL'] if 'LIMIT_BAL' in data.columns else 1
    bill = data['BILL_AMT1'] if 'BILL_AMT1' in data.columns else 0
    pay1 = data['PAY_1'] if 'PAY_1' in data.columns else 0
    
    # Feature Engineering
    data['util_rate'] = (bill / limit).replace([np.inf, -np.inf], 0).fillna(0)
    data['risk_interaction'] = pay1 * data['util_rate']
    
    # Ensure all features exist
    for col in feature_names:
        if col not in data.columns: data[col] = 0
            
    return data[feature_names]

# ไฟล์: app.py
# แก้ไขในฟังก์ชัน run_prediction

def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
    if model is None: return df
    
    processed = preprocess_input(df)
    probs = model.predict_proba(processed)[:, 1]
    df['Probability'] = probs

    limit = df['LIMIT_BAL'] if 'LIMIT_BAL' in df.columns else 1
    bill = df['BILL_AMT1'] if 'BILL_AMT1' in df.columns else 0
    df['util_rate'] = (bill / limit).replace([np.inf, -np.inf], 0).fillna(0)
    bins = [-1, 0.2, 0.4, 0.6, 1.1]
    
    labels_color = ['red', 'orange', 'yellow', 'green']
    
    df['Risk_Color'] = pd.cut(df['Probability'], bins=bins, labels=labels_color)
    
    bins_segment = [-1, 0.4, 0.7, 1.1]
    labels_segment = ['High Risk', 'Medium Risk', 'Good Payer']
    
    df['Risk_Segment'] = pd.cut(df['Probability'], bins=bins_segment, labels=labels_segment)
    
    return df

def analyze_demographics(df):
    """ฟังก์ชันวิเคราะห์ประชากร (ใช้โค้ดเดิมของคุณ)"""
    mapping_sex = {1: 'Male', 2: 'Female'}
    mapping_edu = {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'}
    mapping_mar = {1: 'Married', 2: 'Single', 3: 'Others'}
    
    df['Label_Sex'] = df['SEX'].map(mapping_sex)
    df['Label_Edu'] = df['EDUCATION'].map(mapping_edu)
    df['Label_Mar'] = df['MARRIAGE'].map(mapping_mar)
    
    bins = [20, 30, 40, 50, 60, 100]
    labels = ['20-30', '31-40', '41-50', '51-60', '60+']
    df['Label_Age'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    
    stats = {}
    for col, name in [('Label_Sex', 'sex'), ('Label_Edu', 'education'), 
                      ('Label_Mar', 'marriage'), ('Label_Age', 'age')]:
        grp = df.groupby([col, 'Risk_Segment'], observed=False).size().unstack(fill_value=0)
        chart_data = {
            "labels": grp.index.tolist(),
            "high_risk": grp.get('High Risk', pd.Series(0, index=grp.index)).tolist(),
            "medium_risk": grp.get('Medium Risk', pd.Series(0, index=grp.index)).tolist(),
            "low_risk": grp.get('Good Payer', pd.Series(0, index=grp.index)).tolist(),
        }
        stats[name] = chart_data
    return stats

# --- Lifespan Manager (โหลด Model และ Data ตอนเริ่ม Server) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names, df_global
    print("🔄 Starting up...")
    
    # 1. Load Model
    try:
        if os.path.exists(MODEL_PATH):
            model_artifact = joblib.load(MODEL_PATH)
            model = model_artifact['model']
            if 'features' in model_artifact: feature_names = model_artifact['features']
            elif 'feature_names' in model_artifact: feature_names = model_artifact['feature_names']
            else: feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'util_rate', 'risk_interaction']
            print(f"✅ Model loaded.")
        else:
            print(f"⚠️ Warning: {MODEL_PATH} not found.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

    # 2. Load Default Data (สำคัญสำหรับหน้า Search)
    try:
        if os.path.exists(DATA_PATH):
            df_global = pd.read_csv(DATA_PATH)
            if model:
                df_global = run_prediction(df_global)
            print(f"✅ Data loaded: {len(df_global)} rows")
        else:
            print(f"⚠️ Warning: {DATA_PATH} not found.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        
    yield
    print("🛑 Shutting down...")

# --- App Definition ---
app = FastAPI(title="Credit Risk Analytics API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# --- Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/search.html")
async def read_search():
    return FileResponse('search.html')

@app.get("/api/model/meta")
async def get_model_meta():
    if not model: raise HTTPException(status_code=503, detail="Model not loaded")
    return {"name": "Super Ensemble", "version": "3.0", "features": feature_names}

@app.get("/api/summary/risk_counts")
async def get_risk_counts():
    if df_global is None: raise HTTPException(status_code=503, detail="Data not available")
    counts = df_global['Risk_Color'].value_counts().to_dict()
    return {
        "total": len(df_global),
        "by_color": {
            "green": counts.get('green', 0),
            "yellow": counts.get('yellow', 0),
            "orange": counts.get('orange', 0),
            "red": counts.get('red', 0)
        }
    }

# 🔥 Endpoint สำคัญสำหรับหน้า Search 🔥
@app.get("/api/debtors")
async def search_debtors(
    query: Optional[str] = "",
    risk: Optional[str] = "",
    page: int = 1,
    page_size: int = 50,
    sort: str = "ID.asc"
):
    if df_global is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    # 1. Filter
    filtered = df_global.copy()
    if query:
        # แปลง ID เป็น string เพื่อค้นหาบางส่วน
        filtered = filtered[filtered['ID'].astype(str).str.contains(query, na=False)]
    if risk:
        filtered = filtered[filtered['Risk_Color'] == risk]
        
    # 2. Sort
    if sort == 'ID.asc':
        filtered = filtered.sort_values('ID', ascending=True)
    elif sort == 'ID.desc':
        filtered = filtered.sort_values('ID', ascending=False)
    elif sort == 'prob.asc':
        filtered = filtered.sort_values('Probability', ascending=True)
    elif sort == 'prob.desc':
        filtered = filtered.sort_values('Probability', ascending=False)
        
    # 3. Pagination
    total_items = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = filtered.iloc[start_idx:end_idx].copy()
    
    # Map column names ให้ตรงกับที่ search.js ต้องการ
    items['prob'] = items['Probability']
    items['risk_class'] = items['Risk_Color']
    items['default_label'] = items['default.payment.next.month']
    
    return {
        "total": total_items,
        "page": page,
        "page_size": page_size,
        "items": items.to_dict(orient="records")
    }

@app.post("/api/admin/rebuild")
async def rebuild_data():
    global df_global
    try:
        df_global = pd.read_csv(DATA_PATH)
        df_global = run_prediction(df_global)
        return {"success": True, "message": "Data rebuilt successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/analyze_portfolio")
async def analyze_portfolio_endpoint(file: UploadFile = File(...)):
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df = run_prediction(df) # ใช้ฟังก์ชันกลางทำนายผล
        
        # คำนวณ Stats (Overview + Demographics)
        total_customers = len(df)
        total_exposure = df['BILL_AMT1'].sum()
        risky_exposure = df[df['Risk_Segment'] == 'High Risk']['BILL_AMT1'].sum()
        
        segments = {}
        for seg in ['Good Payer', 'Medium Risk', 'High Risk']:
            sub = df[df['Risk_Segment'] == seg]
            segments[seg] = {"count": len(sub), "value": float(sub['BILL_AMT1'].sum())}

        demographics = analyze_demographics(df)
        
        top_risk = df.sort_values(by="Probability", ascending=False).head(10)[
            ['ID', 'Probability', 'Risk_Segment', 'LIMIT_BAL', 'BILL_AMT1']
        ].to_dict(orient='records')

        return {
            "overview": {
                "total_customers": total_customers,
                "total_exposure": float(total_exposure),
                "risky_exposure": float(risky_exposure),
                "segments": segments
            },
            "demographics": demographics,
            "top_risky_customers": top_risk
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(segment: Optional[str] = "All"):
    if df_global is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    df = df_global.copy()
    
    # 1. Metrics (Scorecards)
    total_debtors = len(df)
    total_exposure = df['BILL_AMT1'].sum() if 'BILL_AMT1' in df.columns else 0
    
    # 2. Risk Distribution (4 สี)
    risk_labels = ['green', 'yellow', 'orange', 'red']
    counts = df['Risk_Color'].value_counts().reindex(risk_labels, fill_value=0)
    risk_dist = {"labels": risk_labels, "data": counts.tolist()}
    
    risk_dist = {
        "labels": risk_labels,
        "data": counts.tolist()
    }
    
    # 3. Demographics
    sex_map = {1: 'Male', 2: 'Female'}
    s_counts = df['SEX'].map(sex_map).fillna('Other').value_counts()
    demographics = {}
    demographics['sex'] = {"labels": s_counts.index.tolist(), "data": s_counts.values.tolist()}
    
    edu_map = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
    e_counts = df['EDUCATION'].map(edu_map).fillna('Others').value_counts()
    demographics['education'] = {"labels": e_counts.index.tolist(), "data": e_counts.values.tolist()}
    
    mar_map = {1: 'Married', 2: 'Single', 3: 'Others'}
    m_counts = df['MARRIAGE'].map(mar_map).fillna('Others').value_counts()
    demographics['marriage'] = {"labels": m_counts.index.tolist(), "data": m_counts.values.tolist()}
    
    #4. Monthly Trends (Apr - Sep) ---
    # Mapping: 6=Apr, 5=May, ..., 1=Sep
    months_label = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
    pay_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
    
    # คำนวณผลรวมแต่ละเดือน
    trend_bill = [df[c].sum() if c in df.columns else 0 for c in bill_cols]
    trend_pay = [df[c].sum() if c in df.columns else 0 for c in pay_cols]
    
    trends = {
        "months": months_label,
        "bill": [float(x) for x in trend_bill],
        "pay": [float(x) for x in trend_pay]
    }
    
    # 5. Good Payer Insights ---
    # ดึงข้อมูลเฉพาะกลุ่ม Good Payer (สีเขียว)
    good_payers = df[df['Risk_Color'] == 'green']
    if len(good_payers) > 0:
        gp_count = len(good_payers)
        gp_avg_limit = good_payers['LIMIT_BAL'].mean()
        gp_total_pay = good_payers['PAY_AMT1'].sum()
    else:
        gp_count = 0
        gp_avg_limit = 0
        gp_total_pay = 0
        
    good_payer_stats = {
        "count": int(gp_count),
        "avg_limit": float(gp_avg_limit),
        "total_pay_latest": float(gp_total_pay)
    }
    # 6 Table Data
    # ใช้ Risk_Segment ที่แบ่ง 3 ระดับไว้แล้ว
    # เรียงลำดับ: High Risk, Medium Risk, Good Payer
    seg_labels = ['High Risk', 'Medium Risk', 'Good Payer']
    table_data = []
    
    for seg in seg_labels:
        sub_df = df[df['Risk_Segment'] == seg]
        count = len(sub_df)
        exposure = sub_df['BILL_AMT1'].sum() if 'BILL_AMT1' in sub_df.columns else 0
        
        table_data.append({
            "segment": seg,
            "count": count,
            "exposure": float(exposure)
        })
    months_label = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
    pay_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
    trend_bill = [df[c].sum() if c in df.columns else 0 for c in bill_cols]
    trend_pay = [df[c].sum() if c in df.columns else 0 for c in pay_cols]
    trends = {"months": months_label, "bill": [float(x) for x in trend_bill], "pay": [float(x) for x in trend_pay]}

    # Demographics
    sex_map = {1: 'Male', 2: 'Female'}
    s_counts = df['SEX'].map(sex_map).fillna('Other').value_counts()
    demographics = {}
    demographics['sex'] = {"labels": s_counts.index.tolist(), "data": s_counts.values.tolist()}
    
    edu_map = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
    e_counts = df['EDUCATION'].map(edu_map).fillna('Others').value_counts()
    demographics['education'] = {"labels": e_counts.index.tolist(), "data": e_counts.values.tolist()}
    
    mar_map = {1: 'Married', 2: 'Single', 3: 'Others'}
    m_counts = df['MARRIAGE'].map(mar_map).fillna('Others').value_counts()
    demographics['marriage'] = {"labels": m_counts.index.tolist(), "data": m_counts.values.tolist()}
    

    return {
        "metrics": {"total_debtors": total_debtors, "total_exposure": float(total_exposure)},
        "risk_dist": risk_dist,
        "table_data": table_data, # ส่งข้อมูลตารางกลับไป
        "trends": trends,
        "demographics": demographics
    }
    