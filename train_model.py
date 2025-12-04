# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Configuration
DATA_PATH = 'UCI_Credit_Card.csv'
MODEL_PATH = 'super_ensemble_model.joblib'
RANDOM_SEED = 42

def load_data(path):
    df = pd.read_csv(path)
    if 'PAY_0' in df.columns:
        df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    return df

def preprocess_features(df):
    data = df.copy()
    data['EDUCATION'] = data['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})
    
    # Feature Engineering
    data['util_rate'] = data['BILL_AMT1'] / data['LIMIT_BAL']
    data['util_rate'] = data['util_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    data['risk_interaction'] = data['PAY_1'] * data['util_rate']
    
    return data

def train_super_ensemble(X_train, y_train):
    print("⚙️ Building Models...")
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED
    )
    
    # แก้ Warning: เพิ่ม max_iter จาก 300 เป็น 1000 เพื่อให้ MLP เทรนจบ
    mlp_clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            activation='relu', 
            max_iter=1000, # <--- แก้ตรงนี้
            random_state=RANDOM_SEED
        )
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=RANDOM_SEED
    )
    
    ensemble = VotingClassifier(
        estimators=[('gb', gb_clf), ('mlp', mlp_clf), ('rf', rf_clf)],
        voting='soft',
        weights=[2, 1, 1]
    )
    
    print("🚀 Training Ensemble Model...")
    ensemble.fit(X_train, y_train)
    return ensemble

if __name__ == "__main__":
    print("📥 Loading Data...")
    df = load_data(DATA_PATH)
    df_processed = preprocess_features(df)
    
    target_col = 'default.payment.next.month'
    X = df_processed.drop(columns=['ID', target_col])
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    model = train_super_ensemble(X_train, y_train)
    
    print("\n📊 --- Model Performance ---")
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"✅ AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # บันทึก Model โดยใช้ Key ชื่อ 'features' (เพื่อให้ตรงกับ app.py)
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    joblib.dump({
        'model': model, 
        'features': X_train.columns.tolist() # <--- Key นี้สำคัญมาก
    }, MODEL_PATH)
    print("🎉 Done! Model is ready.")