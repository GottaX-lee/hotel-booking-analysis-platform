import os
import io
import json
import pandas as pd
import numpy as np
import joblib
from flask import (
    Flask, render_template, request,
    jsonify, send_file
)
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hotel-booking-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== 公共数据加载 =====
def load_kpi_data():
    df = pd.read_csv('data/processed/cleaned_data.csv')
    return {
        'total_orders': f"{len(df):,}",
        'cancel_rate': f"{df['is_canceled'].mean():.2%}",
        'avg_adr': f"\u00a5{df['adr'].mean():.2f}",
        'avg_lead_time': f"{df['lead_time'].mean():.0f} \u5929",
    }

def load_model_evaluation():
    model = joblib.load('models/xgboost_model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    string_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    X_test = X_test.drop(columns=string_cols)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['\u672a\u53d6\u6d88', '\u53d6\u6d88'])

    importance = model.feature_importances_
    feature_names = X_test.columns.tolist()
    feat_imp = pd.DataFrame({
        '\u7279\u5f81': feature_names,
        '\u91cd\u8981\u6027': importance
    }).sort_values('\u91cd\u8981\u6027', ascending=False).head(15)

    not_canceled = report['\u672a\u53d6\u6d88']
    canceled = report['\u53d6\u6d88']
    return {
        'accuracy': f"{acc:.4f}",
        'auc': f"{auc:.4f}",
        'precision_0': f"{not_canceled['precision']:.4f}",
        'recall_0': f"{not_canceled['recall']:.4f}",
        'f1_0': f"{not_canceled['f1-score']:.4f}",
        'precision_1': f"{canceled['precision']:.4f}",
        'recall_1': f"{canceled['recall']:.4f}",
        'f1_1': f"{canceled['f1-score']:.4f}",
        'train_size': '93,941',
        'test_size': '23,486',
        'feature_importance': feat_imp.to_dict('records'),
    }

# ===== 预处理函数 =====
def preprocess_for_prediction(df):
    df = df.copy()
    for col in ['reservation_status', 'reservation_status_date']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0)
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    if 'adr' in df.columns:
        df = df[(df['adr'] > 0) & (df['adr'] <= 500)]
    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    if 'arrival_date_month' in df.columns:
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        df.drop(columns=['arrival_date_month'], inplace=True)
    if 'total_stay' not in df.columns and 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    if 'total_guests' not in df.columns and 'adults' in df.columns and 'children' in df.columns and 'babies' in df.columns:
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
    return df

# ===== 路由：首页 =====
@app.route('/')
def index():
    return render_template('index.html')

# ===== 路由：数据看板 =====
@app.route('/dashboard')
def dashboard():
    kpi = load_kpi_data()
    return render_template('dashboard.html', kpi=kpi)

# ===== 路由：模型评估 =====
@app.route('/model')
def model():
    eval_data = load_model_evaluation()
    return render_template('model.html', eval=eval_data)

# ===== 路由：批量预测页 =====
@app.route('/predict')
def predict():
    return render_template('predict.html')

# ===== 路由：单样本预测页 =====
@app.route('/predict/single')
def predict_single():
    return render_template('predict_single.html')

# ===== 路由：风险预警页 =====
@app.route('/predict/risk')
def predict_risk():
    return render_template('predict_risk.html')

# ===== API：单样本预测 =====
@app.route('/api/predict/single', methods=['POST'])
def api_predict_single():
    try:
        features = request.get_json()
        from src.predict_single import predict_single as ps
        result = ps(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== API：风险预警 =====
@app.route('/api/predict/risk', methods=['POST'])
def api_predict_risk():
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.7))

        model = joblib.load('models/xgboost_model.pkl')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        string_cols = X_test.select_dtypes(include=['object']).columns.tolist()
        X_test_num = X_test.drop(columns=string_cols)

        y_prob = model.predict_proba(X_test_num)[:, 1]
        y_pred = model.predict(X_test_num)

        total_count = len(y_test)
        high_risk_mask = y_prob >= threshold
        high_risk_count = int(high_risk_mask.sum())
        low_risk_count = total_count - high_risk_count
        risk_rate = high_risk_count / total_count if total_count > 0 else 0

        max_records = 200
        high_risk_indices = np.where(high_risk_mask)[0]
        if len(high_risk_indices) > max_records:
            high_risk_indices = high_risk_indices[:max_records]

        raw = pd.read_csv('data/raw/hotel_bookings.csv')
        records = []
        for idx in high_risk_indices:
            raw_row = raw.iloc[idx] if idx < len(raw) else None
            records.append({
                '序号': idx + 1,
                '酒店类型': raw_row['hotel'] if raw_row is not None else '-',
                '提前预订天数': int(raw_row['lead_time']) if raw_row is not None else '-',
                'ADR': round(raw_row['adr'], 2) if raw_row is not None else '-',
                '押金类型': raw_row['deposit_type'] if raw_row is not None else '-',
                '市场细分': raw_row['market_segment'] if raw_row is not None else '-',
                '入住人数': int(raw_row['adults'] + raw_row.get('children', 0) + raw_row.get('babies', 0)) if raw_row is not None else '-',
                'cancel_probability': round(float(y_prob[idx]), 4),
                '实际取消': '是' if y_test[idx] == 1 else '否',
            })

        return jsonify({
            'success': True,
            'total_count': total_count,
            'high_risk_count': high_risk_count,
            'low_risk_count': low_risk_count,
            'risk_rate': risk_rate,
            'records': records,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== 路由：上传预测 =====
@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': '\u672a\u4e0a\u4f20\u6587\u4ef6'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '\u672a\u9009\u62e9\u6587\u4ef6'}), 400
    try:
        df_original = pd.read_csv(file)
        df = preprocess_for_prediction(df_original.copy())

        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        cat_cols = list(label_encoders.keys())
        for col in cat_cols:
            if col in df.columns:
                le = label_encoders[col]
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        num_cols = [
            'lead_time', 'adr', 'total_stay', 'total_guests', 'arrival_month_num',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
            'previous_cancellations', 'previous_bookings_not_canceled',
            'booking_changes', 'days_in_waiting_list', 'total_of_special_requests'
        ]
        available_num = [c for c in num_cols if c in df.columns]
        if available_num:
            df[available_num] = scaler.transform(df[available_num])

        model = joblib.load('models/xgboost_model.pkl')

        X_train_cols = pd.read_csv('data/processed/X_train.csv').columns.tolist()
        string_train = [c for c in X_train_cols if c in pd.read_csv('data/processed/X_train.csv').select_dtypes(include=['object']).columns]
        train_feature_cols = [c for c in X_train_cols if c not in string_train]

        for col in train_feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[train_feature_cols]

        y_pred = model.predict(df)
        y_proba = model.predict_proba(df)[:, 1]

        results = df_original.copy()
        results['predicted_canceled'] = y_pred
        results['cancel_probability'] = np.round(y_proba, 4)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_results.csv')
        results.to_csv(output_path, index=False, encoding='utf-8-sig')

        preview = results.head(20).to_dict(orient='records')

        return jsonify({
            'success': True,
            'total_rows': len(results),
            'canceled_count': int(y_pred.sum()),
            'not_canceled_count': int(len(y_pred) - y_pred.sum()),
            'preview': preview,
            'download_url': '/predict/download',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== 路由：下载预测结果 =====
@app.route('/predict/download')
def predict_download():
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_results.csv'),
        as_attachment=True,
        download_name='prediction_results.csv',
        mimetype='text/csv'
    )

# ===== Serverless Handler（阿里云函数计算 FC3）=====
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
