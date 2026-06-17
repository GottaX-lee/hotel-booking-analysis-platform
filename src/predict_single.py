import pandas as pd
import numpy as np
import joblib
import torch
import os

def predict_single(features: dict):
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')

    df = pd.DataFrame([features])

    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    if 'arrival_date_month' in df.columns:
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        df.drop(columns=['arrival_date_month'], inplace=True)

    if 'arrival_month_num' not in df.columns:
        df['arrival_month_num'] = 7

    for col in ['reservation_status', 'reservation_status_date', 'company']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0)
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')

    if 'total_stay' not in df.columns and 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    if 'total_guests' not in df.columns and 'adults' in df.columns and 'children' in df.columns and 'babies' in df.columns:
        df['total_guests'] = df['adults'] + df['children'] + df['babies']

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

    X_train = pd.read_csv('data/processed/X_train.csv')
    string_train = X_train.select_dtypes(include=['object']).columns.tolist()
    train_feature_cols = [c for c in X_train.columns if c not in string_train]

    for col in train_feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_feature_cols]

    y_prob = model.predict_proba(df)[:, 1][0]
    y_pred = int(model.predict(df)[0])

    return {
        'prediction': y_pred,
        'probability': round(float(y_prob), 4),
        'label': '取消' if y_pred == 1 else '未取消'
    }
