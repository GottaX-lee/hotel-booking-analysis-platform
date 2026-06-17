import pandas as pd
import numpy as np
import joblib
import json

_cache = {}

def _load_resources():
    if 'model' not in _cache:
        _cache['model'] = joblib.load('models/xgboost_model.pkl')
        _cache['scaler'] = joblib.load('models/scaler.pkl')
        _cache['encoders'] = joblib.load('models/label_encoders.pkl')
        X_train = pd.read_csv('data/processed/X_train.csv')
        string_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        _cache['train_feature_cols'] = [c for c in X_train.columns if c not in string_cols]
        month_map = {
            'January':1,'February':2,'March':3,'April':4,
            'May':5,'June':6,'July':7,'August':8,
            'September':9,'October':10,'November':11,'December':12
        }
        _cache['month_map'] = month_map
    return _cache

def get_default_scenario():
    return {
        'hotel': 'Resort Hotel',
        'lead_time': 100,
        'adr': 100,
        'total_guests': 2,
        'total_stay': 3,
        'total_of_special_requests': 0,
        'previous_cancellations': 0,
        'booking_changes': 0,
        'arrival_date_year': 2016,
        'arrival_date_month': 'July',
        'market_segment': 'Online TA',
        'deposit_type': 'No Deposit',
        'customer_type': 'Transient',
        'meal': 'BB',
        'country': 'PRT',
        'reserved_room_type': 'A',
        'assigned_room_type': 'A',
        'distribution_channel': 'TA/TO',
        'required_car_parking_spaces': 0,
        'is_repeated_guest': 0,
        'adults': 2, 'children': 0, 'babies': 0,
        'agent': 0,
        'stays_in_weekend_nights': 1,
        'stays_in_week_nights': 2,
        'previous_bookings_not_canceled': 0,
        'days_in_waiting_list': 0,
        'arrival_date_week_number': 27,
        'arrival_date_day_of_month': 15,
    }

def predict_scenario(features: dict) -> dict:
    resources = _load_resources()
    model = resources['model']
    scaler = resources['scaler']
    encoders = resources['encoders']
    month_map = resources['month_map']
    train_feature_cols = resources['train_feature_cols']

    df = pd.DataFrame([features])

    if 'arrival_date_month' in df.columns:
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        df.drop(columns=['arrival_date_month'], inplace=True)

    for col in ['reservation_status', 'reservation_status_date', 'company']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0)
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')

    if 'total_stay' not in df.columns:
        df['total_stay'] = df.get('stays_in_weekend_nights', 1) + df.get('stays_in_week_nights', 2)
    if 'total_guests' not in df.columns:
        df['total_guests'] = df.get('adults', 2) + df.get('children', 0) + df.get('babies', 0)

    cat_cols = list(encoders.keys())
    for col in cat_cols:
        if col in df.columns:
            le = encoders[col]
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

    for col in train_feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_feature_cols]

    y_prob = float(model.predict_proba(df)[:, 1][0])
    y_pred = int(model.predict(df)[0])

    return {'prediction': y_pred, 'probability': round(y_prob, 4), 'label': '\u53d6\u6d88' if y_pred == 1 else '\u672a\u53d6\u6d88'}

def run_simulation(scenario_a: dict, scenario_b: dict) -> dict:
    result_a = predict_scenario(scenario_a)
    result_b = predict_scenario(scenario_b)
    diff = abs(result_a['probability'] - result_b['probability'])
    changed = {}
    for key in scenario_a:
        if key in scenario_b and str(scenario_a[key]) != str(scenario_b[key]):
            changed[key] = {'before': scenario_a[key], 'after': scenario_b[key]}
    return {
        'scenario_a': {**result_a, 'params': scenario_a},
        'scenario_b': {**result_b, 'params': scenario_b},
        'probability_diff': round(diff, 4),
        'changed_features': changed,
    }
