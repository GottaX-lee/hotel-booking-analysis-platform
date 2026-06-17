import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import warnings
import json
warnings.filterwarnings('ignore')

def analyze_shap(sample_size=500, save_path='static/images/shap_summary.png'):
    os.makedirs('static/images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    model = joblib.load('models/xgboost_model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    string_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    X_test = X_test.drop(columns=string_cols)

    if sample_size and sample_size < len(X_test):
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test

    print(f'SHAP 样本量: {len(X_sample)}')

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    if 'learner' in config and 'learner_model_param' in config['learner']:
        imp = config['learner']['learner_model_param']
        if isinstance(imp.get('base_score'), str) and '[' in imp['base_score']:
            imp['base_score'] = '0.5'
            booster.load_config(json.dumps(config))
            print('已修复 base_score 格式')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values, X_sample,
        feature_names=X_sample.columns.tolist(),
        show=False, max_display=20
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    abs_mean = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        '特征': X_sample.columns,
        '平均绝对SHAP值': abs_mean
    }).sort_values('平均绝对SHAP值', ascending=False)

    print(f'\nSHAP 图已保存: {save_path}')
    print(f'\n=== SHAP 分析 Top 15 特征重要性 ===')
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f'  {i:2d}. {row["特征"]:30s} {row["平均绝对SHAP值"]:.4f}')

    top5 = importance_df.head(5)['特征'].tolist()
    print(f'\n=== SHAP Top 5 重要特征 ===')
    for i, f in enumerate(top5, 1):
        print(f'  {i}. {f}')

    return importance_df, shap_values, X_sample

if __name__ == '__main__':
    analyze_shap(sample_size=500)
