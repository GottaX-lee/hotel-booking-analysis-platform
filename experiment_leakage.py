import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 200, 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

os.makedirs('static/images', exist_ok=True)

# ===== 加载第5步的干净测试集结果 =====
X_test_clean = pd.read_csv('data/processed/X_test.csv')
y_test_clean = pd.read_csv('data/processed/y_test.csv').values.ravel()

# 获取训练时使用的特征列
string_cols = X_test_clean.select_dtypes(include=['object']).columns.tolist()
train_feature_cols = X_test_clean.drop(columns=string_cols).columns.tolist()
print(f'训练使用的特征列数: {len(train_feature_cols)}')

# ===== 加载原始数据 =====
print('\n====== 数据泄露危害量化对照实验 ======')
raw = pd.read_csv('data/raw/hotel_bookings.csv')
print(f'原始数据: {raw.shape}')

# ===== 泄漏率分析：reservation_status vs is_canceled =====
print('\n====== 泄漏率分析：reservation_status vs is_canceled ======')
leakage_crosstab = pd.crosstab(raw['reservation_status'], raw['is_canceled'], margins=True)
print(leakage_crosstab)

# 详细分析
canceled_mask = raw['reservation_status'] == 'Canceled'
checkout_mask = raw['reservation_status'] == 'Check-Out'
canceled_yes = canceled_mask.sum()
canceled_and_iscanceled = (canceled_mask & (raw['is_canceled'] == 1)).sum()
checkout_and_notcanceled = (checkout_mask & (raw['is_canceled'] == 0)).sum()

print(f'\nreservation_status="Canceled" 且 is_canceled=1: {canceled_and_iscanceled}/{canceled_yes} = {canceled_and_iscanceled/canceled_yes*100:.2f}%')
print(f'reservation_status="Check-Out" 且 is_canceled=0: {checkout_and_notcanceled}/{checkout_mask.sum()} = {checkout_and_notcanceled/checkout_mask.sum()*100:.2f}%')
print(f'\n结论: reservation_status 几乎完全决定了 is_canceled，属于严重数据泄露！')

# ===== 预处理（与训练数据相同流程）=====
print('\n====== 数据预处理 ======')
df = raw.copy()

# 删除泄露列和 company
df = df.drop(columns=['reservation_status', 'reservation_status_date', 'company'])
print(f'删除泄露列后: {df.shape}')

# 填充空值
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['country'] = df['country'].fillna('Unknown')

# 剔除 ADR 异常值
before = len(df)
df = df[(df['adr'] > 0) & (df['adr'] <= 500)]
print(f'剔除 adr 异常 {before - len(df)} 条，剩余: {len(df)}')

# 添加特征
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']
month_map = {
    'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
}
df['arrival_month_num'] = df['arrival_date_month'].apply(lambda x: month_map[x])

# ===== 应用 Label Encoding =====
label_encoders = joblib.load('models/label_encoders.pkl')
cat_cols = list(label_encoders.keys())
for col in cat_cols:
    le = label_encoders[col]
    df[col] = df[col].astype(str)
    # 处理未见过的类别 - 映射为 -1 或 0
    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    df[col] = le.transform(df[col])

print(f'Label Encoding 完成')

# ===== 应用 StandardScaler =====
scaler = joblib.load('models/scaler.pkl')
num_cols = [
    'lead_time', 'adr', 'total_stay', 'total_guests', 'arrival_month_num',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'total_of_special_requests'
]
# 只对存在的数值列做标准化
available_num_cols = [c for c in num_cols if c in df.columns]
df[available_num_cols] = scaler.transform(df[available_num_cols])
print(f'StandardScaler 完成')

# ===== 只保留训练时使用的特征列 =====
y_leakage = df['is_canceled'].values
X_leakage = df.drop(columns=['is_canceled'])
# 与训练特征对齐
for col in train_feature_cols:
    if col not in X_leakage.columns:
        X_leakage[col] = 0
X_leakage = X_leakage[train_feature_cols]
print(f'泄露数据集形状: {X_leakage.shape}')

# ===== 加载模型并预测 =====
model = joblib.load('models/xgboost_model.pkl')
print('\n====== 预测 ======')
y_pred_leak = model.predict(X_leakage)
y_prob_leak = model.predict_proba(X_leakage)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

acc_leak = accuracy_score(y_leakage, y_pred_leak)
auc_leak = roc_auc_score(y_leakage, y_prob_leak)

print(f'泄露数据预测准确率: {acc_leak:.4f}')
print(f'泄露数据预测 AUC: {auc_leak:.4f}')
print(f'\n分类报告:')
print(classification_report(y_leakage, y_pred_leak, target_names=['未取消', '取消']))

# ===== 混淆矩阵 =====
cm_leak = confusion_matrix(y_leakage, y_pred_leak)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_leak, annot=True, fmt='d', cmap='Reds',
            xticklabels=['未取消', '取消'], yticklabels=['未取消', '取消'],
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xlabel('预测', fontsize=12)
ax.set_ylabel('真实', fontsize=12)
ax.set_title('数据泄露数据混淆矩阵', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('static/images/leakage_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n混淆矩阵已保存: static/images/leakage_confusion_matrix.png')

# ===== 对比表格 =====
acc_base = 0.8851
auc_base = 0.9564
acc_inflate = (acc_leak - acc_base) / acc_base * 100
auc_inflate = (auc_leak - auc_base) / auc_base * 100

print('\n' + '=' * 70)
print('数据泄露危害量化对照实验 - 对比表格')
print('=' * 70)
print(f'{"评估对象":<30} {"准确率":<15} {"AUC":<15} {"说明":<20}')
print('-' * 70)
print(f'{"干净测试集（第5步）":<30} {acc_base:<15.4f} {auc_base:<15.4f} {"模型真实泛化能力":<20}')
print(f'{"含泄露列的旧数据":<30} {acc_leak:<15.4f} {auc_leak:<15.4f} {"数据泄露导致的虚高":<20}')
print('-' * 70)
print(f'{"虚高百分比":<30} {"+":<14}{acc_inflate:.2f}% {"+":<14}{auc_inflate:.2f}% {"虚高幅度":<20}')
print('=' * 70)

print(f'\n准确率虚高: {acc_inflate:.2f}%')
print(f'AUC 虚高: {auc_inflate:.2f}%')
