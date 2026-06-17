import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 200, 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# 删除仍为字符串类型的列（arrival_date_month 已由 arrival_month_num 替代）
string_cols = X_train.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    print(f'删除字符串列（已由数值特征替代）: {string_cols}')
    X_train = X_train.drop(columns=string_cols)
    X_test = X_test.drop(columns=string_cols)

print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

# =============================================
# 第一阶段：基线模型
# =============================================
print('\n' + '=' * 50)
print('第一阶段：基线模型训练')
print('=' * 50)

baseline = XGBClassifier(random_state=42, verbosity=0)
baseline.fit(X_train, y_train)

y_pred_base = baseline.predict(X_test)
y_prob_base = baseline.predict_proba(X_test)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_prob_base)

print(f'准确率 (Accuracy): {acc_base:.4f}')
print(f'ROC-AUC: {auc_base:.4f}')
print(f'\n分类报告:')
print(classification_report(y_test, y_pred_base, target_names=['未取消', '取消']))

cm_base = confusion_matrix(y_test, y_pred_base)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未取消', '取消'], yticklabels=['未取消', '取消'],
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xlabel('预测', fontsize=12)
ax.set_ylabel('真实', fontsize=12)
ax.set_title('基线模型混淆矩阵', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('static/images/confusion_matrix_baseline.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n混淆矩阵已保存: static/images/confusion_matrix_baseline.png')

# =============================================
# 第二阶段：优化模型
# =============================================
print('\n' + '=' * 50)
print('第二阶段：优化模型训练')
print('=' * 50)

optimized = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    random_state=42,
    verbosity=0,
)
optimized.fit(X_train, y_train)

y_pred_opt = optimized.predict(X_test)
y_prob_opt = optimized.predict_proba(X_test)[:, 1]

acc_opt = accuracy_score(y_test, y_pred_opt)
auc_opt = roc_auc_score(y_test, y_prob_opt)

print(f'准确率 (Accuracy): {acc_opt:.4f}')
print(f'ROC-AUC: {auc_opt:.4f}')
print(f'\n分类报告:')
print(classification_report(y_test, y_pred_opt, target_names=['未取消', '取消']))

cm_opt = confusion_matrix(y_test, y_pred_opt)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未取消', '取消'], yticklabels=['未取消', '取消'],
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xlabel('预测', fontsize=12)
ax.set_ylabel('真实', fontsize=12)
ax.set_title('优化模型混淆矩阵', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('static/images/confusion_matrix_optimized.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n混淆矩阵已保存: static/images/confusion_matrix_optimized.png')

# ===== 特征重要性 =====
importance = optimized.feature_importances_
feature_names = X_train.columns.tolist()
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': importance
}).sort_values('重要性', ascending=False)

top15 = importance_df.head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    range(len(top15)),
    top15['重要性'].values,
    color='#1f77b4', alpha=0.85, edgecolor='white', linewidth=0.5
)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['特征'].values)
ax.invert_yaxis()
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_ylabel('特征', fontsize=12)
ax.set_title('Top 15 特征重要性排名', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, top15['重要性'].values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

fig.tight_layout()
fig.savefig('static/images/feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n特征重要性图已保存: static/images/feature_importance.png')

print(f'\n=== Top 15 特征重要性排名 ===')
for i, (_, row) in enumerate(top15.iterrows(), 1):
    print(f'  {i:2d}. {row["特征"]:30s} {row["重要性"]:.4f}')

# ===== 保存模型 =====
joblib.dump(optimized, 'models/xgboost_model.pkl')
print(f'\n模型已保存: models/xgboost_model.pkl')

# ===== 对比输出 =====
print('\n' + '=' * 60)
print('基线模型 vs 优化模型 对比')
print('=' * 60)
print(f'{"指标":<20} {"基线模型":<20} {"优化模型":<20} {"提升":<20}')
print('-' * 60)
print(f'{"准确率":<20} {acc_base:<20.4f} {acc_opt:<20.4f} {acc_opt - acc_base:<+20.4f}')
print(f'{"ROC-AUC":<20} {auc_base:<20.4f} {auc_opt:<20.4f} {auc_opt - auc_base:<+20.4f}')
print('=' * 60)
