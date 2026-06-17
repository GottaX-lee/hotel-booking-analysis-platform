import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'cyan': '#17becf',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
}

os.makedirs('static/images', exist_ok=True)

df = pd.read_csv('data/processed/cleaned_data.csv')
print(f'数据加载完成: {df.shape[0]} 行')

# ===== 1. KPI 概览卡片 =====
print('1/5 生成 KPI 概览卡片...')
total_orders = len(df)
cancel_rate = df['is_canceled'].mean()
avg_adr = df['adr'].mean()
avg_lead_time = df['lead_time'].mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
metrics = [
    (f'总订单数\n{total_orders:,}', COLORS['blue']),
    (f'取消率\n{cancel_rate:.2%}', COLORS['red']),
    (f'平均房价 (ADR)\n\u00a5{avg_adr:.2f}', COLORS['green']),
    (f'平均提前预订天数\n{avg_lead_time:.0f} 天', COLORS['orange']),
]
for ax, (text, color) in zip(axes.flat, metrics):
    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=16, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.set_frame_on(False)
    ax.grid(False)
fig.tight_layout()
fig.savefig('static/images/kpi_cards.png', dpi=200, bbox_inches='tight')
plt.close()

# ===== 2. 取消率对比柱状图 =====
print('2/5 生成取消率对比柱状图...')
cancel_rate_hotel = df.groupby('hotel')['is_canceled'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(
    cancel_rate_hotel['hotel'],
    cancel_rate_hotel['is_canceled'],
    color=[COLORS['blue'], COLORS['orange']],
    alpha=0.85, edgecolor='white', linewidth=1.5, width=0.5
)
for bar, rate in zip(bars, cancel_rate_hotel['is_canceled']):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{rate:.2%}',
        ha='center', va='bottom', fontsize=13, fontweight='bold'
    )
ax.set_xlabel('酒店类型', fontsize=13)
ax.set_ylabel('取消率', fontsize=13)
ax.set_title('按酒店类型划分的取消率对比', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, cancel_rate_hotel['is_canceled'].max() * 1.2)
fig.tight_layout()
fig.savefig('static/images/cancel_rate_by_hotel.png', dpi=200, bbox_inches='tight')
plt.close()

# ===== 3. 时间趋势折线图 =====
print('3/5 生成时间趋势折线图...')
df_trend = df.copy()
df_trend['arrival_date_month'] = pd.to_datetime(
    df_trend['arrival_date_month'], format='%B'
).dt.month
df_trend['ym'] = pd.to_datetime(
    df_trend['arrival_date_year'].astype(str) + '-' +
    df_trend['arrival_date_month'].astype(str) + '-01'
)
monthly = df_trend.groupby('ym')['is_canceled'].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    monthly['ym'], monthly['is_canceled'],
    color=COLORS['blue'], lw=2.5, marker='o', ms=4,
    label='取消率'
)
std_val = monthly['is_canceled'].std()
ax.fill_between(
    monthly['ym'],
    monthly['is_canceled'] - std_val,
    monthly['is_canceled'] + std_val,
    alpha=0.12, color=COLORS['blue'], label='\u00b11 \u6807\u51c6\u5dee'
)
ax.set_xlabel('月份', fontsize=13)
ax.set_ylabel('取消率', fontsize=13)
ax.set_title('按月的取消率趋势 (2015-2017)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('static/images/monthly_cancel_trend.png', dpi=200, bbox_inches='tight')
plt.close()

# ===== 4. 特征分布直方图 =====
print('4/5 生成特征分布直方图...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lt_mean = df['lead_time'].mean()
lt_median = df['lead_time'].median()
axes[0].hist(
    df['lead_time'], bins=50,
    color=COLORS['blue'], alpha=0.75,
    edgecolor='white', linewidth=0.5
)
axes[0].axvline(lt_mean, color=COLORS['red'],
                ls='--', lw=2, label=f'均值: {lt_mean:.0f}')
axes[0].axvline(lt_median, color=COLORS['green'],
                ls='-.', lw=2, label=f'中位数: {lt_median:.0f}')
axes[0].set_xlabel('提前预订天数 (lead_time)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title('Lead Time 分布', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

adr_mean = df['adr'].mean()
adr_median = df['adr'].median()
axes[1].hist(
    df['adr'], bins=50,
    color=COLORS['orange'], alpha=0.75,
    edgecolor='white', linewidth=0.5
)
axes[1].axvline(adr_mean, color=COLORS['red'],
                ls='--', lw=2, label=f'均值: {adr_mean:.2f}')
axes[1].axvline(adr_median, color=COLORS['green'],
                ls='-.', lw=2, label=f'中位数: {adr_median:.2f}')
axes[1].set_xlabel('日均房价 (adr)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('ADR 分布', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig('static/images/feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()

# ===== 5. 相关性热力图 =====
print('5/5 生成相关性热力图...')
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
corr_with_target = corr_matrix[['is_canceled']].sort_values(
    by='is_canceled', ascending=False
)

fig, ax = plt.subplots(figsize=(8, max(6, len(corr_with_target) * 0.4)))
sns.heatmap(
    corr_with_target.T,
    annot=True, fmt='.3f', cmap='RdBu_r',
    center=0, vmin=-1, vmax=1,
    linewidths=0.8, linecolor='white',
    cbar_kws={'label': '相关系数', 'shrink': 0.8},
    ax=ax
)
ax.set_title('各特征与取消状态 (is_canceled) 的相关性', fontsize=14, fontweight='bold')
ax.set_xlabel('特征', fontsize=12)
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
fig.savefig('static/images/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

print()
print('所有静态图表已保存到 static/images/ 目录')
print()
print('====== 图表清单 ======')
for f in sorted(os.listdir('static/images')):
    size_kb = os.path.getsize(f'static/images/{f}') / 1024
    print(f'  {f}  ({size_kb:.1f} KB)')
