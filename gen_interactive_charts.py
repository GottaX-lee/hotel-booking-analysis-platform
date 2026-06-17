import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs('static/html', exist_ok=True)

df = pd.read_csv('data/processed/cleaned_data.csv')
print(f'数据加载完成: {df.shape[0]} 行')

COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
}

# ===== 1. KPI 卡片 =====
print('1/5 生成 KPI 卡片...')
total_orders = len(df)
cancel_rate = df['is_canceled'].mean()
avg_adr = df['adr'].mean()
avg_lead_time = df['lead_time'].mean()

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]],
    subplot_titles=('总订单数', '取消率', '平均房价 (ADR)', '平均提前预订天数')
)

fig.add_trace(go.Indicator(
    mode="number",
    value=total_orders,
    number={'font': {'size': 40, 'color': COLORS['blue']}}
), row=1, col=1)

fig.add_trace(go.Indicator(
    mode="number+delta",
    value=cancel_rate * 100,
    number={'suffix': '%', 'font': {'size': 40, 'color': COLORS['red']}},
    delta={'reference': 30, 'valueformat': '.1f'}
), row=1, col=2)

fig.add_trace(go.Indicator(
    mode="number",
    value=round(avg_adr, 2),
    number={'prefix': '\u00a5', 'font': {'size': 40, 'color': COLORS['green']}}
), row=2, col=1)

fig.add_trace(go.Indicator(
    mode="number",
    value=round(avg_lead_time),
    number={'suffix': ' 天', 'font': {'size': 40, 'color': COLORS['orange']}}
), row=2, col=2)

fig.update_layout(
    title={'text': '酒店预订数据总览', 'x': 0.5, 'font': {'size': 22}},
    height=500, template='plotly_white'
)
fig.write_html('static/html/kpi_cards.html')
print('  -> static/html/kpi_cards.html')

# ===== 2. 取消率对比柱状图 =====
print('2/5 生成取消率对比柱状图...')
cancel_rate_hotel = df.groupby('hotel')['is_canceled'].mean().reset_index()
cancel_rate_hotel['取消率'] = (cancel_rate_hotel['is_canceled'] * 100).round(2)
cancel_rate_hotel['订单数'] = df.groupby('hotel').size().values

fig = px.bar(
    cancel_rate_hotel, x='hotel', y='取消率',
    color='hotel',
    color_discrete_map={
        'Resort Hotel': COLORS['blue'],
        'City Hotel': COLORS['orange']
    },
    text='取消率',
    hover_data={'取消率': ':.2f', '订单数': True, 'hotel': False},
    title='按酒店类型划分的取消率对比',
    labels={'hotel': '酒店类型', '取消率': '取消率 (%)'},
    height=500,
)
fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)
fig.update_layout(
    template='plotly_white',
    xaxis_title='酒店类型',
    yaxis_title='取消率 (%)',
    showlegend=False,
)
fig.write_html('static/html/cancel_rate_by_hotel.html')
print('  -> static/html/cancel_rate_by_hotel.html')

# ===== 3. 时间趋势折线图（带滑块） =====
print('3/5 生成时间趋势折线图...')
df_trend = df.copy()
df_trend['arrival_date_month'] = pd.to_datetime(
    df_trend['arrival_date_month'], format='%B'
).dt.month
df_trend['ym'] = pd.to_datetime(
    df_trend['arrival_date_year'].astype(str) + '-' +
    df_trend['arrival_date_month'].astype(str) + '-01'
)
monthly = df_trend.groupby('ym')['is_canceled'].agg(['mean', 'std']).reset_index()
monthly['取消率'] = (monthly['mean'] * 100).round(2)
monthly['上界'] = ((monthly['mean'] + monthly['std']) * 100).round(2)
monthly['下界'] = ((monthly['mean'] - monthly['std']).clip(lower=0) * 100).round(2)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly['ym'], y=monthly['上界'],
    mode='lines', line={'width': 0}, showlegend=False,
    name='上界'
))
fig.add_trace(go.Scatter(
    x=monthly['ym'], y=monthly['下界'],
    mode='lines', line={'width': 0},
    fill='tonexty', fillcolor='rgba(31, 119, 180, 0.12)',
    showlegend=False, name='下界'
))
fig.add_trace(go.Scatter(
    x=monthly['ym'], y=monthly['取消率'],
    mode='lines+markers',
    line={'color': COLORS['blue'], 'width': 2.5},
    marker={'size': 6, 'color': COLORS['blue']},
    name='取消率',
    hovertemplate='<b>%{x|%Y-%m}</b><br>取消率: %{y:.2f}%<extra></extra>'
))

fig.update_layout(
    title={'text': '按月聚合的取消率趋势 (2015-2017)', 'x': 0.5, 'font': {'size': 20}},
    xaxis={
        'title': '月份',
        'rangeslider': {'visible': True, 'thickness': 0.06},
        'rangeselector': {
            'buttons': [
                {'count': 6, 'label': '6个月', 'step': 'month', 'stepmode': 'backward'},
                {'count': 12, 'label': '1年', 'step': 'month', 'stepmode': 'backward'},
                {'count': 24, 'label': '2年', 'step': 'month', 'stepmode': 'backward'},
                {'step': 'all', 'label': '全部'}
            ]
        },
    },
    yaxis={'title': '取消率 (%)'},
    hovermode='x unified',
    template='plotly_white',
    height=550,
)
fig.write_html('static/html/monthly_cancel_trend.html')
print('  -> static/html/monthly_cancel_trend.html')

# ===== 4. 特征分布直方图 =====
print('4/5 生成特征分布直方图...')
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('提前预订天数 (lead_time) 分布', '日均房价 (adr) 分布'),
    horizontal_spacing=0.12
)

fig.add_trace(go.Histogram(
    x=df['lead_time'], nbinsx=50,
    marker_color=COLORS['blue'],
    marker_line_color='white', marker_line_width=0.5,
    opacity=0.75, name='lead_time',
    hovertemplate='区间: %{x}<br>频数: %{y}<extra></extra>'
), row=1, col=1)

lt_mean = df['lead_time'].mean()
lt_median = df['lead_time'].median()
fig.add_vline(x=lt_mean, line_dash='dash', line_color=COLORS['red'],
              line_width=2, row=1, col=1)
fig.add_vline(x=lt_median, line_dash='dot', line_color=COLORS['green'],
              line_width=2, row=1, col=1)

fig.add_trace(go.Histogram(
    x=df['adr'], nbinsx=50,
    marker_color=COLORS['orange'],
    marker_line_color='white', marker_line_width=0.5,
    opacity=0.75, name='adr',
    hovertemplate='区间: %{x}<br>频数: %{y}<extra></extra>'
), row=1, col=2)

adr_mean = df['adr'].mean()
adr_median = df['adr'].median()
fig.add_vline(x=adr_mean, line_dash='dash', line_color=COLORS['red'],
              line_width=2, row=1, col=2)
fig.add_vline(x=adr_median, line_dash='dot', line_color=COLORS['green'],
              line_width=2, row=1, col=2)

fig.update_layout(
    title={'text': '特征分布直方图', 'x': 0.5, 'font': {'size': 20}},
    template='plotly_white',
    height=450,
    bargap=0.02,
    hovermode='x unified',
)
fig.update_xaxes(title_text='提前预订天数', row=1, col=1)
fig.update_yaxes(title_text='频数', row=1, col=1)
fig.update_xaxes(title_text='日均房价', row=1, col=2)
fig.update_yaxes(title_text='频数', row=1, col=2)
fig.write_html('static/html/feature_distributions.html')
print('  -> static/html/feature_distributions.html')

# ===== 5. 相关性热力图 =====
print('5/5 生成相关性热力图...')
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
corr_with_target = corr_matrix[['is_canceled']].sort_values(
    by='is_canceled', ascending=False
)

fig = go.Figure(data=go.Heatmap(
    z=corr_with_target.T.values,
    x=corr_with_target.index.tolist(),
    y=['is_canceled'],
    colorscale='RdBu_r',
    zmid=0,
    zmin=-1, zmax=1,
    text=[[f'{v:.3f}' for v in corr_with_target.T.values[0]]],
    texttemplate='%{text}',
    textfont={'size': 12, 'color': 'black'},
    hovertemplate='<b>%{x}</b><br>相关系数: %{z:.3f}<extra></extra>',
    colorbar={
        'title': '相关系数',
        'titleside': 'right',
        'thickness': 20,
        'len': 0.8,
    }
))

fig.update_layout(
    title={'text': '各特征与取消状态 (is_canceled) 的相关性', 'x': 0.5, 'font': {'size': 20}},
    xaxis={'tickangle': -45, 'title': '特征'},
    yaxis={'title': ''},
    template='plotly_white',
    height=max(400, len(corr_with_target) * 30 + 100),
    margin={'l': 80, 'r': 80, 't': 80, 'b': 120},
)
fig.write_html('static/html/correlation_heatmap.html')
print('  -> static/html/correlation_heatmap.html')

print()
print('所有交互式图表已保存到 static/html/ 目录')
print()
print('====== HTML 文件清单 ======')
for f in sorted(os.listdir('static/html')):
    size_kb = os.path.getsize(f'static/html/{f}') / 1024
    print(f'  {f}  ({size_kb:.1f} KB)')
