import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

class BookingLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :])

def create_sequences(data, seq_length=6):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def run_lstm_forecast(save_path_prefix='static/images/lstm'):
    os.makedirs('static/images', exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv('data/processed/cleaned_data.csv')
    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    df['arrival_date_month_num'] = df['arrival_date_month'].map(month_map)
    df['date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month_num'].astype(str) + '-01'
    )

    monthly = df.groupby('date').agg(
        total_bookings=('is_canceled', 'count'),
        canceled=('is_canceled', 'sum')
    ).reset_index().sort_values('date')
    monthly['cancel_rate'] = monthly['canceled'] / monthly['total_bookings']
    monthly.columns = ['date', 'total_bookings', 'canceled', 'cancel_rate']

    monthly.to_csv('data/processed/monthly_bookings.csv', index=False)

    series = monthly['total_bookings'].values.astype(float)

    seq_length = 6
    X, y = create_sequences(series, seq_length)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mean, std = X_train.mean(), X_train.std()
    X_train_norm = (X_train - mean) / std
    y_train_norm = (y_train - mean) / std
    X_test_norm = (X_test - mean) / std
    y_test_raw = y_test

    X_train_t = torch.FloatTensor(X_train_norm).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train_norm).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test_norm).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = BookingLSTM(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 30 == 0 or epoch == 1:
            print(f'Epoch [{epoch:3d}/{epochs}] Loss: {epoch_loss / len(train_loader):.4f}')

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t.to(device)).cpu().numpy().flatten() * std + mean
        test_preds = model(X_test_t.to(device)).cpu().numpy().flatten() * std + mean

    train_mae = np.mean(np.abs(train_preds - y_train))
    test_mae = np.mean(np.abs(test_preds - y_test_raw))
    print(f'\nTrain MAE: {train_mae:.1f} | Test MAE: {test_mae:.1f}')

    future_months = 12
    last_seq = series[-seq_length:].copy()
    future_preds = []
    for _ in range(future_months):
        seq_norm = (last_seq[-seq_length:] - mean) / std
        with torch.no_grad():
            pred = model(torch.FloatTensor(seq_norm).unsqueeze(0).unsqueeze(-1).to(device)).item() * std + mean
        future_preds.append(max(0, pred))
        last_seq = np.append(last_seq, pred)

    last_date = monthly['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(monthly['date'], monthly['total_bookings'],
            'b-o', label='历史订单数', markersize=4, linewidth=1.5)

    test_start = len(monthly) - len(y_test_raw)
    ax.plot(monthly['date'].iloc[test_start:], y_test_raw,
            'g--', label='测试集预测', linewidth=1.5, alpha=0.7)

    ax.plot(future_dates, future_preds,
            'r--s', label='LSTM 预测（未来12个月）', markersize=5, linewidth=2)

    ax.axvline(x=monthly['date'].iloc[-1], color='gray', linestyle=':', alpha=0.5, label='预测起点')
    ax.fill_between(future_dates,
                     [max(0, p * 0.85) for p in future_preds],
                     [p * 1.15 for p in future_preds],
                     alpha=0.15, color='red', label='±15% 置信区间')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('月订单数', fontsize=12)
    ax.set_title('月度订单量 LSTM 预测', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{save_path_prefix}_forecast.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'预测图已保存: {save_path_prefix}_forecast.png')

    torch.save(model.state_dict(), 'models/lstm_model.pth')
    print(f'LSTM模型已保存: models/lstm_model.pth')

    forecast_results = []
    for i, (d, p) in enumerate(zip(future_dates, future_preds)):
        forecast_results.append({
            'month': d.strftime('%Y-%m'),
            'predicted_bookings': round(p, 0),
            'trend': '上升' if i > 0 and p > future_preds[i-1] else ('下降' if i > 0 else '--')
        })

    cancel_series = monthly['canceled'].values.astype(float)
    X_c, y_c = create_sequences(cancel_series, seq_length)
    split_c = int(len(X_c) * 0.8)
    mean_c, std_c = X_c[:split_c].mean(), X_c[:split_c].std()
    X_c_train_norm = (X_c[:split_c] - mean_c) / std_c
    y_c_train_norm = (y_c[:split_c] - mean_c) / std_c
    X_c_train_t = torch.FloatTensor(X_c_train_norm).unsqueeze(-1)
    y_c_train_t = torch.FloatTensor(y_c_train_norm).unsqueeze(-1)

    model_c = BookingLSTM(input_size=1, hidden_size=64, num_layers=2).to(device)
    optimizer_c = optim.Adam(model_c.parameters(), lr=0.01)
    ds_c = TensorDataset(X_c_train_t, y_c_train_t)
    dl_c = DataLoader(ds_c, batch_size=8, shuffle=True)

    for ep in range(80):
        model_c.train()
        for bx, by in dl_c:
            bx, by = bx.to(device), by.to(device)
            optimizer_c.zero_grad()
            loss = criterion(model_c(bx), by)
            loss.backward()
            optimizer_c.step()

    model_c.eval()
    last_cancel_seq = cancel_series[-seq_length:].copy()
    future_cancels = []
    for _ in range(future_months):
        seq_norm = (last_cancel_seq[-seq_length:] - mean_c) / std_c
        with torch.no_grad():
            pred = model_c(torch.FloatTensor(seq_norm).unsqueeze(0).unsqueeze(-1).to(device)).item() * std_c + mean_c
        future_cancels.append(max(0, pred))
        last_cancel_seq = np.append(last_cancel_seq, pred)

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(monthly['date'], monthly['cancel_rate'] * 100,
             'b-o', label='历史取消率', markersize=4, linewidth=1.5)
    cancel_rate_pred = np.array(future_cancels) / np.array(future_preds) * 100
    ax2.plot(future_dates, cancel_rate_pred,
             'r--s', label='预测取消率（未来12个月）', markersize=5, linewidth=2)
    ax2.axhline(y=monthly['cancel_rate'].mean() * 100, color='gray', linestyle='--', alpha=0.5, label=f'历史均值 ({monthly["cancel_rate"].mean()*100:.1f}%)')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('取消率 (%)', fontsize=12)
    ax2.set_title('月度取消率 LSTM 预测', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f'{save_path_prefix}_cancel_rate.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'取消率预测图已保存: {save_path_prefix}_cancel_rate.png')

    print('\n=== LSTM 季节性预测结果 ===')
    print(f'训练 MAE: {train_mae:.1f} | 测试 MAE: {test_mae:.1f}')
    print(f'\n未来12个月预测:')
    for r in forecast_results:
        print(f'  {r["month"]}: {r["predicted_bookings"]:.0f} 单 ({r["trend"]})')

    result = {
        'forecast': forecast_results,
        'future_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'future_bookings': [round(p, 0) for p in future_preds],
        'future_cancel_rates': [round(float(c), 2) for c in cancel_rate_pred],
        'test_mae': round(test_mae, 1),
        'train_mae': round(train_mae, 1),
    }

    import json
    with open('data/processed/forecast_results.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'预测结果已保存: data/processed/forecast_results.json')

    return result

if __name__ == '__main__':
    run_lstm_forecast()
