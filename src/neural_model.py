import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

class HotelBookingNN(nn.Module):
    """酒店预订取消预测神经网络模型"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rates=[0.3, 0.3, 0.2]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))

def train_neural_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=256, lr=0.001):
    """训练神经网络模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转换为PyTorch张量
    X_train_t = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_t = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
    y_val_t = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).reshape(-1, 1)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = HotelBookingNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练历史
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            val_predicted = (val_outputs > 0.5).float()
            val_correct = (val_predicted == y_val_t.to(device)).sum().item()
            val_acc = val_correct / len(y_val_t)

        history['loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['acc'].append(correct / total)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {history['loss'][-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/neural_model.pth')
    joblib.dump({'input_dim': X_train.shape[1], 'hidden_dims': [256, 128, 64], 'dropout_rates': [0.3, 0.3, 0.2]}, 'models/neural_config.pkl')

    return model, history

def predict_neural(model, X, config=None):
    """使用神经网络进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_t)
    return outputs.cpu().numpy().flatten()
