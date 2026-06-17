import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.neural_model import train_neural_model

print("加载数据...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 删除字符串列（arrival_date_month 已由 arrival_month_num 替代）
string_cols = X_train.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    X_train = X_train.drop(columns=string_cols)
    X_test = X_test.drop(columns=string_cols)
    print(f"已删除字符串列: {string_cols}")
    print(f"处理后训练集: {X_train.shape}")

# 划分验证集（从训练集中再分10%做验证）
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print("训练神经网络...")
model, history = train_neural_model(
    X_train_split, y_train_split, X_val, y_val,
    epochs=100, batch_size=256
)

print("训练完成！")
print(f"最终训练准确率: {history['acc'][-1]:.4f}")
print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")

# 保存训练历史
import json
with open('models/training_history.json', 'w') as f:
    json.dump(history, f)
print("模型已保存到 models/neural_model.pth")

# 绘制训练过程可视化
from src.visualization import plot_training_history
plot_training_history(history, save_path='static/images/training_history.png')
