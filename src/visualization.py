import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_path='static/images/training_history.png'):
    os.makedirs('static/images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['loss']) + 1)

    axes[0, 0].plot(epochs, history['loss'], 'b-', label='训练损失', lw=1.5)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', lw=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss 曲线', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['acc'], 'b-', label='训练准确率', lw=1.5)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', lw=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('准确率曲线', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['loss'], 'g-', label='训练损失', lw=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('训练损失趋势', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['val_acc'], 'purple', label='验证准确率', lw=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('验证准确率趋势', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'训练历史图已保存: {save_path}')
