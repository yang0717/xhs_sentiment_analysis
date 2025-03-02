import os
import logging  # 确保导入了 logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def setup_logging(log_file='结果.txt'):
    # 设置日志记录器
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 每次运行时重写日志
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
# 添加指标曲线绘制：损失和精度分开绘制
def plot_metrics_separated(train_losses, val_losses, train_accuracies, val_accuracies,save_dir="results/"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linestyle='-', marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", linestyle='-', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()

    # 精度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Train Accuracy", linestyle='-', marker='o')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", linestyle='-', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.show()

# 添加混淆矩阵热力图
def plot_confusion_matrix(y_true, y_pred, save_dir="results/"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [f"Class {i}" for i in range(2)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()