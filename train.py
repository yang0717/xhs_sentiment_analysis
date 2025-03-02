import torch
import torch.nn as nn
from utils import plot_confusion_matrix, plot_metrics_separated
from sklearn.metrics import f1_score
from torch.optim import AdamW, Adam, RMSprop
from tqdm import tqdm
import torch.nn.init as init
import logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initialize_weights(model):
    """初始化模型权重"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 使用He初始化方法初始化权重，适用于ReLU激活函数
            if isinstance(param, torch.nn.Conv2d) or isinstance(param, torch.nn.Linear):
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        elif 'bias' in name:
            # 偏置初始化为0
            init.zeros_(param)

def train_and_evaluate(model, train_loader, val_loader, args, patience=5):

    print("模型的层名：")
    for name, layer in model.named_modules():
        print(name)

    criterion = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    # 初始化权重
    model.apply(initialize_weights)

    model = model.to(device)
    best_val_acc = 0.0
    best_model_path = "best_model.pth"
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []


    for epoch in range(args.num_epochs):
        #训练阶段
        model.train()

        train_loss, correct_train, total_train = 0, 0, 0
        y_true_train, y_pred_train = [], []
        train_features = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.num_epochs}"):
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # labels = batch['label'].to(device)
            texts, labels = batch
            input_ids = texts[0].to(device)
            input_ids = input_ids
            labels = labels.to(device)
            outputs = model(input_ids)

            loss = criterion(outputs, labels)

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            # 累加训练损失
            train_loss += loss.item()
            total_train += labels.size(0)

            # 预测和真实值
            # probabilities = torch.sigmoid(outputs)
            # preds = (probabilities >= 0.5).long()
            #C损失的预测
            _, preds = torch.max(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        # 计算训练精度与 F1
        train_accuracy = correct_train / total_train
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), desc=f"Validation Epoch {epoch + 1}/{args.num_epochs}"):
                # input_ids = batch['input_ids'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                # labels = batch['label'].to(device)
                texts, labels = batch
                input_ids = texts[0].to(device)
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)

                # 累加验证损失
                val_loss += loss.item()
                total_val += labels.size(0)

                # 预测和真实值
                _, preds = torch.max(outputs, dim=1)#C损失的预测值

                correct_val += (preds == labels).sum().item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())


        #计算验证集平均损失并进行学习率调整
        val_loss /= len(val_loader)


        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_without_improvement = 0
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= patience:
        #         print(f"Early stopping at epoch {epoch + 1}")
        #         break

        # scheduler.step(val_loss)
        # 计算验证精度与 F1
        val_accuracy = correct_val / total_val
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_log = (f"Epoch {epoch + 1}/{args.num_epochs}\n"
                    f"Train Loss: {train_loss / len(train_loader):.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}\n"
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        print(epoch_log)
        logging.info(epoch_log)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型, 验证集准确率: {best_val_acc:.4f}")




    # 绘制混淆矩阵热力图
    plot_confusion_matrix(y_true_val, y_pred_val)
    # 绘制指标曲线
    plot_metrics_separated(train_losses, val_losses, train_accuracies, val_accuracies)

    print(f"Training complete. Best validation accuracy: {max(val_accuracies):.4f}")
    logging.info(f"Training complete. Best validation accuracy: {max(val_accuracies):.4f}")


    print(f"训练完成，最佳验证集准确率: {best_val_acc:.4f}, 模型已保存至 {best_model_path}")

    return model