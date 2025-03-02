import torch
from transformers import BertTokenizer
from mamba_postencoding import BaseNdMamba2  # 假设您在 model.py 中定义了模型
from config import parse_args

# 定义预测函数
def predict_(text):
    # 选择设备（GPU或CPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')  # 根据需要选择其他预训练模型

    # 定义数据转换（预处理）函数
    def preprocess(text):
        # 使用BERT分词器进行编码
        encoded_dict = tokenizer.encode_plus(
            text,  # 输入文本
            add_special_tokens=True,  # 添加特殊tokens [CLS] 和 [SEP]
            max_length=152,  # 最大长度，根据需要调整
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 超出部分截断
            return_attention_mask=True,
            return_tensors='pt',  # 返回PyTorch tensors
        )
        return encoded_dict

    # 对输入文本进行预处理
    encoded_input = preprocess(text)
    input_ids = encoded_input['input_ids'].to(device)
    print(input_ids.shape)
    attention_mask = encoded_input['attention_mask'].to(device)

    # 初始化模型
    # 假设模型参数与训练时一致，如预训练模型名称、类别数等
    model = BaseNdMamba2(cin=args.cin,
                         cout=args.cout,
                         mamba_dim=args.mamba_dim,
                         vocab_size=args.vocab_size,
                         hidden_size=args.MLP_hidden_size,
                         num_classes=args.num_classes)  # 根据实际类别数调整

    # 加载模型权重
    model_weight_pth = './best_model.pth'
    model.load_state_dict(torch.load(model_weight_pth, map_location=device))

    # 将模型移动到指定设备
    model.to(device)

    # 设置模型为评估模式
    model.eval()

    # 定义类别索引到标签的映射
    class_indict = {'1': '负面', '0': '正面'}

    with torch.no_grad():
        # 获取模型输出
        output = model(input_ids)

        # 计算预测概率（假设输出为logits）
        predict = torch.softmax(output, dim=1)

        # 获取预测的类别索引
        predict_cla = torch.argmax(predict, dim=1).item()

    # 返回预测的标签和对应的概率
    return class_indict[str(predict_cla)], predict[0][predict_cla].item()


# 示例用法
if __name__ == '__main__':
    input_file = './data.txt'  # 替换为您的文本文档路径

    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            text = line.strip()
            if text:
                label, probability = predict_(text)
                print(f"第{idx}行预测类别: {label}, 概率: {probability:.4f}")
