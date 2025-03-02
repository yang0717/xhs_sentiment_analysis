from config import parse_args
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        # 如果text不是字符串类型，返回空字符串
        return ''
    # 删除网址
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove special characters
    text = re.sub(r"[^A-Za-z\s]", "", text)

    return text.lower().strip()

def map_label(label):
    if label == 'Positive':
        return 0
    elif label == 'Negative':
        return 1
    elif label == 'Irrelevant':
        return 2
    elif label == 'Neutral':
        return 3



def read_Data(file):
    # 使用 open() 打开文件并忽略无法解码的字符
    with open(file, 'r', encoding='utf-8') as f:
        # 通过pandas读取csv数据
        df = pd.read_csv(f)

    texts = df['text'].fillna('').tolist() # 读取第二列的文本
    labels = df['target'].tolist()  # 读取第一列的标签

    # texts = [clean_text(text) for text in texts]
    # labels = [map_label(label) for label in labels]

    # 将标签转换为整数
    labels = [int(label) for label in labels]

    return texts, labels


# def read_Data(file):
#     # 读取文件
#     all_data = open(file, "r", encoding="utf-8").read().split("\n")
#     # 得到所有文本、所有标签、句子的最大长度
#     texts, labels, max_length = [], [], []
#     for data in all_data:
#         if data:
#             text, label = data.split("\t")
#             max_length.append(len(text))
#             texts.append(text)
#             labels.append(label)
#     # 根据不同的数据集返回不同的内容
#     if os.path.split(file)[1] == "train.txt":
#         max_len = max(max_length)
#         return texts, labels, max_len
#     return texts, labels



class Textdataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = BertTokenizer.from_pretrained(parse_args().tokenizer_name)

    def __getitem__(self, index):
        # 取出一条数据并截断长度
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        # 分词
        text_id = self.tokenizer.tokenize(text)
        # 加上起始标志
        text_id = ["[CLS]"] + text_id

        # 编码
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # 掩码  -》
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # 编码后  -》长度一致
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))
        # str -》 int
        label = int(label)

        # 转化成tensor
        token_ids = torch.tensor(token_ids,dtype=torch.long)
        mask = torch.tensor(mask,dtype=torch.long)
        label = torch.tensor(label,dtype=torch.long)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label = read_Data("./data/chnsenticorp_htl/train.csv")
    max_len = 158
    print(train_text[0], train_label[0])
    trainDataset = Textdataset(train_text, train_label, max_len)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)

    for data in trainDataloader:
        batch_data,batch_label = data
        print(batch_data,batch_label)
