import torch.nn as nn
import torch.nn.functional as F
from nd_mamba2 import Mamba2



class BaseNdMamba2(nn.Module):
    def __init__(self, cin, cout, mamba_dim, vocab_size, hidden_size, num_classes,  **mamba2_args):
        super().__init__()
        assert mamba_dim % 64 == 0, "mamba_dim 必须是64的倍数"
        self.embedding = nn.Embedding(vocab_size, cin)
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)  # 调整通道数到mamba_dim
        n_layer = mamba2_args.get('n_layer', 1)
        self.n_layer = n_layer

        # 创建包含 n_layer 层的正向和反向 Mamba2 模型
        self.mamba2_for_layers = nn.ModuleList([
            Mamba2(mamba_dim, **mamba2_args) for _ in range(n_layer)
        ])  # 正向
        self.mamba2_back_layers = nn.ModuleList([
            Mamba2(mamba_dim, **mamba2_args) for _ in range(n_layer)
        ])  # 反向
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)  # 调整通道数到cout

        # 修改为4层MLP，并添加LayerNorm层
        self.classification = nn.Sequential(
            nn.Linear(cout, hidden_size),
            nn.LayerNorm(hidden_size),  # 批归一化
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 批归一化
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), # 批归一化
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        # self.drop = nn.Dropout(dropout_prob) # Dropout 层


    def forward(self, x):
        x = self.embedding(x)  # x.shape(batch_size, seq_len, mamba_dim; 16, 128, 64)
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将x pad到64的倍数
        x = self.fc_in(x)  # 调整通道数为目标通道数

        # 逐层通过 Mamba2 层
        for i in range(self.n_layer):
            x1, _ = self.mamba2_for_layers[i](x)
            x2, _ = self.mamba2_back_layers[i](x.flip(1))
            x2 = x2.flip(1)
            x = x + x1 + x2 # 使用残差连接

        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.mean(dim=1)   # 对序列维度进行均值池化
        x = self.classification(x)  # MLP分类器
        return x
