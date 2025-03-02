import argparse

def parse_args():
    parser = argparse.ArgumentParser('Text classification Training Script', add_help=False)

    #模型参数
    parser.add_argument('--vocab_size', default=50277, type=int, help='BERT Base vocab size')
    parser.add_argument('--cin', default=768, type=int, help='in_channel of mamba')
    parser.add_argument('--mamba_dim', default=256, type=int, help='Hidden size of mamba')
    parser.add_argument('--cout', default=256, type=int, help='out_channel of mamba')
    parser.add_argument('--MLP_hidden_size', default=128, type=int, help='out_channel of mamba')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes for binary classification')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of mambablock layers')
    parser.add_argument('--max_length', default=150, type=int, help='Max token length')
    parser.add_argument('--dropout', default=0.1, type=int, help='Max token length')


    #训练参数
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', '--lr', default=1e-5, type=float, help='Learning rate (default: 2e-5)')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs (default: 5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization strength')

    #数据地址
    parser.add_argument('--train_path', default='./data/weibo_my_10k/train.csv', type=str, help='Train data Path')
    parser.add_argument('--val_path', default='./data/weibo_my_10k/test.csv', type=str, help='val data Path')
    parser.add_argument('--tokenizer_name', default='./bert-base-chinese', type=str, help='tokenizer pre-model Path')

    args = parser.parse_args()

    return args