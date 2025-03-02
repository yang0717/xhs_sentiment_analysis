from config import parse_args
from mamba_postencoding import BaseNdMamba2
from dataset2 import read_Data, Textdataset
from train import train_and_evaluate
from utils import setup_logging
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders(train_texts, train_labels, val_texts, val_labels, max_lengh, batch_size):
    train_dataset = Textdataset(train_texts, train_labels, max_lengh)
    val_dataset = Textdataset(val_texts, val_labels, max_lengh)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__=='__main__':
    setup_logging()
    args = parse_args()
    train_texts, train_labels= read_Data(args.train_path)
    val_texts, val_labels = read_Data(args.val_path)

    train_loader, val_loader = create_dataloaders(train_texts,train_labels,val_texts,val_labels,
                                                  max_lengh=args.max_length,
                                                  batch_size=args.batch_size
                                                  )

    model = BaseNdMamba2(cin=args.cin,
                         cout=args.cout,
                         mamba_dim=args.mamba_dim,
                         vocab_size=args.vocab_size,
                         hidden_size=args.MLP_hidden_size,
                         num_classes=args.num_classes)

    train_and_evaluate(model, train_loader, val_loader,args)