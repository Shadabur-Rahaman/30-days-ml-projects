import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32)
        self.y1 = torch.tensor(df.iloc[:, -2].values, dtype=torch.long)
        self.y2 = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]
