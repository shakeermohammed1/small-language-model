from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, tokenized_data, block_size=512):
        self.data = tokenized_data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
