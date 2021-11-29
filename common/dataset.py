import random
import torch
from torch.utils.data import Dataset
import sys
sys.path.insert(0, '..')


class WholeDataset(Dataset):
    def __init__(self, len):
        self.len = len
    def __getitem__(self, index):
        return torch.randn(3,512,512), random.randrange(0,10)
    def __len__(self):
        return  self.len


if __name__ == "__main__":
    pass