import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE
from torch.utils.data import Dataset, DataLoader


class SaeDataSet(Dataset):
    def __init__(self):
        self.data = np.load(PUZZLE_FILE)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]



if __name__ == "__main__":
    dataloader = DataLoader(SaeDataSet(), batch_size=32)
    x = dataloader.__iter__().__next__()
    print(x.size())
