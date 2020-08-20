import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE, BASE_SIZE
from torch.utils.data import Dataset, DataLoader
import sys
import os

TRAIN_EXAMPLES = 90000
# TEST_EXAMPLES = 10000

np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=sys.maxsize)

class SimpleDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]


def get_train_and_test_dataset(data):
    train_set = SimpleDataSet(data[:TRAIN_EXAMPLES])
    test_set = SimpleDataSet(data[TRAIN_EXAMPLES:])
    return train_set, test_set



if __name__ == "__main__":
    dataloader, _ = DataLoader(SimpleDataSet(np.load(PUZZLE_FILE)), batch_size=32)
    x = dataloader.__iter__().__next__()
    print(x.size())
