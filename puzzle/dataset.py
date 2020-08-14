import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE
from torch.utils.data import Dataset, DataLoader


TRAIN_EXAMPLES = 90000
# TEST_EXAMPLES = 10000

class SaeDataSet(Dataset):
    def __init__(self, is_train):
        data = np.load(PUZZLE_FILE)
        if is_train:
            self.data = data[:TRAIN_EXAMPLES]
        else:
            self.data = data[TRAIN_EXAMPLES:]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]



if __name__ == "__main__":
    dataloader = DataLoader(SaeDataSet(is_train=True), batch_size=32)
    x = dataloader.__iter__().__next__()
    print(x.size())
