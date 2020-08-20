import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE, BASE_SIZE
from torch.utils.data import Dataset, DataLoader
import sys
import os

TRAIN_EXAMPLES = 90000
# TEST_EXAMPLES = 10000

np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=sys.maxsize)

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


PUZZLE_FILE_FO = "puzzle/puzzle_data/puzzles_fo.npy"

class FoSaeDataSet(Dataset):
        def __init__(self, is_train):
            if not os.path.isfile(PUZZLE_FILE_FO):
                data = np.load(PUZZLE_FILE)
                self.data = np.zeros(shape=(data.shape[0], 9, data.shape[2]*data.shape[3]), dtype=np.float32)
                for k, x in enumerate(data):
                    if k % 1000 == 0:
                        print("Generating Puzzle Object Oriented DataSet From Puzzle Image...... {}".format(k))
                    for i in range(3):
                        for j in range(3):
                            img = np.zeros(shape=x[0].shape)
                            img[i*BASE_SIZE:(i+1)*BASE_SIZE, j*BASE_SIZE:(j+1)*BASE_SIZE] =\
                                x[0, i*BASE_SIZE:(i+1)*BASE_SIZE, j*BASE_SIZE:(j+1)*BASE_SIZE]
                            self.data[k, i*3+j] = img.flatten()
                np.save(PUZZLE_FILE_FO, self.data)
            else:
                self.data = np.load(PUZZLE_FILE_FO)

            if is_train:
                self.data = self.data[:TRAIN_EXAMPLES]
            else:
                self.data = self.data[TRAIN_EXAMPLES:]

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return self.data.shape[0]


if __name__ == "__main__":
    dataloader = DataLoader(FoSaeDataSet(is_train=True), batch_size=32)
    x = dataloader.__iter__().__next__()
    print(x.size())
