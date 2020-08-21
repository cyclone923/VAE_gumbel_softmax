import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE, BASE_SIZE
from torch.utils.data import Dataset, DataLoader
import sys
import os

TRAIN_EXAMPLES = 90000
# TEST_EXAMPLES = 10000

np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=sys.maxsize)
np.random.seed(0)

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

def get_view_dataset(data, n):
    idx = np.random.choice(len(data), n, replace=False)
    new_arr = np.array(data[idx])
    return SimpleDataSet(new_arr)

def load_data(fo_logic):
    PUZZLE_FILE_FO = "puzzle/puzzle_data/puzzles_fo.npy"
    if not fo_logic:
        data = np.load(PUZZLE_FILE)
    else:
        if not os.path.isfile(PUZZLE_FILE_FO):
            data_img = np.load(PUZZLE_FILE)
            data = np.zeros(shape=(data_img.shape[0], 9, data_img.shape[2] * data_img.shape[3]), dtype=np.float32)
            for k, x in enumerate(data_img):
                if k % 1000 == 0:
                    print("Generating Puzzle Object Oriented DataSet From Puzzle Image ...... {}".format(k))
                for i in range(3):
                    for j in range(3):
                        img = np.zeros(shape=x[0].shape)
                        img[i * BASE_SIZE:(i + 1) * BASE_SIZE, j * BASE_SIZE:(j + 1) * BASE_SIZE] = \
                            x[0, i * BASE_SIZE:(i + 1) * BASE_SIZE, j * BASE_SIZE:(j + 1) * BASE_SIZE]
                        data[k, i * 3 + j] = img.flatten()
            np.save(PUZZLE_FILE_FO, data)
        else:
            data = np.load(PUZZLE_FILE_FO)

    return data



if __name__ == "__main__":
    dataloader, _ = DataLoader(SimpleDataSet(np.load(PUZZLE_FILE)), batch_size=32)
    x = dataloader.__iter__().__next__()
    print(x.size())
