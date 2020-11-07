import numpy as np
from puzzle.generate_puzzle import PUZZLE_FILE, SUCCESOR_FILE, MAX_SUCCESSOR
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import matplotlib.pyplot as plt

TRAIN_EXAMPLES = 36000
VALIDATION_EXAMPLES = 4000
TEST_EXAMPLES = 200


np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=sys.maxsize)
np.random.seed(0)

class SimpleDataSet(Dataset):
    def __init__(self, data, successor_index):
        self.data = data
        self.successor_index = successor_index

        padding = np.cumsum(np.sum(self.successor_index >= 0, axis=1))
        self.data_set_length = padding[-1]
        self.fast_mapping = {}

        i = 0
        remain_item = 0
        for item in range(self.__len__()):
            if item >= padding[i]:
                i += 1
                remain_item = 0
            self.fast_mapping[item] = (i, remain_item)
            remain_item += 1

    def __getitem__(self, item):
        i, remain_item = self.fast_mapping[item]
        j = 0
        index_buffer = self.successor_index[i]
        while remain_item != 0 or index_buffer[j] == -1:
            if not index_buffer[j] == -1:
                remain_item -= 1
            j += 1
        successor = self.successor_index[i, j]
        return self.data[i], self.data[successor]


    def __len__(self):
        return self.data_set_length


def get_train_and_test_dataset(data, successors):
    print("Loaded Data Size {}".format(data.shape))
    data_set = SimpleDataSet(data, successors)
    return random_split(
        data_set, [TRAIN_EXAMPLES, VALIDATION_EXAMPLES, TEST_EXAMPLES,
        len(data_set) - TRAIN_EXAMPLES - VALIDATION_EXAMPLES - TEST_EXAMPLES]
    )

def load_data():
    return np.load(PUZZLE_FILE), np.load(SUCCESOR_FILE)



if __name__ == "__main__":
    dataloader = DataLoader(SimpleDataSet(np.load(PUZZLE_FILE), np.load(SUCCESOR_FILE)), batch_size=16, shuffle=True)
    all_x_pre, all_x_suc = dataloader.__iter__().__next__()
    plt.gca()
    for i, (x_pre, x_suc) in enumerate(zip(all_x_pre, all_x_suc)):
        print(i)
        x_pre = x_pre[0]
        x_suc = x_suc[0]
        plt.imshow(x_pre)
        plt.pause(0.1)
        plt.imshow(x_suc)
        plt.pause(0.1)
