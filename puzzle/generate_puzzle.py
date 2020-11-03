from torchvision import datasets
import numpy as np
from collections import defaultdict
from skimage.transform import resize
from skimage.exposure import equalize_hist
import itertools
import matplotlib.pyplot as plt

PUZZLE_FILE = "puzzle/puzzle_data/puzzles.npy"
SUCCESOR_FILE = "puzzle/puzzle_data/successor.npy"
# PUZZLE_FILE_SPC = "puzzle/puzzle_data/puzzles_spc.npy"
BASE_SIZE = 14
MAX_SUCCESSOR = 4

np.random.seed(0)

def generate_bases():
    def normalize(image):
        # into 0-1 range
        if image.max() == image.min():
            return image - image.min()
        else:
            return (image - image.min()) / (image.max() - image.min())

    def equalize(image):
        return equalize_hist(image)

    def enhance(image):
        return np.clip((image - 0.5) * 3, -0.5, 0.5) + 0.5

    def preprocess(image):
        image = image.astype(float)
        image = resize(image, (BASE_SIZE, BASE_SIZE))
        image = equalize(image)
        image = normalize(image)
        image = enhance(image)
        return image

    mnist_data = datasets.MNIST('./data', train=False)
    ten_digit = defaultdict(lambda : [])

    for data, target in zip(mnist_data.data, mnist_data.targets):
        target = target.item()
        if target == 9:
            continue
        if len(ten_digit[target]) < 10:
            data = preprocess(data.numpy())
            ten_digit[target].append(data)

    one_digit = {
        0: ten_digit[0][1], 1: ten_digit[1][0], 2: ten_digit[2][0], 3: ten_digit[3][4],
        4: ten_digit[4][3], 5: ten_digit[5][1], 6: ten_digit[6][1], 7: ten_digit[7][3],
        8: ten_digit[8][1]
    }

    # plt.cla()
    # for k in sorted(one_digit.keys()):
    #     plt.imshow(one_digit[k], cmap='gray')
    #     plt.pause(0.1)

    return one_digit

def to_image_puzzle(puzzles, base):
    imgage_puzzles = []
    for p in puzzles:
        image = np.zeros(shape=(1, BASE_SIZE * 3, BASE_SIZE * 3), dtype=np.float32)
        for i, base_row in enumerate(p):
            s_i = i * BASE_SIZE
            e_i = i * BASE_SIZE + BASE_SIZE
            for j, base_ind in enumerate(base_row):
                s_j = j * BASE_SIZE
                e_j = j * BASE_SIZE + BASE_SIZE
                image[:, s_i:e_i, s_j:e_j] = base[base_ind]
        imgage_puzzles.append(image)
    return np.stack(imgage_puzzles)

def valid_swap_index(x, y, n_row, n_col):
    return not x < 0 and not y < 0 and x < n_row and y < n_col

def get_single_successor_for_single_puzzle(p, n_row, n_col):
    x, y = np.where(p == 0)
    for i in [-1, 1]:
        for swap_x, swap_y in [(x+i, y), (x, y+i)]:
            if valid_swap_index(swap_x, swap_y, n_row, n_col):
                successor = np.array(p)
                temp = successor[swap_x, swap_y]
                successor[swap_x, swap_y] = 0
                successor[x, y] = temp
                yield successor
            else:
                yield None

np_to_tp = lambda x: tuple(map(tuple, x))

def hash_puzzle_to_index(puzzles):
    index_lookup = {}
    for i, p in enumerate(puzzles):
        index_lookup[np_to_tp(p)] = i
    return index_lookup


def get_successor_index(puzzles):
    # Up, Left, Down, Right, -1 stands for invalid successor
    successor_index = - np.ones(shape=(puzzles.shape[0], MAX_SUCCESSOR), dtype=np.int)
    n_puzzles, n_row, n_col = puzzles.shape
    look_up = hash_puzzle_to_index(puzzles)
    for i, p in enumerate(puzzles):
        if i % 10000 == 0:
            print(i)
        for j, one_successor in enumerate(get_single_successor_for_single_puzzle(p, n_row, n_col)):
            if one_successor is not None:
                successor_index[i, j] = look_up[np_to_tp(one_successor)]
    return successor_index


def generate_puzzles(base):
    puzzles = []
    for n, p in enumerate(itertools.permutations([0,1,2,3,4,5,6,7,8])):
        if n % 10000 == 0:
            print(n)
        p = np.resize(p, (3,3))
        puzzles.append(p)
    puzzles = np.stack(puzzles)
    np.random.shuffle(puzzles)

    successor_index = get_successor_index(puzzles)
    images = to_image_puzzle(puzzles, base)
    print(successor_index.shape, images.shape)

    np.save(PUZZLE_FILE, images)
    np.save(SUCCESOR_FILE, successor_index)

# def generate_puzzles_specific(base, ps):
#     imgs = []
#     for n, p in enumerate(ps):
#         puzzle = np.zeros(shape=(1, BASE_SIZE*3, BASE_SIZE*3), dtype=np.float32)
#         for i, base_row in enumerate(p):
#             s_i = i * BASE_SIZE
#             e_i = i * BASE_SIZE + BASE_SIZE
#             for j, base_ind in enumerate(base_row):
#                 s_j = j*BASE_SIZE
#                 e_j = j*BASE_SIZE + BASE_SIZE
#                 puzzle[:, s_i:e_i, s_j:e_j] = base[base_ind]
#         imgs.append(puzzle)
#     imgs = np.stack(imgs)
#     print(imgs.shape)
#     np.random.shuffle(imgs)
#
#     data_img = imgs
#     data = np.zeros(shape=(data_img.shape[0], 9, data_img.shape[2] * data_img.shape[3]), dtype=np.float32)
#     for k, x in enumerate(data_img):
#         print("Generating Puzzle Object Oriented DataSet From Puzzle Image ...... {}".format(k))
#         for i in range(3):
#             for j in range(3):
#                 img = np.zeros(shape=x[0].shape)
#                 img[i * BASE_SIZE:(i + 1) * BASE_SIZE, j * BASE_SIZE:(j + 1) * BASE_SIZE] = \
#                     x[0, i * BASE_SIZE:(i + 1) * BASE_SIZE, j * BASE_SIZE:(j + 1) * BASE_SIZE]
#                 data[k, i * 3 + j] = img.flatten()
#     np.save(PUZZLE_FILE_SPC, data)


if __name__ == "__main__":
    base = generate_bases()
    generate_puzzles(base)
    # ps = np.array(
    #     [
    #         [
    #             [1,7,3],
    #             [2,0,6],
    #             [5,4,8]
    #         ],
    #         [
    #             [1,7,3],
    #             [2,0,8],
    #             [5,4,6]
    #         ],
    #         [
    #             [1,7,3],
    #             [2,6,0],
    #             [5,4,8]
    #         ],
    #         [
    #             [1,7,3],
    #             [2,6,8],
    #             [5,4,0]
    #         ],
    #         [
    #             [1,7,3],
    #             [2,8,0],
    #             [5,4,6]
    #         ],
    #         [
    #             [1,7,3],
    #             [2,8,6],
    #             [5,4,0]
    #         ]
    #     ]
    # )
    # generate_puzzles_specific(base, ps)






