from torchvision import datasets
import numpy as np
from collections import defaultdict
from skimage.transform import resize
from skimage.exposure import equalize_hist
import itertools
import matplotlib.pyplot as plt

PUZZLE_FILE = "puzzle/puzzle_data/puzzles.npy"
BASE_SIZE = 14

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

def generate_puzzles(base):
    imgs = []
    for n, p in enumerate(itertools.permutations([0,1,2,3,4,5,6,7,8])):
        if n % 1000 == 0:
            print(n)
        p = np.resize(p, (3,3))
        puzzle = np.zeros(shape=(1, BASE_SIZE*3, BASE_SIZE*3), dtype=np.float32)
        for i, base_row in enumerate(p):
            s_i = i * BASE_SIZE
            e_i = i * BASE_SIZE + BASE_SIZE
            for j, base_ind in enumerate(base_row):
                s_j = j*BASE_SIZE
                e_j = j*BASE_SIZE + BASE_SIZE
                puzzle[:, s_i:e_i, s_j:e_j] = base[base_ind]
        imgs.append(puzzle)
    imgs = np.stack(imgs)
    print(imgs.shape)
    np.random.shuffle(imgs)
    np.save(PUZZLE_FILE, imgs)


if __name__ == "__main__":
    base = generate_bases()
    generate_puzzles(base)





