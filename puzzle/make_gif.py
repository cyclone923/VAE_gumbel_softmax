import imageio
import os
from puzzle.util import SAMPLE_DIR, ACTION_DIR, BACK_TO_LOGIT


def to_gif():
    for dir, out_name in zip([SAMPLE_DIR, ACTION_DIR], ["samples", "actions"]):
        images = []
        for filename in sorted(os.listdir(dir), key=lambda x: int(x.split(".")[0])):
            images.append(imageio.imread(os.path.join(dir, filename)))
        print(len(images))
        imageio.mimsave('{}-{}.gif'.format(out_name, "btl" if BACK_TO_LOGIT else "nv"), images)


if __name__ == '__main__':
    to_gif()
