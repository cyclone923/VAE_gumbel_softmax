import imageio
import os
from puzzle.train import SAMPLE_DIR, ACTION_DIR


for dir, out_name in zip([SAMPLE_DIR, ACTION_DIR], ["samples", "actions"]):
    images = []
    for filename in sorted(os.listdir(dir), key=lambda x: int(x.split(".")[0])):
        images.append(imageio.imread(os.path.join(dir, filename)))
    print(len(images))
    imageio.mimsave('{}.gif'.format(out_name), images)
