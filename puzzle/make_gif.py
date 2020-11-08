import imageio
import os

dir1 = "image/samples"
dir2 = "image/actions"


for dir, out_name in zip([dir1, dir2], ["samples", "actions"]):
    images = []
    for filename in sorted(os.listdir(dir), key=lambda x: int(x.split(".")[0])):
        images.append(imageio.imread(os.path.join(dir, filename)))
    print(len(images))
    imageio.mimsave('{}.gif'.format(out_name), images)
