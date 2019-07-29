
"""gaomingda 6.29
generate train images from mxnet io format;
and save to My_Files/traindata
and label save to My_Files/train.txt
"""
import sys
sys.path.append("/home/gaomingda/insightface/recognition")
from image_iter import FaceImageIter

import cv2
import os
import mxnet as mx
import numpy as np

save_root = '/home/gaomingda/insightface/recognition/createLmdb/My_Files'

train_dataiter = FaceImageIter(
        batch_size=4,
        data_shape=(3, 112, 112),
        path_imgrec="/home/gaomingda/insightface/datasets/ms1m-retinaface-t1/train.rec",
        shuffle=True,
        rand_mirror=False,
        mean=None,
        cutoff=False,
        color_jittering=0,
        images_filter=0,
    )
data_nums = train_dataiter.num_samples()
train_dataiter.reset()
train_dataiter.is_init = True

f = open(os.path.join(save_root, "train.txt"), 'w')
f.truncate()

for i in range(data_nums):
    label, s, _, _ = train_dataiter.next_sample()
    img_ = mx.image.imdecode(s) #mx.ndarray
    img = np.array(img_.asnumpy(), dtype=np.uint8)
    img = img[:, :, ::-1]
    #print("img shape", img.shape)
    cv2.imwrite(save_root+'/traindata/{}.jpg'.format(i), img)

    f.writelines("{}.jpg".format(i) + " " + str(int(label)) + "\n")

    if i%1000==0:
        print(i)

f.close()
print("done")