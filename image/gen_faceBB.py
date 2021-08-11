#!/usr/bin/env python
import numpy as np
from image import cams
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# path = '/Volumes/LaCie/NUS/Dataset/SSLR/sslr/sslr/human/'
path = '/home/qian/Documents/dataset/sslr/human/'

C = cams.Model()

for f in os.listdir(os.path.join(path, 'audio')):
    if f.endswith('.wav'):
        file = os.path.join(path, 'gt_frame', f[:-4] + '.w8192_o4096_qianspeech.txt') # for each audio file
        #file = os.path.join(path, 'gt_frame', f[:-4] + '.w8192_o4096.txt') # for each audio file

        print(file)
        idx=0
        for line in open(file, "r"): # for each 3D point
            c=line.strip().split()
            if idx==0: # initalize
                L = len(open(file).readlines())
                data=np.zeros([L, len(c)])
            #print(c[0:1])
            data[idx, :]=np.asarray(c, dtype=np.float64).reshape(1, 7)
            idx = idx + 1
        # add noise in 3D
        # project 3D point to image
        obj_3d = data[:, 1:4]
        bbox, Irb, Itl, Iimg,_ = cams.gen_bboximg(obj_3d, C)  # L by 4 bbox

        # display
        for t in range(bbox.shape[0]):
            fr=data[t, 0]
            img = mpimg.imread(os.path.join(path, 'video_gt', f[:-4], 'p%06d'%fr+'.png'))
            plt.imshow(img)
            plt.plot(Irb[t, 0], Irb[t, 1], 'bo')
            plt.plot(Itl[t, 0], Itl[t, 1], 'go')
            plt.plot(Iimg[t, 0], Iimg[t, 1], 'ro')
            ax = plt.gca()
            rect = patches.Rectangle((bbox[t, 0], bbox[t, 1]), bbox[t, 2], bbox[t, 3], linewidth=1, edgecolor='y', facecolor='none')
            ax.add_patch(rect)
            plt.title(t)
            plt.pause(0.1)
            plt.clf()
        # print(f'{data[0,0]:05d}')

# display



plt.imshow(img)
plt.plot(Irb[:, 0], Irb[:, 1], 'bo')
plt.plot(Itl[:, 0], Itl[:, 1], 'go')
plt.plot(Iimg[:, 0], Iimg[:, 1], 'ro')

ax = plt.gca()
for bix in range(bbox.shape[0]):
    rect = patches.Rectangle((bbox[bix, 0], bbox[bix, 1]), bbox[bix, 2], bbox[bix, 3], linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(rect)