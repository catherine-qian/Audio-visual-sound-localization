#!/usr/bin/env python
"""
project_to_image.py

Example of projecting 3D location to the image.

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# intriniscs:
# - camera matrix
_CAM_MATRIX = np.array([
    [592.779158393, 0.0, 328.225295327],
    [0.0, 591.991356581, 234.241795451],
    [0.0, 0.0, 1.0],
])
# - distortion coefficients
_CAM_DIST = np.array([
    0.121716106149, -0.481610730654, -0.00144629247554, 0.00131082543473,
    0.477385359915
])

# extrinsics:
# - rotation:
_CAM_R = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])
# - translation:
_CAM_T = np.array([0.0, 0.163 - 0.2066, -0.087])


def qvfid(t, stamps):
    """
    Search for the video frame ID from a given time
    """
    if t <= stamps[0]:
        return 0
    elif t >= stamps[-1]:
        return len(stamps) - 1

    # binary search
    sid = 0
    eid = len(stamps)
    while eid - sid > 1:
        mid = (sid + eid) // 2
        if t >= stamps[mid]:
            sid = mid
        else:
            eid = mid
    assert eid == sid + 1
    return sid


# path = '/idiap/temp/whe/sslr/human'
path='/Volumes/LaCie/NUS/Dataset/SSLR/sslr/sslr/human/'

win_size = 8192
hop_size = 4096
fs = 48000
sid = 's2_14'
infile = os.path.join(path,'gt_frame', '%s.w8192_o4096_qianspeech.txt' % sid)

stamps = np.loadtxt(os.path.join(path, 'video_gt', sid, 'stamps'))

idx = 0
for line in open(infile, "r"):  # for each 3D point
    c = line.strip().split()
    obj_im, _ = cv2.projectPoints(
        np.asarray(c, dtype=np.float64)[1:4], _CAM_R, _CAM_T, _CAM_MATRIX,
        _CAM_DIST)
    obj_im = np.squeeze(obj_im)

    # display
    afid = int(c[0])  # audio frame ID
    t = (afid * hop_size + win_size / 2) / fs  # time
    vfid = qvfid(t, stamps)  # video frame ID

    img = mpimg.imread(os.path.join(path, 'video_gt', sid, 'p%06d.png' % vfid))

    plt.imshow(img)
    plt.plot(obj_im[0], obj_im[1], 'bo')
    plt.title(c[0])
    plt.pause(0.1)
    plt.clf()
