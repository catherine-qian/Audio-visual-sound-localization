
import numpy as np
import cv2
import math
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os

class Model():
     # intriniscs:
    # - camera matrix
    def __init__(self):
        self._CAM_MATRIX = np.array([
            [592.779158393, 0.0, 328.225295327],
            [0.0, 591.991356581, 234.241795451],
            [0.0, 0.0, 1.0],
        ])
        # - distortion coefficients
        self._CAM_DIST = np.array([
            0.121716106149, -0.481610730654, -0.00144629247554, 0.00131082543473,
            0.477385359915
        ])
        # extrinsics:
        # - rotation:
        self._CAM_R = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ])
        # - translation:
        self._CAM_T = np.array([0.0, 0.163 - 0.2066, -0.087])
        self.Img_size = [640, 480]


def gen_bbox3d(obj_3d, W, H):
    # generate top left and bottom right points
    L = len(obj_3d)
    # bottom right, top left

    yx = obj_3d[:, 1] / obj_3d[:, 0]
    th = np.arctan(yx) - math.pi / 2
    vec_rb = np.concatenate( (W / 2 * np.cos(th).reshape(L, 1), W / 2 * np.sin(th).reshape(L, 1), -np.ones((L, 1)) * H / 2), axis=1)  # right bottom
    vec_tl = np.concatenate(  (-W / 2 * np.cos(th).reshape(L, 1), -W / 2 * np.sin(th).reshape(L, 1), np.ones((L, 1)) * H/2), axis=1)  # top left
    Prb = obj_3d + vec_rb
    Ptl = obj_3d + vec_tl
    P3d = obj_3d + np.zeros(Prb.shape)
    return Prb, Ptl, P3d

def proj3DtoImg(P3d, C):
    Pimg, _ = cv2.projectPoints(P3d, C._CAM_R, C._CAM_T, C._CAM_MATRIX, C._CAM_DIST)
    Pimg = np.squeeze(Pimg)
    val = (P3d[:, 0] >= 0) & (Pimg[:, 0] >= 0) & (Pimg[:, 0] <= C.Img_size[0]) & (Pimg[:, 1] >= 0) & (Pimg[:, 1] <= C.Img_size[1]) # check (1) x>0;  (2) projection on image
    return Pimg, val

def gen_bboximg(obj_3d, C):
    # obj_3d should be L by 3 matrix
    W, H = 0.14, 0.18

    #Iimg, _ = proj3DtoImg(obj_3d, C)

    Prb, Ptl, P3d = gen_bbox3d(obj_3d, W, H)  # create bbox in 3D
    Irb, valr = proj3DtoImg(Prb, C)
    Itl, vall = proj3DtoImg(Ptl, C)
    Iimg, valc = proj3DtoImg(P3d, C)
     # image projection

    # create BBox on image
    bbox = np.zeros((obj_3d.shape[0], 4), dtype=float)
    f1 = [i for i, v in enumerate(vall & valr) if v ]
    bbox[f1, :] = np.concatenate((Itl[f1, :], Irb[f1, :] - Itl[f1, :]), axis=1)

    # top left points are on image, but bottom right is not
    f2 = [i for i, v in enumerate(vall & ~valr) if v & (sum((Irb[i, :] + Itl[i, :])/2 <= C.Img_size) == 2)]
    bbox[f2, :] = np.concatenate((Itl[f2, :], np.minimum(C.Img_size - Itl[f2, :], Irb[f2, :] - Itl[f2, :])), axis=1)

    val = (np.sum(bbox, axis=1) > 0).reshape(len(bbox), 1) # bbox validated
    return bbox, Irb, Itl, Iimg, val


# # generate BBOX
#
# # object coordinates w.r.t the microphone array (center).
# # X, Y and Z axes indicate front, left and up, respectively.
# obj_3d = np.array([
#     [1.705, -0.245, 0.033],  # front
#     [1.613, 0.336, 0.047]
# ])
#
# C = Model()
#
# bbox, Irb, Itl, Iimg = gen_bboximg(obj_3d, C)  # L by 4 bbox



# # display
# import matplotlib.patches as patches
# path='/Volumes/LaCie/NUS/Dataset/SSLR/sslr/sslr/human/'
# #path = '/home/qian/Documents/dataset/sslr/human/'
#
# img = mpimg.imread(os.path.join(path, 'video_gt','s3_34', 'r000071.png'))
#
# plt.imshow(img)
# plt.plot(Irb[:, 0], Irb[:, 1], 'bo')
# plt.plot(Itl[:, 0], Itl[:, 1], 'go')
# plt.plot(Iimg[:, 0], Iimg[:, 1], 'ro')
#
# ax = plt.gca()
# for bix in range(bbox.shape[0]):
#     rect = patches.Rectangle((bbox[bix, 0], bbox[bix, 1]), bbox[bix, 2], bbox[bix, 3], linewidth=1, edgecolor='y', facecolor='none')
#     ax.add_patch(rect)

# def proj3DtoImg(P3d):
#     Img_size = [640, 480]
#     Pimg, _ = cv2.projectPoints(P3d, _CAM_R, _CAM_T, _CAM_MATRIX, _CAM_DIST)
#     Pimg = np.squeeze(Pimg)
#     val = (Pimg[:, 0]>=0) & (Pimg[:, 0]<= Img_size[0]) & (Pimg[:, 1]>=0) & (Pimg[:, 1]<=Img_size[1]) # check wither its on image
#     return Pimg, val
#
# p, val=proj3DtoImg(pr)