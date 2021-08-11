#!/usr/bin/env python
import numpy as np
from image import cams
import scipy.io as sio
import hdf5storage
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os
import math
from scipy.io import loadmat
from ai import cs




def wrap2pi(a1, a2):
    return np.pi - abs(abs(a1 - a2) - np.pi)

def gaussmf(mu, sigma):
    y = np.zeros([1, 51])
    if sigma !=0:
        x= np.linspace(1, 51, 51, endpoint=True)
        y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return y

def gaussbbox(bbox, W, H):
    vx = gaussmf((bbox[0] + bbox[2] / 2) / W * 51, bbox[2] / (W) * 51 )
    vy = gaussmf((bbox[1] + bbox[3] / 2) / H * 51, bbox[3] / (H) * 51)
    return vx, vy


def genbbox(path, filename, std3d=0, stdface=0):
    Mc = np.array([0.0023, 0, 0]) # microphone center
    data = hdf5storage.loadmat(os.path.join(path,filename))
    Pos3D = data['Pos3D']

    I = data['I']

    noise = np.random.normal(0, 1, Pos3D.shape)*std3d
    Pos_new = Pos3D+noise

    # create BBOX
    C = cams.Model()
    bbox1, _, _, _, p1 = cams.gen_bboximg(Pos_new[:, :3], C)
    bbox2, _, _, _, p2 = cams.gen_bboximg(Pos_new[:, 3:], C)
    bbox2[np.where(I==1), :]= np.zeros((1,4))
    p2[np.where(I==1), :] = False

    # add image noise
    bbox1[np.where(p1)[0],:] =bbox1[np.where(p1)[0],:]+np.random.normal(0, 1, bbox1[np.where(p1)[0],:].shape)*stdface
    bbox2[np.where(p2)[0],:] =bbox2[np.where(p2)[0],:]+np.random.normal(0, 1, bbox2[np.where(p2)[0],:].shape)*stdface

    # compute error
    DR=(np.sum(p1 & (I==1))+ np.sum(p2 & (I == 2)))/np.sum(I) # detection rate

    # az: gt
    Pos3D = Pos3D - np.repeat(Mc.reshape(1,3), 2, axis=0).reshape(1, 6) # mic coordinates
    _, gt1, _ = cs.cart2sp(Pos3D[:, 0], Pos3D[:, 1], Pos3D[:, 2])
    _, gt2, _ = cs.cart2sp(Pos3D[:, 3], Pos3D[:, 4], Pos3D[:, 5])

    # az: estimate
    Pos_new = Pos_new - np.repeat(Mc.reshape(1,3), 2, axis=0).reshape(1, 6) # mic coordinates
    _, th1, _ = cs.cart2sp(Pos_new[:, 0], Pos_new[:, 1], Pos_new[:, 2])
    _, th2, _ = cs.cart2sp(Pos_new[:, 3], Pos_new[:, 4], Pos_new[:, 5])

    T =len(th1)
    er1 = abs(np.rad2deg(wrap2pi(th1, gt1))).reshape(T,1)[ (p1 == True) & (I >=1) ] # wrap the error to pi
    er2 = abs(np.rad2deg(wrap2pi(th2, gt2))).reshape(T,1)[ (p2 == True) & (I ==2) ]

    MAE = (np.sum(er1)+np.sum(er2))/(len(er1)+len(er2)) # MAE
    ACC = (np.sum(er1 <= 5) + np.sum(er2 <= 5)) / (len(er1) + len(er2))

    print('std=%.3f, DR=%.3f, MAE=%.3f, ACC=%.2f'%(std3d, DR, MAE, ACC))

    # # create video feature
    W = C.Img_size[0]
    H = C.Img_size[1]
    Vx = np.zeros((T, 51))  # x feature
    Vy = np.zeros((T, 51))  # % y feature

    for t in range(len(Pos3D)):
        vx1, vx2, vy1, vy2 = np.zeros([1, 51]), np.zeros([1, 51]), np.zeros([1, 51]), np.zeros([1, 51])

        if I[t] >= 1 and p1[t]:
            vx1, vy1 = gaussbbox(bbox1[t], W, H)
        if I[t] == 2 and p2[t]:
            vx2, vy2 = gaussbbox(bbox2[t], W, H)

        Vx[t, :] = np.maximum(vx1, vx2)
        Vy[t, :] = np.maximum(vy1, vy2)

        # # display
        # plt.plot(Vx[t])
        # plt.plot(Vy[t])
        # plt.title(t)
        # plt.pause(0.01)
        # plt.cla()

    # save the results
    er = {'MAE': MAE, 'ACC': ACC, 'DR': DR}
    info = {'camData': C, 'std3d': std3d, 'Pos_new': Pos_new}
    savefile=os.path.join(path,'facedata', 'SSLRcam_'+ filename[:-9] +'face_3Dstd' + str(std3d) + 'facestd'+str(stdface)+'.mat')
    sio.savemat(savefile,{'Vx': Vx, 'Vy': Vy, 'I': I, 'er': er, 'info': info, 'bbox1':bbox1, 'bbox2':bbox2})
    print(savefile)


path = '/home/qian/Documents/data/sslr/'

#path = '/Users/xinyuan/Documents/NUS/Research/Data/sslrdata/'
Files=['qianspeech_Testing1_pos3d.mat', 'qianspeech_Testing2_pos3d.mat', 'qianspeech_Training_pos3d.mat',
       'qianfaceall_Testing1_pos3d.mat', 'qianfaceall_Testing2_pos3d.mat', 'qianfaceall_Training_pos3d.mat']
Files=['qianfaceall_Testing1_pos3d.mat', 'qianfaceall_Testing2_pos3d.mat', 'qianfaceall_Training_pos3d.mat']
std3d = 0
stdface=5

for idx in range(len(Files)):
    genbbox(path, Files[idx], std3d,stdface)