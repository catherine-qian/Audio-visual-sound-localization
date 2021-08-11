from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import logging
import torch.nn as nn
import math


class Unbuffered:
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)  # Write the data of stdout here to a text file as well

    def flush(self):
        pass


def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


def ACC(MAE, th):
    MAE=np.array(MAE)
    acc=np.sum(MAE<=th) / len(MAE)
    accmae=np.mean(MAE[MAE<=th])
    return acc, accmae


def minmax_scaletensor(x):
    xmin = x.min(1, keepdim=True)[0]
    xmax = x.max(1, keepdim=True)[0]
    xnorm = (x - xmin) / (xmax - xmin)
    return xnorm


def zscale(x, axis=1):  # Z normalization
    xmean = x.mean(axis, keepdim=True)
    xvar = x.var(axis, keepdim=True)
    xnorm = (x - xmean) / torch.sqrt(xvar)
    return xnorm


def minmax_norm2d(data_in, faxis):

    dmin = data_in.min(axis=faxis, keepdims=True)
    dmax = data_in.max(axis=faxis, keepdims=True)
    data_out = ((data_in - dmin) / (dmax - dmin))
    return data_out

def normalization(X):
    _range = np.max(X)-np.min(X)
    return (X-np.min(X))/_range

def standardization(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    return (X-mu)/sigma

def faceframes(X):
    frV1=np.max(X[:,306:],axis=1)>1/51 # with face detection
    frV0=~frV1 #without face detection

    frV1=np.where(frV1)[0]
    frV0=np.where(frV0)[0]
    return frV1, frV0

def torch_minmaxscale(X,dim=1):
    max_input, _ = torch.max(X,dim=dim,keepdim=True)
    min_input, _ = torch.min(X, dim=dim, keepdim=True)
    return (X-min_input)/(max_input -  min_input)


def MAEeval_face(Y_pred_t, Yte, Ite,Xte):
    # ------------ error evaluate   ----------
    frV1=np.max(Xte[:,306:],axis=1)>1/51 # with face detection
    frV0=~frV1 #without face detection
    er=np.zeros([Ite.shape[0],1])
    DoA = []
    for i in range(Ite.shape[0]):  # time
        hyp = Y_pred_t[i]  # our estimate
        y_1 = Yte[i]  # GT DoA label

        if Ite[i] == 1:  # single source
            gt = np.where(y_1 == 1)[0]  # ground truth
            pred = np.argmax(hyp)  # predict
            ang = angular_distance_compute(gt, pred)[0]
            er[i]=ang
        if Ite[i] == 2:  # 2 sources
            gt = np.where(y_1 == 1)[0]  # ground truth
            pred = np.argmax(hyp)  # predict, dominant speaker
            hyp2 = np.roll(hyp, 180 - pred)
            hyp2[180 - 15:180 + 15] = 0  # don't consider 25 degs
            pred = np.append(pred, np.argmax(np.roll(hyp2, pred - 180)))

            error = angular_distance_compute(gt.reshape(2, 1), pred)
            if (error[0, 0] + error[1, 1]) <= (error[0, 1] + error[1, 0]):
                er[i]=(error[0, 0]+error[1, 1])/2
            else:
                er[i]=(error[0, 1]+error[1, 0])/2

        DoA.append(pred)

    erV1 = np.sum(er[frV1]) / sum(frV1)
    erV0 = np.sum(er[frV0]) / sum(frV0)
    print("Testing MAE:\t face: %2.2f \t no face: %2.2f" % (erV1, erV0))

    frV1=np.where(frV1)[0]
    frV0=np.where(frV0)[0]
    return erV1, erV0,er, frV1, frV0

def MAEeval(Y_pred_t, Yte, Ite):
    # ------------ error evaluate   ----------
    erI1, erI2 = [], []

    DoA = []
    for i in range(Ite.shape[0]):  # time
        hyp = Y_pred_t[i]  # our estimate
        y_1 = Yte[i]  # GT DoA label

        if Ite[i] == 1:  # single source
            gt = np.where(y_1 == 1)[0]  # ground truth
            pred = np.argmax(hyp)  # predict
            ang = angular_distance_compute(gt, pred)[0]
            erI1.append(ang)
        if Ite[i] == 2:  # 2 sources
            gt = np.where(y_1 == 1)[0]  # ground truth
            pred = np.argmax(hyp)  # predict, dominant speaker
            hyp2 = np.roll(hyp, 180 - pred)
            hyp2[180 - 15:180 + 15] = 0  # don't consider 25 degs
            pred = np.append(pred, np.argmax(np.roll(hyp2, pred - 180)))

            error = angular_distance_compute(gt.reshape(2, 1), pred)
            if (error[0, 0] + error[1, 1]) <= (error[0, 1] + error[1, 0]):
                erI2.append(error[0, 0])
                erI2.append(error[1, 1])
            else:
                erI2.append(error[0, 1])
                erI2.append(error[1, 0])

        DoA.append(pred)

    MAE1, MAE2 = sum(erI1) / len(erI1), sum(erI2) / len(erI2)
    ACC1, ACC1mae = ACC(erI1, 5)
    ACC2, ACC2mae = ACC(erI2, 5)
    print("Testing MAE:\t MAE1: %.8f \t MAE2: %.8f" % (MAE1, MAE2))
    return MAE1, ACC1, MAE2, ACC2, erI1, erI2, ACC1mae, ACC2mae


# load the data
class MyDataloaderClass(Dataset):
    def __init__(self, X_data, Y_data):  # X_data is feature, Y_data is label (Gaussian or binary)
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class MyDataloaderClassTest(Dataset):
    def __init__(self, X_data):  # X_data is feature, Y_data is label (Gaussian or binary)
        self.x_data = X_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


def display(modelname, ep, EP, MAEl1, MAEl2, ACCl1, ACCl2, MAEh1, MAEh2, ACCh1, ACCh2, Ih=[1], Il=[1]):
    # display the results
    epmin = []
    ACCl1, ACCl2 , ACCh1, ACCh2 =ACCl1*100, ACCl2*100 , ACCh1*100, ACCh2*100

    if Il.__len__()==1 and Ih.__len__()==1:
        Il1, Il2, Ih1, Ih2 = 178 / 207, 29 / 207, 788 / 929, 141 / 929
        hlratio, lhratio = 929/207929, 207000/207929
    else:
        Il1, Il2 = np.sum(Il==1)/Il.__len__(), np.sum(Il==2)/Il.__len__()
        Ih1, Ih2 = np.sum(Ih==1)/Ih.__len__(), np.sum(Ih==2)/Ih.__len__()

    hlratio, lhratio = Ih.__len__()/(Il.__len__()+Ih.__len__()), Il.__len__()/(Il.__len__()+Ih.__len__())
    # --------- display the result -----------
    # torch.save(model, os.path.join('..','model',modelname))
    mael, accl = MAEl1[ep] * Il1 + MAEl2[ep] * Il2, ACCl1[ep] * Il1 + ACCl2[ep] * Il2
    maeh, acch = MAEh1[ep] * Ih1 + MAEh2[ep] * Ih2, ACCh1[ep] * Ih1 + ACCh2[ep] * Ih2
    maeall, accall = mael * lhratio + maeh * hlratio, accl * lhratio + acch * hlratio
    print("Testing-model-%s: ep=%d MAE(loudspk)=%.2f(%.1f), %.2f(%.1f), MAE(human)=%.2f(%.1f), %.2f(%.1f)| %.2f(%.1f), %.2f(%.1f)| %.2f(%.1f)"% (
    modelname, ep, MAEl1[ep], ACCl1[ep], MAEl2[ep], ACCl2[ep], MAEh1[ep], ACCh1[ep], MAEh2[ep], ACCh2[ep], \
    mael, accl, maeh, acch, maeall, accall))

    ## --------  Overall display --------

    if ep == EP - 1:  # last epoch
        allmael = np.mean(MAEl1) * Il1 + np.mean(MAEl2) * Il2
        allaccl = np.mean(ACCl1) * Il1 + np.mean(ACCl2) * Il2
        allmaeh = np.mean(MAEh1) * Ih1 + np.mean(MAEh2) * Ih2
        allacch = np.mean(ACCh1) * Ih1 + np.mean(ACCh2) * Ih2
        allmae, allacc = allmael * lhratio+ allmaeh * hlratio, allaccl * lhratio + allacch * hlratio
        print("Overall Testing-model-%s MAE(loudspk)=%.2f(%.1f), %.2f(%.1f), MAE(human)=%.2f(%.1f), %.2f(%.1f)| %.2f(%.1f), %.2f(%.1f)| %.2f(%.1f)"
              % (modelname, np.mean(MAEl1), np.mean(ACCl1), np.mean(MAEl2), np.mean(ACCl2), np.mean(MAEh1),
               np.mean(ACCh1), np.mean(MAEh2), np.mean(ACCh2), allmael, allaccl, allmaeh, allacch, allmae, allacc))

        mael, accl, maeh, acch, maeall, accall = np.zeros(EP), np.zeros(EP), np.zeros(EP), np.zeros(EP), np.zeros(
            EP), np.zeros(EP)
        for ep in range(0, EP):
            mael[ep], accl[ep] = MAEl1[ep] *  Il1 + MAEl2[ep] * Il2, ACCl1[ep] *  Il1 + ACCl2[
                ep] * Il2
            maeh[ep], acch[ep] = MAEh1[ep] * Ih1 + MAEh2[ep] * Ih2, ACCh1[ep] * Ih1 + ACCh2[
                ep] * Ih2
            maeall[ep], accall[ep] = mael[ep] * lhratio + maeh[ep] * hlratio, accl[ep] * lhratio + acch[ep] * hlratio
            print(
                "ep=%d %s MAE(loudspk)=%.2f(%.1f), %.2f(%.1f), MAE(human)=%.2f(%.1f), %.2f(%.1f)| %.2f(%.1f), %.2f(%.1f)| %.2f(%.1f)"
                % (ep, modelname, MAEl1[ep], ACCl1[ep], MAEl2[ep], ACCl2[ep], MAEh1[ep], ACCh1[ep], MAEh2[ep], ACCh2[ep], \
                mael[ep], accl[ep], maeh[ep], acch[ep], maeall[ep], accall[ep]))

        # display the minimum value
        epmin = np.argmin(maeall)
        print("Min: %s MAE(loudspk)=%.2f(%.1f), %.2f(%.1f), MAE(human)=%.2f(%.1f), %.2f(%.1f)| %.2f(%.1f), %.2f(%.1f)| %.2f(%.1f)"
              %(modelname, MAEl1[epmin], ACCl1[epmin], MAEl2[epmin], ACCl2[epmin], MAEh1[epmin], ACCh1[epmin],
                 MAEh2[epmin], ACCh2[epmin], \
                 mael[epmin], accl[epmin], maeh[epmin], acch[epmin], maeall[epmin], accall[epmin]))
    return maeall,  accall, epmin


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger
