import hdf5storage
import os
import socket
import funcs
from torch.utils.data import DataLoader
import numpy as np
import loaddata
import matplotlib.pyplot as plt



def dataread(BATCH_SIZE,args):
    print("prepare the data! ")
    if socket.gethostname() == "x99ews1":  # server
        path = '/home/qian/Documents/data/sslr/'
    else:
        path = '/Users/xinyuan/Documents/NUS/Research/Data/sslrdata/'
    print(path)
    ###########################################################################################

    Atr, Ate1, Ate2, apath, fnorm, faxis = loaddata.audioname(args)
    Vtr, te1, te2, trname, tename = loaddata.facename(args)
    """
    Testing- audio
    """
    test1 = hdf5storage.loadmat(os.path.join(apath, Ate1))  # human
    Xte1, Yte1, Ite1 = test1['Xte1'],test1['Yte1'], test1['Ite1']

    test2 = hdf5storage.loadmat(os.path.join(apath, Ate2))  # load speaker
    Xte2, Yte2, Ite2 = test2['Xte2'], test2['Yte2'], test2['Ite2']

    # normalization
    Xte1, Xte2 = fnorm(Xte1, axis=faxis), fnorm(Xte2, axis=faxis)
 
    """
    Testing - face
    """
    if Vtr != []: # with face data
        facete1 = hdf5storage.loadmat(os.path.join(path, "facedata", te1))
        facete2 = hdf5storage.loadmat(os.path.join(path, "facedata", te2))

        if args.Vy: # if use video vertical information
            Xte1 = np.concatenate([Xte1, facete1[trname[0]], facete1[trname[1]]], axis=1)
            Xte2 = np.concatenate([Xte2, facete2[trname[0]], facete2[trname[1]]], axis=1)
        else:
            Xte1 = np.concatenate([Xte1, facete1[trname[0]]], axis=1)
            Xte2 = np.concatenate([Xte2, facete2[trname[0]]], axis=1)

        # change to normal distribution
        Xte1[~(np.sum(Xte1[:, 306:], axis=1) > 0), 306:] = 1/51
        Xte2[~(np.sum(Xte2[:, 306:], axis=1) > 0), 306:] = 1/51

    """
    Training
    """
    if args.train:  # need training set
        # audio file

        if args.trA == 'gccsnrall': # load all SNR files
            train = hdf5storage.loadmat(os.path.join(apath, 'qianTraining_GCCPHAT_SSLR_SNR-20dB.mat'))
            Xtr, Ytr, Itr, Ztr = train['Xtr'], train['Ytr'], train['Itr'], train['Ztr']
            for snr in [-10,0,10,20]:
                train = hdf5storage.loadmat(os.path.join(apath, 'qianTraining_GCCPHAT_SSLR_SNR'+str(snr)+'dB.mat'))
                Xtr=np.concatenate((Xtr,train['Xtr']), axis=0)
                Ytr = np.concatenate((Ytr, train['Ytr']), axis=0)
                Itr = np.concatenate((Itr, train['Itr']), axis=0)
                Ztr = np.concatenate((Ztr, train['Ztr']), axis=0)
        else:
            train = hdf5storage.loadmat(os.path.join(apath, Atr))
            Xtr, Ytr, Itr, Ztr = train['Xtr'], train['Ytr'], train['Itr'], train['Ztr']
        Xtr=fnorm(Xtr, axis=faxis)


        # include face file face file
        if Vtr != []:
            facetr = hdf5storage.loadmat(os.path.join(path, "facedata", Vtr))
            Trpt=Xtr.shape[0]/facetr[trname[0]].shape[0] # repeat face detection with train

            if args.Vy:  # if use video vertical information
                Xtr = np.concatenate([Xtr, facetr[trname[0]].repeat(Trpt, axis=0), facetr[trname[1]].repeat(Trpt, axis=0)], axis=1)
            else:
                Xtr = np.concatenate([Xtr, facetr[trname[0]].repeat(Trpt, axis=0)], axis=1)

            Xtr[~(np.sum(Xtr[:, 306:], axis=1) > 0), 306:] = 1 / 51

    else:  # no training data
        train_loader = []
        Xtr, Ytr, Itr, Ztr = [], [], [], []
    print('finish data preparation')


    if Vtr and args.VO:  # train/test: with only face information
        # face flag
        Ftr = np.max(Xtr[:, 306:], axis=1) > 1 / 51  # with face detection
        Fte1 = np.max(Xte1[:, 306:], axis=1) > 1 / 51  # with face detection
        Fte2 = np.max(Xte2[:, 306:], axis=1) > 1 / 51  # with face detection

        Xtr, Ytr, Itr = Xtr[Ftr, :], Ytr[Ftr, :], Itr[Ftr, :]
        Xte1, Yte1, Ite1 = Xte1[Fte1,:], Yte1[Fte1,:], Ite1[Fte1,:]
        Xte2, Yte2, Ite2 = Xte2[Fte2,:], Yte2[Fte2,:], Ite2[Fte2,:]

    if args.train:
        train_loader_obj = funcs.MyDataloaderClass(Xtr, Ztr)  # Xtr-data feature, Ztr-Gaussian-format label
        train_loader = DataLoader(dataset=train_loader_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


    return train_loader,  Xtr, Ytr, Itr, Ztr, Xte1, Yte1, Ite1, Xte2, Yte2, Ite2

