# # Data preparation & model define
import torch
import funcs
from sklearn.preprocessing import minmax_scale
import modelclass
import importlib


def get_model(modelname):
    m = importlib.import_module("modelclass.{}".format(modelname))
    return m


def Dataextract(modelname, lossname):
    models = get_model(modelname)

    ## ---- loss function -----
    if lossname == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean')
    if lossname == 'CEL':
        criterion = torch.nn.CrossEntropyLoss(reduce=True)

    return models, criterion


def facename(args):
    # no face information
    if args.trV == 'none' or args.teV == 'none':
        tr, te1, te2 = [], [], []
        print('audio-only: no face')
        return tr, te1, te2," ", " "

    # training data
    trname = ['Vx', 'Vy']
    if args.trV == 'faceCAV3D':
        tr = "qianTraining_GCCPHAT_SSLRface_3Dstd0.2facestd0.mat"
    elif args.trV == 'faceSSLR':
        tr = "SSLRcam_qianfaceall_Training_face_3Dstd0.2facestd0.mat"
    elif 'faceSWAP' in args.trV:
        tr = "TrainingSSLRcam_faceSWAPpct" + str(args.trV[8:]) + ".mat"
        trname = ['VxSWAP', 'VySWAP']
    elif 'faceFP' in args.trV:
        tr = "TrainingSSLRcam_faceFPpct" + str(args.trV[6:]) + ".mat"
        trname = ['VxFP', 'VyFP']

    tename = ['Vx', 'Vy']
    # testing data
    if args.teV == 'faceCAV3D':
        te1, te2 = "qianTesting1_GCCPHAT_SSLRface_3Dstd0.2facestd0.mat", "qianTesting2_GCCPHAT_SSLRface_3Dstd0.2facestd0.mat"
    elif args.teV == 'faceSSLR':
        te1, te2 = "SSLRcam_qianfaceall_Testing1_face_3Dstd0.2facestd0.mat", "SSLRcam_qianfaceall_Testing2_face_3Dstd0.2facestd0.mat"
        # te1='human_retinaface.mat'
    elif 'faceSWAP' in args.teV:
        te1, te2 = "Testing1SSLRcam_faceSWAPpct" + str(args.trV[8:]) + ".mat", "Testing2SSLRcam_faceSWAPpct" + str(
            args.trV[8:]) + ".mat"
        tename = ['VxSWAP', 'VySWAP']
    elif 'faceFP' in args.teV:
        te1, te2 = "Testing1SSLRcam_faceFPpct" + str(args.trV[6:]) + ".mat", "Testing2SSLRcam_faceFPpct" + str(
            args.trV[6:]) + ".mat"
        tename = ['VxFP', 'VyFP']

    return tr, te1, te2, trname, tename


def audioname(args):
    # training data
    fnorm, faxis = minmax_scale, 1
    if args.trA == 'gcc' or args.trA == 'qiangcc':
        tr, apath = "qianTraining_GCCPHAT_SSLR.mat", '/home/qian/Documents/data/sslr/'
    elif "gccsnr" in args.trA:
        snr = args.trA[6:]  # snr in dB
        tr, apath = "qianTraining_GCCPHAT_SSLR_SNR" + snr + "dB.mat", '/home/qian/Documents/data/sslr/'
    elif "fan" in args.trA:
        tr, apath = "qianTraining_GCCPHAT_SSLR_SNR_" + args.trA + ".mat", '/home/qian/Documents/data/sslr/'
    elif args.trA == 'melgcc':
        tr, apath = "Training_melGCCPHAT_SSLR.mat", '/home/maulik/data/sslr'
        fnorm, faxis = funcs.minmax_norm2d, (1, 2)  # normalization function
    elif args.trA == 'stft': # STFT
        tr, apath = "Training_melGCCPHAT_SSLR.mat", '/home/maulik/data/sslr'

    # test data
    if args.teA == 'gcc' or args.teA == 'qiangcc':
        te1, te2, apath = "qianTesting1_GCCPHAT_SSLR.mat", "qianTesting2_GCCPHAT_SSLR.mat", '/home/qian/Documents/data/sslr/'
        fnorm, faxis = minmax_scale, 1
    elif "gccsnr" in args.teA:
        snr = args.teA[6:]  # snr in dB
        te1, te2, apath = "qianTesting1_GCCPHAT_SSLR_SNR" + snr + "dB.mat", "qianTesting2_GCCPHAT_SSLR_SNR" + snr + "dB.mat", '/home/qian/Documents/data/sslr/'
        fnorm, faxis = minmax_scale, 1
    elif args.trA == 'melgcc':
        te1,te2, apath = "Testing1_melGCCPHAT_SSLR.mat", "Testing2_melGCCPHAT_SSLR.mat", '/home/maulik/data/sslr'
        fnorm, faxis = funcs.minmax_norm2d, (1, 2)

    return tr, te1, te2, apath, fnorm, faxis
