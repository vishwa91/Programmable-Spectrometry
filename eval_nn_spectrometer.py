#!/usr/bin/env python

'''
    Evaluate neural networks on the full cube
'''

# System imports
import os
import sys
import pickle
import glob

# Please suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
from scipy import signal, io, interpolate
from skimage import color

# Torch
import torch

# Plotting
import matplotlib.pyplot as plt
import cv2

# Split data into train and test
from sklearn.model_selection import train_test_split, GridSearchCV

# Classification reports
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as metric

import utils

if __name__ == '__main__':
    expname = 'lab4'                                 # Name of experiment
    modeldir = 'models'
    nfilters = 4                                    # Number of filters?
    filters_name = 'test'                           # Name of filters
    lmb1 = 600
    lmb2 = 900
    res = 10
    illumfile = 'none'                             # Evaluate with system spectral response

    # Load model
    print('Loading neuralnet model')
    loadname ='%s/%s_%s_%d_model.pkl'%(modeldir, filters_name,
                                       illumfile, nfilters)
    with open(loadname, 'rb') as f:
        filters, net, window, scaler, material_dict  = pickle.load(f)

    # Switch to evaluation mode
    filters.eval()
    net.eval()

    # Load cube
    print('Loading spectral data')
    foldername = 'data/%s'%expname
    collection = utils._load_spectra_folder(foldername,
                                            smoothen=False,
                                            bsize=20,
                                            minval=0,
                                            maxval=1e5)
    spectra, data_wvl, label_names, light, offset = collection
    labels = utils.names2labels(label_names, material_dict)

    spectra = spectra[labels != -1, :]
    labels = labels[labels != -1]

    # Constants for filtering
    sigma = res/(2*np.sqrt(2*np.log(2)))
    dlambda = data_wvl[0, 1] - data_wvl[0, 0]
    winsize = int(12*sigma/dlambda)

    #spectra = utils.smoothen_spectra(spectra, res/dlambda)

    wvl_window = data_wvl.ravel()[:winsize]
    cwl = wvl_window.mean()

    kernel = np.exp(-pow(wvl_window - cwl, 2)/(2*sigma*sigma))


    idx1 = abs(data_wvl - lmb1).ravel().argmin()
    idx2 = abs(data_wvl - lmb2).ravel().argmin()

    # Load illuminant to multiply with data
    illumdata = io.loadmat('data/illuminants/%s.mat'%illumfile)
    illuminant = illumdata['illuminant']
    wvl_illum = illumdata['wavelengths']

    func = interpolate.interp1d(wvl_illum.ravel(),
                                illuminant.ravel(),
                                kind='linear',
                                bounds_error=False,
                                fill_value=0)
    illuminant = func(data_wvl).reshape(1, -1)

    # Just clip, do not window
    spectra = spectra*(window).reshape(1, -1)*illuminant
    #spectra = (spectra*illuminant)[:, idx1:idx2]
    #data_wvl = data_wvl[:, idx1:idx2]

    spectra = spectra/spectra.sum(1).reshape(-1, 1)

    # Scale data
    spectra = scaler.transform(spectra)

    # Then evaluate network
    print('Evaluating classifier network')
    filters_output = filters(torch.tensor(spectra.astype(np.float32)))
    net_output = net(filters_output[:, None, :])

    # Compute softmax
    net_output = net_output.exp()/net_output.exp().sum(1, keepdim=True)

    # Predict
    _, labels_pred = torch.max(net_output, 1)

    labels_pred = labels_pred.detach().numpy()

    # Print results
    label_union = np.union1d(np.unique(labels), np.unique(labels_pred))
    names = list(material_dict.keys())
    names = [names[idx] for idx in label_union]
    report = classification_report(labels, labels_pred,
                                   labels=label_union,
                                   target_names=names)
    print(report)
    print(confusion_matrix(labels, labels_pred))

    files = glob.glob('%s/*.mat'%foldername)
    cnt = 0
    for idx, file in enumerate(files):
        if 'zero' in file or 'spectralon' in file:
            continue
        else:
            print('%d, %s'%(cnt, os.path.split(file)[1]))
            cnt += 1

    # Compute precision recall and other metrics
    precision, recall, fbeta_score, support = metric(labels,
                                                     labels_pred,
                                                     average=None)

    fname = '%s_%s_%d'%(filters_name, illumfile, nfilters)
    io.savemat('results/%s_%s_metrics.mat'%(expname, fname),
               {'precision': precision,
                'recall': recall,
                'fbeta_score': fbeta_score,
                'support':support})
               

    fig0 = plt.figure(0)
    plt.plot(labels, label='GT')
    plt.plot(labels_pred, label='Pred')
    plt.legend()
    plt.savefig('labels_cmp.png')

    fig1 = plt.figure(1)
    plt.plot(data_wvl.T, spectra[::50, :].T)
    plt.savefig('spectra.png')

    #net_output = net_output.detach().numpy()
    #fig2 = plt.figure(2)
    #plt.plot(net_output[2400:2500, :].T)

    plt.show()
    

