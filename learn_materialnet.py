#!/usr/bin/env python

'''
     Central script for learning spectral filters on spectrometric data.
'''

# System imports
import os
import sys
import pickle
import warnings
import argparse
import glob
from tqdm import tqdm
import pdb

warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
from scipy import io
from scipy import interpolate
from scipy.signal import gaussian

# Plotting
import matplotlib.pyplot as plt
plt.ion()

# PyTorch
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.decomposition import PCA

# Import train/test splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Standard prescaler to condition data
from sklearn.preprocessing import StandardScaler

# Import our models
import models
import utils
from utils import UnitScaler

if __name__ == '__main__':
    # Data source type. Unless you are using data from your own optical
    # setup, set this to 'spectrometer'
    srctype = 'spectrometer'

    # If True, window data to wavelengths for which the optical setup was
    # built.
    window_data = True

    # For now, fix the batch size
    batch_size = 256

    # Data root
    root = 'data/'

    # Range of wavelengths. NIR is 600 - 900, VIS is 400 - 700 and
    # VISNIR is 400 -900. The accuracy is highly dependent on the
    # wavelength range.
    wvlrange = 'NIR'

    # Smoothening details
    polyfilt = True
    res = 10

    # Limits to remove zero or saturated data
    minval = 3000
    maxval = 70000

    # Grab 10% of testing data for validation
    valid_size = 0.1

    # Parse the command line.
    # To get list of options, type "python learn_materialnet.py -h", or
    # check train.sh
    args = models.create_parser('network.ini')

    # Material names -- usually a subsete of all materials
    classes = ['fabric', 'paper', 'plant',
               'plastic', 'wood']

    material_dictionary = {classes[idx]:idx for idx in range(len(classes))}

    # Foldernames -- Edit this to add more data or remove data folders
    if srctype == 'spectrometer':
        folders = ['skin', 'lab2', 'lab3', 'office1', 'schenleypark1',
                   'schenleypark2', 'porter1', 'sorrels1', 'nsh1']
    elif 'transfer' in srctype:
        folders = ['%s_%s'%(srctype, args.illuminant)]

        minval = 0
        maxval = 1e9

        polyfilt = False
    elif 'setup' in srctype:
        folders = ['%s_%s'%(srctype, args.illuminant)]

        minval = 5e-2
        maxval = 1 - 5e-2
        polyfilt = True

    # Save models here
    savedir = 'models'

    # Find which device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Starting materialnet. Training on %s'%device)

    # Generate wavelnegth ranges
    if wvlrange == 'VIS':
        lmb1 = 400
        lmb2 = 700
    elif wvlrange == 'NIR':
        lmb1 = 600
        lmb2 = 900
    elif wvlrange == 'VISNIR':
        lmb1 = 400
        lmb2 = 900

    print('Collecting data')
    foldernames = [os.path.join(root, folder) for folder in folders]
    collection = utils.load_spectra_folders(foldernames,
                                            smoothen=True,
                                            bsize=10,
                                            minval=minval,
                                            maxval=maxval)
    data, wavelengths, label_names = collection

    labels = utils.names2labels(label_names, material_dictionary)

    # Remove any unknown labels
    data = data[labels != -1, :]
    labels = labels[labels != -1]

    # Clip data after imposing a (smooth) window
    sigma = res/(2*np.sqrt(2*np.log(2)))
    winsize = int(12*sigma/(wavelengths[0, 1] - wavelengths[0, 0]))

    wvl_window = wavelengths.ravel()[:winsize]
    cwl = wvl_window.mean()

    kernel = np.exp(-pow(wvl_window - cwl, 2)/(2*sigma*sigma))

    # Create window
    window = np.zeros(wavelengths.size)

    idx1 = abs(wavelengths - lmb1).ravel().argmin()
    idx2 = abs(wavelengths - lmb2).ravel().argmin()

    window[idx1:idx2] = 1
    window = np.convolve(np.convolve(window, kernel/kernel.sum(), mode='same'),
                         kernel/kernel.sum(), mode='same')

    if polyfilt:
        print('Filtering data')
        for idx in tqdm(range(data.shape[0])):
            spec = np.convolve(data[idx, :], kernel/kernel.sum(), mode='same')
            data[idx, :] = spec
    
    if srctype == 'spectrometer':
        # Load illuminant to multiply with data -- If you wish to
        # learn model only on reflectance data, set args.illuminant to
        # 'none'.
        #
        # Instead, if you have illumination x spectral response of your 
        # optical setup, save that data in data/illuminants and set
        # args.illuminant to that name.
        illumdata = io.loadmat('data/illuminants/%s.mat'%args.illuminant)
        illuminant = illumdata['illuminant']
        wvl_illum = illumdata['wavelengths']

        func = interpolate.interp1d(wvl_illum.ravel(),
                                    illuminant.ravel(),
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=0)
        illuminant = func(wavelengths).reshape(1, -1)

        if window_data:
            data = data*window.reshape(1, -1)*illuminant
        else:
            data = (data*illuminant)[:, idx1:idx2]
            wavelengths = wavelengths[:, idx1:idx2]

    elif srctype == 'setup':
        if window_data:
            data = data*window.reshape(1, -1)

    # Divide data by its sum
    data = data/data.sum(1).reshape(-1, 1)

    # Remove data with negative spectra
    valids = ~(data.min(1) < 0)
    data = data[valids, :]
    labels = labels[valids]

    nbands = data.shape[1]
    nlabels = len(classes)

    # Split data
    print('Splitting data')
    Xtrain, Xtest, ytrain, ytest = train_test_split(data.astype(np.float32),
                                                    labels.ravel().astype(int),
                                                    train_size=args.train,
                                                    stratify=labels)

    print('Splitting for validation')
    Xvalid, Xtest, yvalid, ytest = train_test_split(Xtest,
                                                    ytest,
                                                    train_size=valid_size,
                                                    stratify=ytest)

    print('Learning scaler')
    scaler = UnitScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    Xvalid = scaler.transform(Xvalid)

    # Learn PCA basis so that we can initialize filters with this.
    print('Learning PCA filters for initialization')
    pca = PCA(n_components=args.nfilters)
    pca.fit(Xtrain)
    init_filters = torch.Tensor(pca.components_)

    # Convert data to tensors
    Xtrain = torch.from_numpy(Xtrain)
    Xtest = torch.from_numpy(Xtest)
    Xvalid = torch.from_numpy(Xvalid)

    ytrain = torch.from_numpy(ytrain)
    ytest = torch.from_numpy(ytest)
    yvalid = torch.from_numpy(yvalid)

    # Create dataloader
    trainset = models.MaterialNetDataset(Xtrain, ytrain)
    testset = models.MaterialNetDataset(Xtest, ytest)
    validset = models.MaterialNetDataset(Xvalid, yvalid)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)

    # Create classification criterion. Add weights for classes
    weights = np.sqrt(np.bincount(ytrain).astype(np.float32))

    weights = torch.from_numpy(weights.sum()/weights).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # (Thanks to Rick!) Create two separate modules
    print('Creating filter module')
    filters = nn.Linear(nbands, args.nfilters)

    with torch.no_grad():
        filters.weight = nn.Parameter(init_filters)

    print('Creating classifier module')
    net = models.MaterialClassifier(input_dim=args.nfilters,
                                    output_dim=nlabels,
                                    config=args.config)

    # Send network to gpu
    filters.to(device)
    net.to(device)

    # Create optimizer
    params = list(filters.parameters()) + list(net.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate,
                           weight_decay=args.decay)

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.9,
                                  patience=20, verbose=True)

    # Now run
    print('Now starting optimization')
    best_valid_accuracy = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        # Predict on validation dataset at every epoch and see how we are doing
        with torch.no_grad():
            valid_accuracy = 0
            for data in validloader:
                inputs, labels_batch = data
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)

                filter_outputs = filters(inputs)
                outputs = net(filter_outputs)
                _, labels_pred = torch.max(outputs.data, 1)

                valid_accuracy += (labels_pred == labels_batch).sum().item()

            valid_accuracy /= Xvalid.shape[0]
            print('Epoch %d; Validation accuracy %.2f'%(epoch,
                                                        valid_accuracy*100))
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_epoch = epoch
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump([filters, net], f)
                    
        # Step the scheduler with this validation score
        scheduler.step(valid_accuracy)
                
        for i, data in enumerate(trainloader, 0):
            # Inputs
            inputs, labels_batch = data

            # Send inputs to device
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Forward, backward, optimizer
            filter_outputs = filters(inputs)
            outputs = net(filter_outputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

    # Load best epoch model
    print('Loading best model from epoch %d'%best_epoch)

    with open('best_model.pkl', 'rb') as f:
        filters, net = pickle.load(f)

    print('Evaluating the model')
    # Switch to evaluation mode
    net.eval()

    # Now evaluate accuracy for test dataset
    ypred = torch.zeros_like(ytest)

    for i, data in enumerate(testloader, 0):
        inputs, labels_batch = data
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)

        filter_outputs = filters(inputs)
        outputs = net(filter_outputs)
        _, ypred[i*batch_size:(i+1)*batch_size] = torch.max(outputs.data, 1)

    # Get confusion matrix and report
    print('Computing metrics')
    confusion = confusion_matrix(ytest, ypred)
    report = classification_report(ytest, ypred, target_names=classes)

    print(report)

    # Grab the filters and bias
    learned_filters = filters.weight.to('cpu').detach().numpy()
    learned_bias = filters.bias.to('cpu').detach().numpy()

    # Get learned model to CPU
    filters.to('cpu')
    net.to('cpu')

    # If windowing is done, the filters also need to be windowed
    if window_data:
        learned_filters = learned_filters*window.reshape(1, -1)

    mat_dict = dict()
    mat_dict['confusion'] = confusion
    mat_dict['report'] = report
    mat_dict['ytrue'] = ytest
    mat_dict['ypred'] = ypred
    mat_dict['filters'] = learned_filters
    mat_dict['wvl_filters'] = wavelengths
    mat_dict['bias'] = learned_bias

    io.savemat('%s/%s_%s_%d_aux.mat'%(args.savedir,
                                      args.experiment,
                                      args.illuminant,
                                      args.nfilters),
               mat_dict)

    # Pickle the mode
    savename = '%s/%s_%s_%d_model.pkl'%(args.savedir,
                                        args.experiment,
                                        args.illuminant,
                                        args.nfilters)
    with open(savename, 'wb') as f:
        pickle.dump([filters, net, window, scaler, material_dictionary], f)
