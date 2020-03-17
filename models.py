#!/usr/bin/env python

'''
Description of files goes here.
'''

# System imports
from sklearn.decomposition import PCA
import os
import sys
import argparse
import configparser
import ast
import pdb

# Scientific computing
import numpy as np

# PyTorch stuff
import torch
from torch import nn
import torch.nn.functional as F

# Skorch
from skorch import NeuralNetClassifier

torch.manual_seed(0)


class AutoEncoder(nn.Module):
    '''
        Create autoencoder to learn a low dimensional projection of spectra.

        Inputs:
            input_dim: Dimension of spectra
            feature_dim: Hidden dimension, which translates to number of 
                discriminating filters.

    '''

    def __init__(self, input_dim=256, feature_dim=10):
        # Initialize the nn.Module
        super(AutoEncoder, self).__init__()

        # Now create encoder
        self.encoder = nn.Linear(input_dim, feature_dim, bias=False)

        # And then decoder
        self.decoder = nn.Linear(feature_dim, input_dim, bias=False)

    def forward(self, X, **kwargs):
        # Apply encoder and then decoder
        X = self.encoder(X)
        X = self.decoder(X)

        return X


class MaterialClassifier(nn.Module):
    '''
        NeuralNet module for classifying spectral features. The module is
        designed such that the first layer forms the set of discriminating
        filters.

        The general configuration:
            input -> linear -> batch norm. -> ReLU -> (repeat) -> predict

        Inputs:
            input_dim: Dimension of input data. This could be the spectrum
                directly or projection onto PCA basis (learned separately)
            output_dim: Number of classes in training
            dropout: Fraction of connections to be dropped during training
            config: (Python) list of sizes of each layer. By default, it is
                [64] which translates to one hidden dimension of 64 outputs
            init_filters: Intial entries for first layer. This is useful
                for initialization with PCA filters.
    '''

    def __init__(self, input_dim=256,
                 output_dim=10,
                 dropout=0.5,
                 config=[64],
                 init_filters=None):
        '''
            Initialize class.
        '''
        # Initialize nn.module
        super(MaterialClassifier, self).__init__()

        # Create functional units
        self.dropout = nn.Dropout(dropout)

        # List of modules
        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Insert input dimension and output dimensions
        config.insert(0, input_dim)
        config.append(output_dim)

        # Now for each layer, create a linear layer, then batchnorm, then
        # relu
        for idx in range(len(config)-1):
            # Linear
            self.linear_layers.append(nn.Linear(config[idx],
                                                config[idx+1]))
            # Batch normalization
            self.bn_layers.append(nn.BatchNorm1d(num_features=config[idx+1]))

        # Initialize weights if provided
        if init_filters is not None:
            with torch.no_grad():
                self.linear_layers[0].weight = torch.nn.Parameter(init_filters)

    def forward(self, X, **kwargs):
        '''
            Foward operator for torch
        '''
        for idx in range(len(self.linear_layers)-1):
            # Linear
            X = self.linear_layers[idx](X)

            # Batch normalization
            X = self.bn_layers[idx](X.permute(0, 2, 1)).permute(0, 2, 1)

            # ReLU
            X = F.relu(X)

            # Dropout
            X = self.dropout(X)

        # Final layer has only linear
        X = self.linear_layers[-1](X)

        return X.squeeze()


class MaterialNetDataset(torch.utils.data.Dataset):
    '''
        Data loader for training classifier on material spectra.

        Inputs:
            data: Input data with each row being spectral signature.
            labels: Labels for each spectral class

        Outputs:
            Decided by torch.
    '''

    def __init__(self, data, labels, transform=None):
        '''
            Simply create a member for data
        '''
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        T = self.data.shape[1]
        #N = self.labels.shape[1]
        return self.data[idx, :].reshape(-1, T), self.labels[idx]


def create_parser(config_name='config.ini'):
    '''
        Create a commandline parser for configuring neuralnet training for
        spectral classification.

        Inputs:
            config_name: Configuration file that contains number of layers
                and number of neurons per layer.

        Outputs:
            args: Namespace object with following fields:
                experiment: Name of experiment. Required.
                nfilters: Number of filters to learn. This is the first
                    hidden dimension of network. Default is 10
                dropout: Dropout fraction. Default is 0.5
                savedir: Directory for saving final model
                train: Fraction of data to use for training. Default is 0.5
                lr: Learning rate. Default is 0.1
                epochs: Number of epochs to train for. Default is 10
    '''
    # Create a new parser
    parser = argparse.ArgumentParser(description='MaterialNet parser')

    # Name of experiment
    parser.add_argument('-e', '--experiment', action='store',
                        required=True, type=str, help='Experiment name')

    # Number of filters to learn
    parser.add_argument('-n', '--nfilters', action='store', type=int,
                        default=10, help='Number of filters to learn')

    # Dropout rate
    parser.add_argument('-d', '--dropout', action='store', type=float,
                        default=0.5, help='Dropout')

    # Save directory
    parser.add_argument('-s', '--savedir', action='store', type=str,
                        default='./', help='Save directory')

    # Training fraction
    parser.add_argument('-t', '--train', action='store', type=float,
                        default=0.5, help='Training fraction')

    # Learning rate
    parser.add_argument('-l', '--learning_rate', action='store',  type=float,
                        default=0.1, help='Learning rate')

    # Number of epochs
    parser.add_argument('-i', '--epochs', action='store', type=int,
                        default=10, help='Number of epochs')

    # Weight decay constant
    parser.add_argument('-w', '--decay', action='store', type=float,
                        default=0, help='Weight decay for optimizer')

    # Illuminant
    parser.add_argument('-b', '--illuminant', action='store', type=str,
                        default='constant', help='Illuminant/bulb')

    # Now arse
    args = parser.parse_args()

    # Next read configuration file
    config = configparser.ConfigParser()
    config.read(config_name)

    # Parse the configuratoin file
    args.config = ast.literal_eval(config['ARCH']['config'])
    args.datapath = config['DATA']['path']
    args.dataname = config['DATA']['name']

    # All good, let's return
    return args
