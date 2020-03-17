#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# System imports
import os
import sys
import time
import pickle
import pdb
import glob

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal

# Plotting
import matplotlib.pyplot as plt

def load_spectra_folders(foldernames, zero_name='zero', ref_name='spectralon',
                         minval=5000, maxval=70000, smoothen=False, bsize=10):
    '''
        Load spectral profiles from multiple folders.

        Inputs:
            foldername: Folder name
            zero_name: Name of offset spectral measurement
            ref_name: Name of light source spectral measurement
            minval, maxval: Valid range of spectral measurements
            smoothen: If True, smoothen the spectral profiles before division
                by reference
            bsize: If smoothen is True, this is the size of boxcar

        Outputs:
            spectra: nsamples x nwavelengths matrix
            wavelengths: Wavelengths for the spectra
            label_names: nsamples list of label names
    '''

    spectra = []
    label_names = []

    for foldername in foldernames:
        data = _load_spectra_folder(foldername, zero_name, ref_name,
                                    minval, maxval, smoothen, bsize)
        spectra_chunk, wavelengths, label_names_chunk, _,_ = data

        spectra.append(spectra_chunk)
        label_names += label_names_chunk

    spectra = np.vstack(spectra)

    return spectra, wavelengths, label_names

def _load_spectra_folder(foldername, zero_name='zero', ref_name='spectralon',
                         minval=5000, maxval=70000, smoothen=False, bsize=10):
    '''
        Load spectral profiles from a folder.

        Inputs:
            foldername: Folder name
            zero_name: Name of offset spectral measurement
            ref_name: Name of light source spectral measurement
            minval, maxval: Valid range of spectral measurements
            smoothen: If True, smoothen the spectral profiles before division
                by reference
            bsize: If smoothen is True, this is the size of boxcar

        Outputs:
            spectra: nsamples x nwavelengths matrix
            wavelengths: Wavelengths for the spectra
            label_names: nsamples list of label names
            light: Spectrum of light source
            zero: Spectrum of offset
    '''
    # Load zero spectrum
    data = io.loadmat('%s/%s.mat'%(foldername, zero_name))
    zero = data['spectrum'].mean(0)
    wavelengths = data['wavelengths']

    # Load light spectrum
    data = io.loadmat('%s/%s.mat'%(foldername, ref_name))
    light = data['spectrum'].mean(0) - zero

    if smoothen:
        kernel = np.ones(bsize)/bsize
        light = np.convolve(light, kernel, 'same')

    spectra = []
    label_names = []
    valids = []

    # Get hold of all mat files
    filenames = glob.glob('%s/*.mat'%foldername)

    for filename in filenames:
        if (zero_name in filename) or (ref_name in filename):
            continue

        data = io.loadmat(filename)
        spectrum = data['spectrum']
        nspectrum = spectrum.shape[0]

        validity = (spectrum.max(1) > minval)*(spectrum.max(1) < maxval)

        spectrum = spectrum - zero

        if smoothen:
            for idx in range(nspectrum):
                spectrum[idx, :] = np.convolve(spectrum[idx, :],
                                               kernel, 'same')

        spectrum = spectrum/light

        spectra.append(spectrum)
        valids.append(validity.reshape(-1, 1))

        filename = filename.replace('\\', '/')
        label_name = filename.split('/')[-1].split('_')[0]

        label_names += [label_name for idx in range(nspectrum)]

    # Concatenate spectra
    spectra = np.vstack(spectra)
    valids = np.vstack(valids)

    spectra = spectra[valids.ravel() == 1, :]
    label_names = [label_names[idx] for idx in range(valids.size) if valids[idx] == 1]

    return spectra, wavelengths, label_names, light, zero

def stack2mosaic(imstack):
    '''
        Convert a 3D stack of images to a 2D mosaic

        Inputs:
            imstack: (H, W, nimg) stack of images

        Outputs:
            immosaic: A 2D mosaic of images
    '''
    H, W, nimg = imstack.shape

    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))

    immosaic = np.zeros((H*nrows, W*ncols), dtype=imstack.dtype)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img_idx = row_idx*ncols + col_idx
            if img_idx >= nimg:
                return immosaic

            immosaic[row_idx*H:(row_idx+1)*H, col_idx*W:(col_idx+1)*W] = \
                                              imstack[:, :, img_idx]

    return immosaic
def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1))
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def savep(data, filename):
    '''
        Tiny wrapper to store data as a python pickle.

        Inputs:
            data: List of data
            filename: Name of file to save
    '''
    f = open(filename, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def loadp(filename):
    '''
        Tiny wrapper to load data from python pickle.

        Inputs:
            filename: Name of file to load from

        Outputs:
            data: Output data from pickle file
    '''
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def display_time(total_time):
    '''
        Tiny wrapper to print time in an appropriate way.

        Inputs:
            total_time: Raw time in seconds

        Outputs:
            None
    '''
    if total_time < 60:
        print('Total scanning time: %.2f seconds'%total_time)
    elif total_time < 3600:
        print('Total scanning time: %.2f minutes'%(total_time/60))
    elif total_Time < 86400:
        print('Total scanning time: %.2f hours'%(total_time/3600))
    else:
        print('Total scanning time: %.2f days'%(total_time/86400))
        print('... what are you really doing?')

def dither(im):
    '''
        Implements Floyd-Steinberg spatial dithering algorithm

        Inputs:
            im: Grayscale image normalized between 0, 1

        Outputs:
            imdither: Dithered image
    '''
    H, W = im.shape
    imdither = np.zeros((H+1, W+1))

    # Pad the last row/column to propagate error
    imdither[:H, :W] = im
    imdither[H, :W] = im[H-1, :W]
    imdither[:H, W] = im[:H, W-1]
    imdither[H, W] = im[H-1, W-1]

    for h in range(0, H):
        for w in range(1, W):
            oldpixel = imdither[h, w]
            newpixel = (oldpixel > 0.5)
            imdither[h, w] = newpixel

            err = oldpixel - newpixel
            imdither[h, w+1] += (err * 7.0/16)
            imdither[h+1, w-1] += (err * 3.0/16)
            imdither[h+1, w] += (err * 5.0/16)
            imdither[h+1, w+1] += (err * 1.0/16)

    return imdither[:H, :W]

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def deconvwnr1(sig, kernel, wconst=1e-2):
    '''
        Deconvolve a 1D signal using Wiener deconvolution

        Inputs:
            sig: Input signal
            kernel: Impulse response
            wconst: Wiener deconvolution constant

        Outputs:
            sig_deconv: Deconvolved signal
    '''

    sigshape = sig.shape
    sig = sig.ravel()
    kernel = kernel.ravel()

    N = sig.size
    M = kernel.size

    # Padd signal to regularize 
    sig_padded = np.zeros(N+2*M)
    sig_padded[M:-M] = sig

    # Compute Fourier transform
    sig_fft = np.fft.fft(sig_padded)
    kernel_fft = np.fft.fft(kernel, n=(N+2*M))

    # Compute inverse kernel
    kernel_inv_fft = np.conj(kernel_fft)/(np.abs(kernel_fft)**2 + wconst)

    # Now compute deconvolution
    sig_deconv_fft = sig_fft*kernel_inv_fft

    # Compute inverse fourier transform
    sig_deconv_padded = np.fft.ifft(sig_deconv_fft)

    # Clip
    sig_deconv = np.real(sig_deconv_padded[M//2:M//2+N])

    return sig_deconv.reshape(sigshape)

def lowpassfilter(data, order=5, freq=0.5):
    '''
        Low pass filter the input data with butterworth filter.
        This is based on Zackory's github repo: 
            https://github.com/Healthcare-Robotics/smm50

        Inputs:
            data: Data to be filtered with each row being a spectral profile
            order: Order of butterworth filter
            freq: Cutoff frequency

        Outputs:
            data_smooth: Smoothed spectral profiles
    '''
    # Get butterworth coefficients
    b, a = signal.butter(order, freq, analog=False)

    # Then just apply the filter
    data_smooth = signal.filtfilt(b, a, data)

    return data_smooth

def smoothen_spectra(data, bsize=10, method='gauss'):
    '''
        Smoothen rows of spectra with some kernel

        Inputs:
            data: nsamples x nwavelengths spectral matrix
            bsize: Size of blur kernel. For gaussian blur, it is FWHM
            method: 'box', 'poly', or 'gauss'. If ply, bsize is the order of
                the polynomial

        Outputs:
            data_smooth: Smoothened data
    '''
    data_smooth = np.zeros_like(data)

    if method == 'box':
        kernel = np.ones(bsize)/bsize
    elif method == 'gauss':
        sigma = bsize/(2*np.sqrt(2*np.log(2)))
        kernlen = int(sigma*12)
        x = np.arange(-kernlen//2, kernlen//2)
        kernel = np.exp(-(x*x)/(2*sigma*sigma))

    for idx in range(data.shape[0]):
        data_smooth[idx, :] = np.convolve(data[idx, :], kernel, 'same')

    return data_smooth

def polyfilt(data, polyord=10):
    '''
        Polynomial filter a 1D vector.

        Inputs:
            data: 1D vector which requires smoothening
            polyord: Order of the polynomial to use for fitting

        Outputs:
            data_filt: poly fitted data
    '''
    datashape = data.shape

    data_vec = data.ravel()
    N = data_vec.size

    x = np.arange(N)

    coefs = np.polyfit(x, data_vec, polyord)
    func = np.poly1d(coefs)
    data_fit = func(x)

    return data_fit.reshape(datashape)

def names2labels(label_names, label_dict):
    '''
        Convert a list of label names to an array of labels

        Inputs:
            label_names: List of label names
            label_dict: A dictionary of the form, label_name:label_idx.

        Outputs:
            labels: Array of label indices. The label is -1 if key was not found
                in the dictionary
    '''
    labels = []

    for label_name in label_names:
        if label_name in label_dict.keys():
            labels.append(label_dict[label_name])
        else:
            labels.append(-1)

    return np.array(labels)

class UnitScaler(object):
    '''
        This is a place holder for StandardScaler when scaling is not utilized
    '''
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x

if __name__ == '__main__':
    foldername = '../experiments/spectra/lab2'
    lmb1 = 600
    lmb2 = 950
    
    spectra, wavelengths, label_names, _, _ = _load_spectra_folder(foldername,
                                                                   smoothen=True,
                                                                   bsize=20)

    wavelengths = wavelengths.ravel()

    idx1 = abs(wavelengths - lmb1).argmin()
    idx2 = abs(wavelengths - lmb2).argmin()

    # Create material dictionary
    material_names = np.unique(label_names)
    material_dict = dict(zip(material_names, np.arange(material_names.size)))

    labels = names2labels(label_names, material_dict)

    specraw = spectra[::50, idx1:idx2]
    specfilt = lowpassfilter(specraw, freq=0.2)

    plt.plot(wavelengths[idx1:idx2], specraw.T, label='Raw')
    plt.plot(wavelengths[idx1:idx2], specfilt.T, label='Polyfilt')
    plt.legend()
    plt.show()

    
