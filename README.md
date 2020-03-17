# Programmable Spectrometry: Per-pixel Material Classification using Learned Spectral Filters

This repository contains spectrometric data, as well as python scripts necessary to learn spectral filters for per-pixel material classification.

Additional details can be obtained in the paper given below.

## Data
Download spectrometry data from [here](https://www.dropbox.com/sh/y5mjfqlqt4mu97a/AADEWvfiFtJvGptIkh3TIePSa?dl=0) and save it under **data** folder. The current folder
structure will then be:

* train.sh
* ...
* data
    * home1
    * home2
    * ...
    * sorrels1


## Scripts and learning procedure
This folder contains the following files:

1. **train.sh**: Use this shell script to learn spectral filters
2. **learn\_materialnet.py**: Central python script to learn spectral filters
3. **models.py**: Python module to build neural network model and argument parser
4. **network.ini**: Define the size of each fully connected layer in this file
5. **utils.py**: Miscellaneous utilities

The python training script requires the following options:

* **--experiment**: Name of experiment
* **--nfilters**: Number of spectral filters
* **--dropout**: Dropout fraction (0 to 1)
* **--savedir**: Directory to save final learned model
* **--train**: Fraction of data to use for training
* **--learning\_rate**: Learning rate
* **--epochs**: Number of epochs
* **--decay**: Decay factor, used to smoothen learned spectral filters
* **--illuminant**: Name of illuminant file. Set it to "none" to have no illuminant

The file ```train.sh``` simplifies the options by setting all options except experiment name. You can edit other options in this file

Once you learn the filters, you can test the accuracy of learned filters on different data with ```eval_nn_filters.py```.


## Paper
To be available soon!

# Citation
```
@inproceedings{saragadam2020programmable,
  title={Programmable Spectrometry: Per-pixel Material Classification using Learned Spectral Filters},
  author={Saragadam, Vishwanath and Sankaranarayanan, Aswin C},
  booktitle={IEEE International Conference on Computational Photography (ICCP)},
  pages={1--11},
  year={2020},
  organization={IEEE}
}
```

