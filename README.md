# Programmable Spectrometry: Per-pixel Material Classification using Learned Spectral Filters

This repository contains spectrometric data, as well as python scripts necessary to learn spectral filters for per-pixel material classification.

Additional details can be obtained in the paper given below.

# Data
Download spectrometry data from [here](https://www.dropbox.com/sh/y5mjfqlqt4mu97a/AADEWvfiFtJvGptIkh3TIePSa?dl=0) and save it under **data** folder. The current folder
structure will then be:
./
|\_ train.sh
|\_ ...
|\_ data
    |\_ home1
    |\_ home2
    |\_ ...
    |\_ sorrels1


# Scripts
This folder contains the following files:
1. **train.sh**: Use this shell script to learn spectral filters
2. **learn\_materialnet.py**: Central python script to learn spectral filters
3. **models.py**: Python module to build neural network model and argument parser
4. **network.ini**: Define the size of each fully connected layer in this file
5. **utils.py**: Miscellaneous utilities

# Paper
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

