# Master's thesis

This software is practical part of my Master's thesis at Faculty of Information Technology, Brno University of Technology. The program uses a multiobjective NSGA-II algorithm for designing accurate and compact CNNs.

## Prerequisites

To use this software, you will need to install [Anaconda3](https://docs.anaconda.com/anaconda/install/).

## Installation

Create Anaconda environment:
```
conda env create -f tf-gpu-2.4.1.yml
```
Activate Anaconda environment:
```
source activate tf-gpu-2.4.1-pristas
```

## Usage
You have 2 options how to work with this program:
1. Use GUI:
```
python3 gui.py
```
2. Use command line:
```
main.py
main.py -h | --help
main.py [options]
Options:
  -h --help   Show help message.
  -p --pop    Number of individuals in each generation (integer > 1) [default: 15].
  -g --gen    Number of generations (integer > 0) [default: 15]
  -m --mut    Mutation probability (float 0-1) [default: 0.15].
  --phases    Number of phases in genotypes (integer > 1) [default: 2].
  --modules   Number of modules in genotypes (integer 1-10) [default: 6].
```
