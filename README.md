This repository is a part of 3rd place solution in [Kaggle PLAsTiCC Astronomical Classification](https://www.kaggle.com/c/PLAsTiCC-2018).
Our team (major tom) consists of 3 members (mamas, yuvals, nyanp), and this repository contains code of nyanp's models and features which was used by models of teammates.

## Directory structure
- features/ : Feature engineering functions
- model/ : Training LightGBM
- lsst/ : LSST throuputs files copied from sncosmo
- config.py : Configuration relates to I/O directory and debug mode
- step*.py : Entrypoint of feature engineering and modeling

## Environment setup
You can run script both Windows and Linux. If you use your local machine, ~200GB RAM is required to run.

### Setup on your local machine
1. Prepare a virtual environment for this project (python 3.5 is recommended)
1. Clone this repository
1. Run `pip install -r requirements.txt`

### Setup on GCP
Recommended environment:
- OS: Ubuntu 16.04 LTS
- CPU: vCPU x 32
- RAM: 208GB
- Storage: 100GB+ (SSD storage is recommended)

After creating GCP instance, connect it with SSL and follow these commands:
```bash
sudo apt-get update
sudo apt install gcc g++ git tmux
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
sh Anaconda3-5.3.1-Linux-x86_64.sh
conda create -n plasticc python=3.5
source activate plasticc

git clone https://github.com/nyanp/kaggle-PLASTiCC
cd kaggle-PLASTiCC
git checkout cleanup

pip install numpy
pip install -r requirements.txt
```

## Data setup
- `mkdir input`
- `cd input`
- download following files to `input` directory (via [Kaggle API](https://github.com/Kaggle/kaggle-api) or any way you want)
    - training_set.csv
    - test_set.csv
    - training_set_metadata.csv
    - test_set_metadata.csv


## Configure
You can change I/O directory and running mode by editing [config.py](config.py). See comments in the file in detail.

## Run
See [entry_points.md](entry_points.md). If you just want to use precompiled feature set to train yuval/mamas's model,
use feature files in share/ directory.
