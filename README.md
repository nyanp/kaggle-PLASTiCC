This repository is a part of 3rd place solution in [Kaggle PLAsTiCC Astronomical Classification](https://www.kaggle.com/c/PLAsTiCC-2018).
Our team (major tom) consists of 3 members (mamas, yuvals, nyanp), and this repository contains code to

- Reproduce nyanp's LightGBM model
- Create nyanp features for yuval/mamas's model

## Environment setup
You can run these scripts on your local machine or GCP. I used GCP for extracting template features, and local machine (Win10, Core i7-6700K, 48GB RAM) for others.

### Setup on your local machine
1. Prepare a virtual environment for this project (python 3.5 is recommended)
1. Clone this repository
1. Run `pip install -r requirements.txt`

### Setup on GCP
Recommended environment:
- OS: Ubuntu 16.04 LTS
- CPU: x16+
- RAM: 120GB+
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
See [entry_points.md](entry_points.md)
