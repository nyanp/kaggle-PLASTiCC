# kaggle-PLASTiCC

## install
- use following commands on GCP (ubuntu 16.04)

```bash
sudo apt install gcc g++ git tmux
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
sh Anaconda3-5.3.1-Linux-x86_64.sh
conda create -n plasticc python=3.6
pip install numpy pandas feather-format lightgbm sncosmo astropy iminuit tqdm matplotlib
pip install pyarrow==0.9.0
git clone https://github.com/nyanp/kaggle-PLASTiCC
cd kaggle-PLASTiCC
mkdir input
gsutil -m cp -r gs://your-directory-for-input-files/ input/
```
