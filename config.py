
DATA_DIR = "input/"  # Save input files in feather-format

FEATURE_SAVE_DIR = "features_all_reproduce/"  # Save features here

FEATURE_LOAD_DIR = "features_all/"  # Load features from here when training LightGBM models

DEBUG_CSV_SAVE_DIR = "features_all/"  # Save the first 1000 lines for each feature in debug mode

SHARE_DIR = "share/"  # Feature files shared with teammates

SUBMIT_DIR = "output/"  # Save submission file here

MODEL_DIR = "log/"  # Write confusion matrix, feature importance, oof prediction etc

SUBMIT_FILENAME = "experiment65.csv"

TRAINING_ONLY = 0  # If 1 skip test data handling and make no submission

USE_FIRST_CHUNK_FOR_TEMPLATE_FITTING = True  # (for debug) Use only first 300 objects to make sncosmo features

TSFRESH_N_JOBS = 0  # Number of jobs in tsfresh features

# 'original': reproduce my single model
# 'salt2': remove all template features except salt2 from 'original'
# 'no-template': remove all tepmlate features from 'original'
# 'small': use small number of features (< 50)
# 'best': 'original' + 2 more template features
MODELING_MODE = 'original'

USE_PSEUDO_LABEL = True
