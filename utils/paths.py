import os

from utils.directory_helper import DirectoryHelper

root_path = DirectoryHelper.root_path()

# Config yaml file
config_path = os.path.join(root_path, 'config/experiment.yaml')

# Data of the dataset
dataset_path = os.path.join(root_path, 'dataset/data')

# Config of the dataset
dataset_config_path = os.path.join(root_path, 'dataset/config')

# Experiment result csv
csv_save_path = os.path.join(root_path, 'result/csv')

# Experiment setup yaml
exp_info_save_path = os.path.join(root_path, 'result/exp_info')

# Experiment result visualization
analysis_fig_path = os.path.join(root_path, 'result/analysis_fig')

# Experiment result analysis report
analysis_report_path = os.path.join(root_path, 'result/analysis/report.csv')

# Checkpoint
ckpt_save_path = os.path.join(root_path, 'ckpt/')

# Caches
cache_path = os.path.join(root_path, 'cache/')