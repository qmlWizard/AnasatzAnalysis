from utils.data_generator import DataGenerator
from utils.model import QNN
import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml
import argparse
import shutil
from collections import namedtuple
import json
import time
import os
import datetime




if __name__ == '__main__':

    "Analysing Quantum Ansatz for detecting defects of PCB's"

    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/test.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    data_generator = DataGenerator( 
                                    dataset_name = config.dataset['name'],
                                    file_path = None if config.dataset['file'] == 'None' else config.dataset['file'],
                                    n_samples= config.dataset['n_samples'],
                                    n_pca_features= config.dataset['pca_features'],
                                    scaler_max= np.pi,
                                    scaler_min= -np.pi
                                  )
    
    features, target = data_generator.generate_dataset()

    print(target)