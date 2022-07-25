from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import wfdb
from config import Config


def scaling(signal, sigma=0.1):
    scalingFactor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, signal.shape[1]))
    Noise = np.matmul(np.ones((signal.shape[0], 1)), scalingFactor)
    return signal * Noise


def shift(signal, interval=20):
    for col in range(signal.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        signal[:, col] += offset / 1000
    return signal


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    return sig


class ECGDataset(Dataset):
    def __init__(self, config: Config, phase='test', fold=list(range(1, 11))):
        super(ECGDataset, self).__init__()
        if 'exercise' in config.task:
            self.start = config.start
        else:
            self.start = 0
        self.length = config.length
        self.phase = phase
        self.data_dir = config.data_dir
        # read data reference
        data = pd.read_csv(os.path.join(
            config.label_dir, config.experiment+'.csv'))
        # filter [length path and fold]
        self.classes = config.classes
        data = data[data['fold'].isin(fold)]
        self.data = data#.loc[:100]  # subset
        self.task = config.task
        # select leads
        leads = ['I', 'II', 'III', 'aVR', 'aVL',
                 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.leads = np.where(np.in1d(leads, config.leads))[0]
        # label dict to speed up reading file
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        # load data from disk
        if self.task == 'st_feature':
            ecg_data, _ = wfdb.rdsamp(row['path'])
        else:
            ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, row['path']))
        length = self.length
        ecg_data = ecg_data[self.start:self.start+length, self.leads]
        # Transform data
        ecg_data = transform(ecg_data, self.phase == 'train')
        #shape is (length, channels)
        steps = ecg_data.shape[0]
        ecg_data = ecg_data[-length:, :]
        result = np.zeros((length, len(self.leads)))
        result[-steps:, :] = ecg_data
        # get labels
        ecg_id = row['ecg_id']
        if self.label_dict.get(ecg_id) is not None:
            labels = self.label_dict.get(ecg_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[ecg_id] = labels

        if self.task == 'exercise_feature':
            return torch.from_numpy(result.transpose()).float(),\
                row['path']
        else:
            return torch.from_numpy(result.transpose()).float(),\
                torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)


def split_data(seed):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]


def load_datasets(config: Config):
    # shuffle train val test folders
    train_folds, val_folds, test_folds = split_data(seed=config.seed)
    # load datasets based on task
    train_dataset = ECGDataset(config, phase='train', fold=train_folds)
    val_dataset = ECGDataset(config, phase='val', fold=val_folds)
    test_dataset = ECGDataset(config, phase='test', fold=test_folds)
    # return train_dataloader, val_dataloader, test_dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False, num_workers=config.num_workers, pin_memory=True)
    print(f'Leads selection: {config.leads}\nLength: {config.length}')
    return train_dataloader, val_dataloader, test_dataloader


def load_feature_input(config: Config):
    fold = range(1, 11)
    dataset = ECGDataset(config, phase='test', fold=fold)
    test_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True)
    return test_dataloader
