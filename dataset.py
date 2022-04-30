from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd, numpy as np
import os, wfdb

from config import Config
def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    Noise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * Noise

def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig

def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class ECGDataset(Dataset):
    def __init__(self, config:Config, phase, fold):
        super(ECGDataset,self).__init__()
        self.task=config.task
        self.phase=phase
        self.data_dir=config.data_dir
        #read data reference
        data=pd.read_csv(os.path.join(config.label_dir,config.task+'.csv'))
        #filter [length path and fold]
        self.classes= data.columns.tolist()[1:-3]
        data=data[data['fold'].isin(fold)]
        self.data=data
        #select leads
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.leads = np.where(np.in1d(leads, config.leads))[0]
        #label dict to speed up reading file
        self.data_dict = {}
        self.label_dict = {}
        

    def __getitem__(self, index: int):
        row=self.data.iloc[index]
        #load data from disk
        ecg_data,_=wfdb.rdsamp(os.path.join(self.data_dir,row['path']))
        length=int(self.data['length'])*int(self.data['sample_rate'])
        #Transform data
        ecg_data = transform(ecg_data, self.phase == 'train')
        #shape is (length, channels)
        steps=ecg_data.shape[0]
        ecg_data=ecg_data[-length:,self.leads]
        result=np.zeros((length,len(self.leads)))
        result[-steps:,:]=ecg_data
        # get labels
        ecg_id=row['ecg_id']
        if self.label_dict.get(ecg_id):
            labels=self.label_dict.get(ecg_id)
        else:
            labels=row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[ecg_id]=labels
        return torch.from_numpy(result.transpose()).float(),\
                torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)


def split_data(seed=42):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]

def load_datasets(config:Config):
    #shuffle train val test folders
    train_folds, val_folds, test_folds = split_data(seed=config.seed)
    #load datasets based on task
    train_dataset = ECGDataset(config, phase='train', fold=train_folds)
    val_dataset = ECGDataset(config, phase='val', fold=val_folds)
    test_dataset = ECGDataset(config, phase='test', fold=test_folds)
    # return train_dataloader, val_dataloader, test_dataloader  
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader
