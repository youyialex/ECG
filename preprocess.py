import ast
from torch import classes
from wandb import Classes
from config import Config
import os
from glob import glob
import wfdb
from tqdm import tqdm
import pandas as pd
import numpy as np
from utility import aggregate_diagnostic, select_diag_form_task


def CPSC_label(data_dir, task):
    recordpaths = glob(os.path.join(data_dir, '*.hea'))
    results = []
    dx_dict = {
        '426783006': 'SNR',  # Normal sinus rhythm
        '164889003': 'AF',  # Atrial fibrillation
        '270492004': 'IAVB',  # First-degree atrioventricular block
        '164909002': 'LBBB',  # Left bundle branch block
        '713427006': 'RBBB',  # Complete right bundle branch block
        '59118001': 'RBBB',  # Right bundle branch block
        '284470004': 'PAC',  # Premature atrial contraction
        '63593006': 'PAC',  # Supraventricular premature beats
        '164884008': 'PVC',  # Ventricular ectopics
        '429622005': 'STD',  # ST-segment depression
        '164931005': 'STE',  # ST-segment elevation
    }
    classes = list(set(dx_dict.values()))
    for recordpath in tqdm(recordpaths):
        patient_id = recordpath.split('/')[-1][:-4]
        _, meta_data = wfdb.rdsamp(recordpath[:-4])
        sample_rate = meta_data['fs']
        length = meta_data['sig_len']
        dx = meta_data['comments'][2]
        dx = dx[4:] if dx.startswith('Dx: ') else ''
        dxs = [dx_dict.get(code, '') for code in dx.split(',')]
        labels = [0] * 9
        for idx, label in enumerate(classes):
            if label in dxs:
                labels[idx] = 1
        results.append([patient_id] + labels +
                       [length, sample_rate, patient_id])
    df = pd.DataFrame(data=results, columns=[
                      'ecg_id']+classes+['length', 'sample_rate', 'path'])
    # assign folder number
    n = len(df)
    folds = np.zeros(n, dtype=np.int8)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        folds[start:end] = i + 1
    df['fold'] = np.random.permutation(folds)
    columns = df.columns
    df['keep'] = df[classes].sum(axis=1)
    df = df[df['keep'] > 0]
    return df[columns]


def ptbxl_label(data_dir, task, sample_freq):
    results = []
    # read ecg info
    database = pd.read_csv(os.path.join(
        data_dir, 'ptbxl_database.csv'))
    database.scp_codes = database.scp_codes.apply(
        lambda x: ast.literal_eval(x))
    # read label dict and classes
    agg_dir = os.path.join(data_dir, 'scp_statements.csv')
    agg_df, classes = select_diag_form_task(agg_dir, task)
    print(classes)
    # select corresponding sample frequency file
    file_path = 'filename_hr'if sample_freq == 500 else 'filename_lr'
    for _, row in tqdm(database.iterrows(),total=database.shape[0]):
        _, meta_data = wfdb.rdsamp(os.path.join(data_dir, row[file_path]))
        sample_rate = meta_data['fs']
        length = meta_data['sig_len']
        labels = [0] * len(classes)
        # convert scp_codes to corresponding classes
        row_classes = aggregate_diagnostic(row['scp_codes'], agg_df, task)
        for idx, label in enumerate(classes):
            if label in row_classes:
                labels[idx] = 1
        # if no label, skip
        if sum(labels) > 0:
            results.append([row['ecg_id']] + labels +
                           [length, sample_rate, row[file_path], row['strat_fold']])
    # generate dataframe
    return pd.DataFrame(data=results, columns=[
        'ecg_id']+classes+['length', 'sample_rate', 'path', 'fold'])


def ukbiobank_label(data_dir, task):
    # TODO:generate label csv
    return data_dir


def preprocess_label(config: Config):
    label_csv = os.path.join(config.label_dir, config.experiment+'.csv')
    print('Preprocessing label...')
    # if label csv exists, read corresponding csv
    if os.path.exists(label_csv):
        label = pd.read_csv(label_csv)
    # if label csv not exists, generate label csv
    elif config.task == 'CPSC':
        label = CPSC_label(config.data_dir, config.task)
    elif config.task == 'ukbiobank':
        label = ukbiobank_label(config.data_dir, config.task)
    else:
        label = ptbxl_label(config.data_dir, config.task,
                            config.sampling_frequency)
    # save csv
    label.to_csv(label_csv, index=None)
    # print diagnose info and return classes
    classes = label.columns.drop(
        ['ecg_id', 'length', 'sample_rate', 'path', 'fold'])
    print('Diagnosis Count:\n', label[classes].sum())
    print('ECG Count:\t', len(label))
    print('Total classes:\t',len(classes))
    return classes.to_list()
