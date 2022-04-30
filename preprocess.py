from config import Config 
import os,glob,wfdb
import pandas as pd,numpy as np
def CPSC_label(data_dir,task,label_csv):
    #TODO:generate label csv
    recordpaths = glob(os.path.join(data_dir, '*.hea'))
    results = []
    dx_dict = {
        '426783006': 'SNR', # Normal sinus rhythm
        '164889003': 'AF', # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001': 'RBBB', # Right bundle branch block
        '284470004': 'PAC', # Premature atrial contraction
        '63593006': 'PAC', # Supraventricular premature beats
        '164884008': 'PVC', # Ventricular ectopics
        '429622005': 'STD', # ST-segment depression
        '164931005': 'STE', # ST-segment elevation
    }
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'] 
    for recordpath in recordpaths:
        patient_id = recordpath.split('/')[-1][:-4]
        _, meta_data = wfdb.rdsamp(recordpath[:-4])
        sample_rate = meta_data['fs']
        length = meta_data['sig_len']
        age = meta_data['comments'][0]
        sex = meta_data['comments'][1]
        dx = meta_data['comments'][2]
        age = age[5:] if age.startswith('Age: ') else np.NaN
        sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
        dx = dx[4:] if dx.startswith('Dx: ') else ''
        dxs = [dx_dict.get(code, '') for code in dx.split(',')]
        labels = [0] * 9
        for idx, label in enumerate(classes):
            if label in dxs:
                labels[idx] = 1
        results.append([patient_id]+ labels+ length+ sample_rate)

    df = pd.DataFrame(data=results, columns=['ecg_id']+classes)
    results = []
   
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
        labels = [0] * 9
        for idx, label in enumerate(classes):
            if label in dxs:
                labels[idx] = 1
        results.append([patient_id] + labels)
    df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
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
    df[columns].to_csv(label_csv, index=None)

    return label_csv

def ukbiobank_label(data_dir,task,label_csv):
    #TODO:generate label csv
    return label_csv

def ptbxl_label(data_dir,task,label_csv):
    #TODO:generate label csv

    return label_csv

def preprocess_label(config:Config):
    label_csv=os.path.join(config.label_dir,config.task+'.csv')
    
    if not os.path.exists(label_csv):
        if config.task=='CPSC':
            CPSC_label(config.data_dir,config.task,label_csv)
        elif config.task=='ukbiobank':
            ukbiobank_label(config.data_dir,config.task,label_csv)
        else:
            ptbxl_label(config.data_dir,config.task,label_csv)