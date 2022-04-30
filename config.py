# save training information
import torch
import os
class Config:
    def __init__(self, experiment):
        self.experiment = experiment
        self.seed = 42
        self.batch_size = 64
        self.num_workers = 4
        
        self.model_name = 'lstm'
        self.result_path=os.path.join('result/',self.experiment)

        self.lr = 0.0001

        self.max_epoch = 100

        self.checkpoints = os.path.join('checkpoints/',self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task = {
                'ptb_all': 'all',
                'ptb_diag': 'diagnostic',
                'ptb_diag_sub': 'subdiagnostic',
                'ptb_diag_super': 'superdiagnostic',
                'ptb_form': 'form',
                'ptb_rhythm': 'rhythm',
                'CPSC': 'CPSC'
                }            
        self.task = task[experiment]
        
        self.leads=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', \
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.sampling_frequency = 500

        self.data_dir = '../data/ptbxl/'
        if self.experiment=='CPSC':
            self.data_dir='../data/CPSC/'
        elif self.experiment=='ukbiobank':
            self.data_dir='../data/ukbiobank/'    

        