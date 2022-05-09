# save training information
import torch
import os

class Config:
    def __init__(self, experiment):
        self.experiment = experiment
        self.classes = []
        self.num_classes = 0
        
        # model to run
        self.model_name = 'Net6channels'
        # up seed
        self.seed = 42
        # batch size
        self.batch_size = 16
        # num of dataloader workers
        self.num_workers = 4
        # learning rate
        self.lr = 0.0001
        # max epoch to train
        self.max_epoch = 100
        # data path
        self.data_dir = '../data/ptbxl/'
        # result path
        self.result_path = os.path.join('result/', self.experiment)
        # model checkpoint path
        self.checkpoints = os.path.join('checkpoints/', self.model_name)
        # date label path
        self.label_dir = 'data_label/'
        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        #select in between ptbxl tasks
        tasks = {
            'ptb_all': 'all',
            'ptb_diag': 'diagnostic',
            'ptb_diag_sub': 'diagnostic_subclass',
            'ptb_diag_super': 'diagnostic_class',
            'ptb_form': 'form',
            'ptb_rhythm': 'rhythm',
            'CPSC': 'CPSC'
        }
        self.task = tasks[experiment]


        # select ECG leads to use
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # select different sampling frequency file
        self.sampling_frequency = 500
        # select crop length of signal ,10s for ptbxl, 30s for CPSC
        self.length=5000

        if self.experiment == 'CPSC':
            self.data_dir = '../data/CPSC/'
            self.length=6000
            self.batch_size=16
        elif self.experiment == 'ukbiobank':
            self.data_dir = '../data/ukbiobank/'
