# save training information
import torch
import os


class Config:
    def __init__(self, experiment):
        self.experiment = experiment
        self.classes = []
        self.num_classes = 0
        self.result_path = os.path.join('result/', self.experiment)

        self.model_name = 'resnet34'

        self.seed = 42

        self.batch_size = 64

        self.num_workers = 4

        self.lr = 0.0001

        self.max_epoch = 100

        self.checkpoints = os.path.join('checkpoints/', self.model_name)

        self.label_dir = 'data_label/'

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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

        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.sampling_frequency = 500
        self.length=5000
        self.data_dir = '../data/ptbxl/'
        if self.experiment == 'CPSC':
            self.data_dir = '../data/CPSC/'
            self.length=15000
            self.batch_size=32
        elif self.experiment == 'ukbiobank':
            self.data_dir = '../data/ukbiobank/'
