# save training information
import torch
import os

class Config:
    def __init__(self, experiment):
        # model to run
        self.model_name = 'resnet34'
        # up seed
        self.seed = 42
        # batch size
        self.batch_size = 64
        # num of dataloader workers
        self.num_workers = 4
        # learning rate
        self.lr = 0.0001
        # max epoch to train
        self.max_epoch = 100
        # weight decay
        self.weight_decay=0
        # Early stopping
        self.last_loss = 100
        self.patience = 3
        self.trigger_times = 0
        # classes to predict
        self.experiment = experiment
        self.classes = []
        self.num_classes = 0
        # result path
        self.result_path = os.path.join('result/', self.experiment)
        # model checkpoint path
        self.checkpoints = os.path.join('checkpoints/', self.model_name)
        self.checkpoint_path=os.path.join(self.checkpoints,\
            self.experiment + '_b'+str(self.batch_size)+'.pth')
        self.threshold_path = os.path.join('threshholds/',\
            self.experiment +' '+self.model_name + '_b'+str(self.batch_size)+'.pth')
        # date label path
        self.label_dir = 'data_label/'
        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # data path
        self.dirs= {
            'ptbxl':'../data/ptbxl/',
            'ukbiobank':'../data/ukbiobank/',
            'CPSC':'../data/CPSC/',
        }       
        self.data_dir = self.dirs['ptbxl']

        #select in between ptbxl tasks
        self.tasks = {
            'ptb_all': 'all',
            'ptb_diag': 'diagnostic',
            'ptb_diag_sub': 'diagnostic_subclass',
            'ptb_diag_super': 'diagnostic_class',
            'ptb_form': 'form',
            'ptb_rhythm': 'rhythm',
            'CPSC': 'CPSC',
            'ukbb_0':'exercise_0',
            'ukbb_st':'st_feature',
        }
        self.task = self.tasks[experiment]


        # select ECG leads to use
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # select different sampling frequency file
        self.sampling_frequency = 500
        # select crop length of signal ,10s for ptbxl, 30s for CPSC
        self.length=5000

        if self.experiment == 'CPSC':
            self.data_dir = self.dirs['CPSC']
            # self.length=6000
            self.batch_size=32
        elif self.experiment in ['ukbb_0']:
            self.length=5000
            self.sampling_frequency = 500
            self.data_dir = self.dirs['ukbiobank']
            self.leads = ['I',  'II', 'III']
            self.start_rest=int(self.sampling_frequency*405)
            self.start_exercise=int(self.sampling_frequency*150)
        elif self.experiment in ['ukbb_st']:
            self.length=5000
            self.sampling_frequency = 500
            self.data_dir = self.dirs['ukbiobank']
            self.leads = ['I']
            self.start_rest=0#int(self.sampling_frequency*405)
            self.start_exercise=0#int(self.sampling_frequency*150)


        # create folders to save result
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs('threshholds/',exist_ok=True)
        os.makedirs('plot/cm',exist_ok=True)
        os.makedirs('shap/individuals',exist_ok=True)
    