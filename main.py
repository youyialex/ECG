from unittest import result
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
from torch import nn, optim, seed
import torch
import random
import os
import utility as util
from torch.utils.data import DataLoader
from preprocess import preprocess_label
import models
from dataset import load_datasets


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, optimizer, criterion, train_dataloader, device):
    model.train()
    loss_sum, it_count = 0, 0
    y_pred = []
    y_true = []
    for inputs, label in tqdm(train_dataloader):
        # transfer data from  cpu to gpu
        inputs, label = inputs.to(device), label.to(device)
        # forward pass
        output = model(inputs)
        # caculate loss
        loss = criterion(output, label)
        # zero parameter gradients
        optimizer.zero_grad()
        # caculate gradient
        loss.backward()
        # update parameter weights
        optimizer.step()
        # append loss
        loss_sum += loss.item()
        it_count += 1
        # append result
        output = torch.sigmoid(output)
        y_pred.append(output.cpu().detach().numpy())
        y_true.append(label.cpu().detach().numpy())

    result={'train_loss': loss_sum / it_count}
    result.update(util.cal_train(y_true, y_pred))
    return result


def test_epoch(model, criterion, test_dataloader, device):
    model.eval()
    loss_sum, it_count = 0, 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, label in test_dataloader:
            # transfer data from  cpu to gpu
            inputs, label = inputs.to(device), label.to(device)
            # forward pass
            output = model(inputs)
            # caculate loss
            loss = criterion(output, label)
            # append loss
            loss_sum += loss.item()
            it_count += 1
            # append result
            output = torch.sigmoid(output)
            y_pred.append(output.cpu().numpy())
            y_true.append(label.cpu().numpy())
    
    result={'test_loss': loss_sum / it_count}
    result.update(util.cal_test(y_true, y_pred))
    return result


def train(config: Config):
    # print config info
    print('# train device:', config.device)
    print('# batch_size: {}  Current Learning Rate: {}'.format(
        config.batch_size, config.lr))
    # initialize seed
    init_seed(config.seed)

    # load datasets
    train_dataloader, val_dataloader, test_dataloader = load_datasets(config)

    # load model
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    print('model_name:{}, num_classes={}'.format(
        config.model_name, config.num_classes))
    model = model.to(config.device)

    # setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    # setup result path
    result_path = os.path.join(config.result_path,
                               config.model_name + '_b' +
                               str(config.batch_size)+'_s' +
                               str(config.sampling_frequency)
                               + '.csv')

    print('>>>>training<<<<')
    for epoch in tqdm(range(1, config.max_epoch + 1)):

        train_res = train_epoch(
            model, optimizer, criterion, train_dataloader, config.device)

        val_res = test_epoch(
            model, criterion, val_dataloader, config.device)

        test_res = test_epoch(
            model, criterion, test_dataloader, config.device)

        # save state dict
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(config.checkpoints,
                        config.experiment + '_b'+str(config.batch_size)+'.pth'))

        result = {}
        result.update(train_res)
        print(val_res)
        # result.update(val_res)
        result.update(test_res)
        # save result
        dt = pd.DataFrame(result,index=[epoch])
        if(epoch == 1):
            dt.index.name = 'epoch'
            dt.to_csv(result_path)
        else:
            dt.to_csv(result_path, mode='a', header=False)


if __name__ == '__main__':
    '''
    tasks = {
            'ptb_all': 'all',
            'ptb_diag': 'diagnostic',
            'ptb_diag_sub': 'diagnostic_subclass',
            'ptb_diag_super': 'diagnostic_class',
            'ptb_form': 'form',
            'ptb_rhythm': 'rhythm',
            'CPSC': 'CPSC'
        }
    '''
    for experiment in [
        # 'CPSC',
        'ptb_all','ptb_diag','ptb_diag_sub',
        'ptb_diag_super',
        'ptb_form','ptb_rhythm'
        ]:
        config = Config(experiment)
        # create folders to save result
        os.makedirs(config.result_path, exist_ok=True)
        os.makedirs(config.checkpoints, exist_ok=True)
        os.makedirs(config.label_dir, exist_ok=True)
        # preprocess label
        config.classes = preprocess_label(config)
        config.num_classes = len(config.classes)
        # train with config pareters
        train(config=config)
