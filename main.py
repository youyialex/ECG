import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
from torch import nn, optim
import torch
import random
import os, pickle
import utility as util
import models
from dataset import load_datasets
from preprocess import preprocess_label
from shap_values import shap_values
def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, optimizer, criterion, train_dataloader, device,postfix):
    model.train()
    loss_sum, it_count = 0, 0
    y_pred = []
    y_true = []
    for inputs, label in tqdm(train_dataloader,ncols=100,postfix=postfix):
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

def val_epoch(model, criterion, test_dataloader, device):
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
    result={'val_loss': loss_sum / it_count}
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
    # initialize seed
    init_seed(config.seed)

    # load datasets
    train_dataloader, val_dataloader, test_dataloader = load_datasets(config)

    # load model
    model = getattr(models, config.model_name)(num_classes=config.num_classes, input_channels=len(config.leads))
    model = model.to(config.device)

    # setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.lr,\
    weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # setup result path
    result_path = os.path.join(config.result_path,
                               config.model_name + '_b' +
                               str(config.batch_size)+'_s' +
                               str(config.sampling_frequency)
                               + '.csv')

    print('>>>>Training<<<<')
    postfix=config.model_name + '_'+config.experiment
    for epoch in tqdm(range(1, config.max_epoch + 1),ncols=100,postfix=postfix):

        train_res = train_epoch(
            model, optimizer, criterion, train_dataloader, config.device,postfix)

        val_res = val_epoch(
            model, criterion, val_dataloader, config.device)

        test_res = test_epoch(
            model, criterion, test_dataloader, config.device)

        # save state dict
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, config.checkpoint_path)

        result = {}
        result.update(train_res)
        result.update(val_res)
        result.update(test_res)
        # save result
        dt = pd.DataFrame(result,index=[epoch])
        if(epoch == 1):
            dt.index.name = 'epoch'
            dt.to_csv(result_path)
        else:
            dt.to_csv(result_path, mode='a', header=False)

        #early stopping    
        if val_res['val_loss']>config.last_loss:
            config.trigger_times += 1
            if config.trigger_times >= config.patience:
                print('Early stopping!\nStart to test process.')
                return 
        else:
            config.trigger_times = 0
        config.last_loss = val_res['val_loss']

def predict_find_thresholds(test_dataloader, model, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, y_true = [], []
    for _, (input, label) in enumerate(tqdm(test_dataloader)):
        input, labels = input.to(device), label.to(device)
        output = model(input)
        output = torch.sigmoid(output)
        output_list.append(output.cpu().detach().numpy())
        y_true.append(labels.cpu().detach().numpy())
    y_trues = np.vstack(y_true)
    y_preds = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        threshold = util.find_threshold(y_true, y_pred)
        thresholds.append(threshold)
    pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds

def predict_result(test_loader, net, device, classes,thresholds):
    output_list, label_list = [], []
    for _, (input, label) in enumerate(tqdm(test_loader)):
        input, labels = input.to(device), label.to(device)
        output = net(input)
        output = torch.sigmoid(output)
        output_list.append(output.cpu().detach().numpy())
        label_list.append(labels.cpu().detach().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = [] 
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(util.cal_predict(y_true, y_pred, y_score,i,classes))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    util.plot_cm(y_trues, y_preds,classes)

def predict(config:Config):
    print('>>>>Predicting<<<<')
    # initialize seed
    init_seed(config.seed)
    # load datasets
    train_dataloader, val_dataloader, test_dataloader = load_datasets(config)
    # load model
    model = getattr(models, config.model_name)(num_classes=config.num_classes, input_channels=len(config.leads))
    model = model.to(config.device)
    model.load_state_dict(torch.load(config.checkpoint_path,\
         map_location=config.device)['model_state_dict'])
    model.eval()
    thresholds = predict_find_thresholds(train_dataloader, model, config.device,config.threshold_path)
    print('Thresholds:', thresholds)
    print('Results on validation data:')
    predict_result(val_dataloader, model, config.device,config.classes, thresholds)


if __name__ == '__main__':

    for experiment in [
        # 'CPSC',
        # 'ptb_all',
        # 'ptb_diag',
        # 'ptb_diag_sub'
        # 'ptb_diag_super',
        # 'ptb_form',
        # 'ptb_rhythm',
        # 'ukbb_0',
        'ukbb_st',
    ]:
        # preprocess data
        config = Config(experiment)
        config.classes=preprocess_label(config)
        config.num_classes=len(config.classes)
        # print config info
        print(f'Device: {config.device}')
        print(f'Batch_size: {config.batch_size}  Current Learning Rate: {config.lr}')
        print(f'Model_name:{config.model_name}, Num_classes={config.num_classes}')
        
        # train with config pareters
        train(config=config)

        # predict
        # predict(config=config)

        # shap values
        # shap_values(config=config)
