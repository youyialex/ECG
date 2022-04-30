import pandas as pd,numpy as np
from tqdm import tqdm
from config import Config
from torch import nn, optim, seed
import torch, random, os
import utility as util
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
    out_list = []
    label_list = []
    for inputs, label in train_dataloader:
        #transfer data from  cpu to gpu
        inputs ,label= inputs.to(device),label.to(device)
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
        #append result
        output = torch.sigmoid(output)
        out_list.append(output.cpu().detach().numpy())
        label_list.append(label.cpu().detach().numpy())
    # expand result to 2d array    
    label_list = np.vstack(label_list)
    out_list = np.vstack(out_list)
    return {'train_loss':loss_sum / it_count}.\
        update(util.cal_train(label_list, out_list))

def test_epoch(model, criterion, test_dataloader, device):
    model.eval()
    loss_sum, it_count = 0, 0
    out_list = []
    label_list = []
    with torch.no_grad():
        for inputs, label in test_dataloader:
            #transfer data from  cpu to gpu
            inputs, label = inputs.to(device), label.to(device)
            # forward pass
            output = model(inputs)
            # caculate loss
            loss = criterion(output, label)
            # append loss
            loss_sum += loss.item()
            it_count += 1
            #append result
            out_list.append(output.cpu().numpy())
            label_list.append(label.cpu().numpy())
    return {'test_loss':loss_sum / it_count}.\
        update(util.cal_test(label_list, out_list))

def train(config=Config()):
    #initialize seed
    init_seed(config.seed)

    # load datasets
    train_dataloader, val_dataloader, \
    test_dataloader, num_classes = load_datasets(config)

    # load model
    model = getattr(models, config.model_name)(num_classes=num_classes)
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model = model.to(config.device)

    # setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    #setup result path
    result_path=os.path.join(config.result_path,\
        config.model_name +'_b'+
        str(config.batch_size)+'_s'+
        str(config.sampling_frequency)
        + '.csv')

    #>>>>train<<<<
    for epoch in tqdm(range(1, config.max_epoch + 1)):

        train_loss, train_auc, train_TPR = train_epoch(model, optimizer, criterion, train_dataloader, config.device)

        val_loss, val_auc, val_TPR = test_epoch(model, criterion, val_dataloader, config.device)

        test_loss, test_auc, test_TPR = test_epoch(model, criterion, test_dataloader, config.device)

        model_state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
             }, os.path.join(config.checkpoints, 
             config.experiment + '_b'+str(config.batch_size)+'.pth'))

        result_list = [[epoch, train_loss, train_auc, train_TPR,
                        val_loss, val_auc, val_TPR,
                        test_loss, test_auc, test_TPR]]

        dt = pd.DataFrame(result_list)
#TODO: save result
        if(epoch==1):
            dt.to_csv(result_path, index=False)
        else:
            dt.to_csv(result_path, mode='a',index=False,header=False) 

if __name__ =='__main__':
    for experiment in []:
        config=Config(experiment)
        os.makedirs(config.result_path , exist_ok=True)
        os.makedirs(config.checkpoints , exist_ok=True)
        #preprocess label
        os.makedirs(config.label_dir , exist_ok=True)
        preprocess_label(config)
        print('# train device:', config.device)
        print('# batch_size: {}  Current Learning Rate: {}'.format( config.batch_size, config.lr))

        train(config=config)