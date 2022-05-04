from unittest import result
import pandas as pd,numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,\
    average_precision_score, confusion_matrix, roc_curve, auc

def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)

def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1
    return sum / count

def cal_train(y_true, y_pred):
    # expand result from N*2d to 2d array
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    result={}
    result['train_auc'] = roc_auc_score(y_true, y_pred)
    return result


def cal_test(y_true, y_pred):
    # expand result from N*2d to 2d array
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    result={}
    result['test_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    # result['test_precision'] = precision_score(y_true, y_pred)
    # result['test_recall'] = recall_score(y_true, y_pred)
    # result['test_f1'] = f1_score(y_true, y_pred)
    result['test_mAP'] = compute_mAP(y_true, y_pred)
    result['test_TPR'] = compute_TPR(y_true, y_pred)
    return result


def select_diag_form_task(agg_dir,task):
    agg_df=pd.read_csv(agg_dir, index_col=0)
    if task == 'all':
        agg_df['all'] = agg_df.index
    if task in(['diagnostic_subclass','diagnostic_class']):
        agg_df = agg_df[agg_df['diagnostic'] == 1]
        classes = agg_df[task].unique().tolist()
    else:
        agg_df = agg_df[agg_df[task] == 1]
        classes = agg_df[task].index.unique().tolist()
    return agg_df,classes

def aggregate_diagnostic(y_dic,agg_df,task):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            if task in(['diagnostic_subclass','diagnostic_class']):
                tmp.append(agg_df.loc[key,task])
            else:
                tmp.append(key)
    # string=','.join(list(set(tmp)))  
    return list(set(tmp))
    