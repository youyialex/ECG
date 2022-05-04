import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def cal_train(y_true, y_pred):
    return {'train_auc':roc_auc_score(y_true, y_pred), \
        'train_auc_micro':roc_auc_score(y_true, y_pred, average='micro')}


def cal_test(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return {'test_precision':precision, \
        'test_recall':recall, 'test_f1':f1, 'test_acc':acc}

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
    