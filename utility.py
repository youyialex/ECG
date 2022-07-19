from unittest import result
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,\
    average_precision_score, confusion_matrix, roc_curve, auc

def find_threshold(y_true, y_pred):
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_pred > threshold) for threshold in thresholds]
    return thresholds[np.argmax(f1s)]

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
    # expand result from N*(2d array) to (N*2d) array
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    result={}
    result['train_auc'] = roc_auc_score(y_true, y_pred)
    return result


def cal_test(y_true, y_pred):
    # expand result from N*(2d array) to (N*2d) array
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    result={}
    result['test_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    result['test_prc']=average_precision_score(y_true, y_pred)
    # result['test_precision'] = precision_score(y_true, y_pred)
    # result['test_recall'] = recall_score(y_true, y_pred)
    # result['test_f1'] = f1_score(y_true, y_pred)
    result['test_mAP'] = compute_mAP(y_true, y_pred)
    # result['test_TPR'] = compute_TPR(y_true, y_pred)
    return result

def cal_predict(y_true, y_pred, y_score,i,classes):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score,average='macro')
    acc = accuracy_score(y_true, y_pred)
    FPR, pltrecall, _ = roc_curve(y_true, y_score)
    plt.plot(FPR,pltrecall,label=str(classes[i])+' area: %0.2f|' % auc+'num:'+str(int(sum(y_true))))
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('plot/AUROC.png')
    return precision, recall, f1, auc, acc #

def plot_cm(y_trues, y_preds, classes,normalize=True, cmap=plt.cm.Blues):
    plt.close()
    for i, label in enumerate(classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1], yticklabels=[0, 1],
           title=label,
           ylabel='True label, positive num: '+str(int(sum(y_true))),
           xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(f'plot/cm/{label}.png')
        plt.close(fig)

def select_diag_form_task(agg_dir,task):
    agg_df=pd.read_csv(agg_dir, index_col=0)
    if task == 'all':
        agg_df['all'] = 1
    if task in(['diagnostic_subclass','diagnostic_class',]):
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
    return list(set(tmp))
    