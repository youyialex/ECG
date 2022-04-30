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