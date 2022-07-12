from math import ceil
import os

import numpy as np
import pandas as pd
import torch
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import ECGDataset
from config import Config
import models


def plot_shap(ecg_data, sv_data, patient_id, label,leads,length):
    # patient-level interpretation along with raw ECG data
    sv_data_mean = np.mean(sv_data, axis=1)
    nleads=sum(sv_data_mean>6e-4)
    top_leads = (-sv_data_mean).argsort()[:nleads]
    print(patient_id, label, str(nleads))
    if nleads == 0:
        return
    x = range(length)
    ecg_data = ecg_data[:, -length:]
    sv_data = sv_data[:, -length:]
    threshold = 0.003 # set threshold to highlight features with high shap values
    fig, axs = plt.subplots(nleads, figsize=(9, nleads))
    fig.suptitle(label)
    for i, lead in enumerate(top_leads):
        sv_upper = np.ma.masked_where(sv_data[lead] >= threshold, ecg_data[lead])
        sv_lower = np.ma.masked_where(sv_data[lead] < threshold, ecg_data[lead])
        if nleads == 1:
            axe = axs
        else:
            axe = axs[i]
        axe.plot(x, sv_upper, x, sv_lower)
        axe.set_xticks([])
        axe.set_yticks([])
        axe.set_ylabel(leads[lead])
    plt.savefig(f'shap/individuals/{label}-{patient_id}.png')
    plt.close(fig)


def summary_plot(svs, y_scores,leads):
    svs2 = []
    n = y_scores.shape[0]
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        sv_data = svs[label, i]
        svs2.append(np.sum(sv_data, axis=1))
    svs2 = np.vstack(svs2)
    svs_data = np.mean(svs2, axis=0)
    plt.plot(leads, svs_data)
    plt.savefig('shap/summary.png')
    plt.clf()


def plot_shap2(svs, y_scores,classes, leads, cmap=plt.cm.Blues):
    # population-level interpretation
    n = y_scores.shape[0]
    results = [[]for i in range(len(classes))]
    print(svs.shape)
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        results[label].append(svs[label, i])
    ys = []
    for label in range(y_scores.shape[1]):
        result = np.array(results[label])
        y = []
        for i, _ in enumerate(leads):
            y.append(result[:,i].sum())
        y = np.array(y) / np.sum(y)
        ys.append(y)
        plt.plot(leads, y)
    ys.append(np.array(ys).mean(axis=0))
    ys = np.array(ys)
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)
    fmt = '.2f'
    xlabels = leads
    ylabels = classes + ['AVG']
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(j, i, format(ys[i, j], fmt),
                    ha='center', va='center',
                    color='white' if ys[i, j] > thresh else 'black')
    np.set_printoptions(precision=2)
    fig.tight_layout()
    plt.savefig('shap/shap2.png')
    plt.clf()
    

def shap_values(config:Config):
    lleads = np.array(config.leads)
    classes=config.classes
    # select patients to explain
    nsplit=ceil(150/config.num_classes)
    background= nsplit*config.num_classes

    df = pd.read_csv(os.path.join(config.label_dir,config.experiment+'.csv'))
    to_explain=[]
    for i in config.classes:
        to_explain.extend(df[df[i]==1].head(nsplit).index.values)
    to_explain = np.array(list(set(to_explain)))
    print(f'patients to explain: {to_explain.shape[0]}')
    # load datasets
    dataset=ECGDataset(config)
    background_inputs = torch.stack([dataset[input][0] for input in to_explain]).to(config.device)

    result_path = f'shap/{config.experiment}.npy'
    if not os.path.exists(result_path):
        print('>>>>Training Shap Model<<<<<')
        svs = []
        y_scores = []
        # load model
        model = getattr(models, config.model_name)(num_classes=config.num_classes)
        model = model.to(config.device)
        model.load_state_dict(torch.load(config.checkpoint_path,\
            map_location=config.device)['model_state_dict'])
        model.eval()
        # initiate explainer
        e = shap.GradientExplainer(model, background_inputs)

        for patient_id in tqdm(to_explain):
            inputs = torch.stack([dataset[patient_id][0]]).to(config.device)
            y_scores.append(torch.sigmoid(model(inputs)).detach().cpu().numpy())
            sv = np.array(e.shap_values(inputs)) # (n_classes, n_samples, n_leads, n_points)
            svs.append(sv)
        svs = np.concatenate(svs, axis=1)
        y_scores = np.concatenate(y_scores, axis=0)
        np.save(result_path, (svs, y_scores))
    svs, y_scores = np.load(result_path, allow_pickle=True)
    print('>>>>Plotting Shap Values<<<<<')
    summary_plot(svs, y_scores, lleads)
    plot_shap2(svs, y_scores,classes=classes,leads=lleads)

    for i,patient_id in enumerate(to_explain):
        ecg_data = dataset[patient_id][0]
        label_idx = np.argmax(y_scores[i])
        sv_data = svs[label_idx, i]
        plot_shap(ecg_data, sv_data, patient_id, classes[label_idx],lleads,config.length)
