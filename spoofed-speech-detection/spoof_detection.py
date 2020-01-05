import os
import time
import numpy as np
import pandas as pd
from bob.bio.spear import preprocessor, extractor
from bob.bio.gmm import algorithm
from bob.io.base import HDF5File
from bob.learn import em
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

WAV_FOLDER = 'Wav/' #'ASV2015dataset/wav/' # Path to folder containing speakers .wav subfolders
LABEL_FOLDER = 'CM_protocol/' #'ASV2015dataset/CM_protocol/' # Path to ground truth csv files

EXT = '.wav'

train_subfls = ['T1']#, 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T13']  #T13 used instead of T10 for gender balance
devel_subfls = ['D1']#, 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
evalu_subfls = ['E1']#, 'E2', 'E3', 'E4', 'E5', 'E6','E7',  'E8', 'E9', 'E10']
train = pd.read_csv(LABEL_FOLDER + 'cm_train.trn', sep=' ', header=None, names=['folder','file','method','source'])
if len(train_subfls): train = train[train.folder.isin(train_subfls)]
train.sort_values(['folder', 'file'], inplace=True)
devel = pd.read_csv(LABEL_FOLDER + 'cm_develop.ndx', sep=' ', header=None, names=['folder','file','method','source'])
if len(devel_subfls): devel = devel[devel.folder.isin(devel_subfls)]
devel.sort_values(['folder', 'file'], inplace=True)
evalu = pd.read_csv(LABEL_FOLDER +'cm_evaluation.ndx', sep=' ', header=None, names=['folder','file','method','source'])
if len(evalu_subfls): evalu = evalu[evalu.folder.isin(evalu_subfls)]

evalu.sort_values(['folder', 'file'], inplace=True)

label_2_class = {'human':1, 'spoof':0}

print('training samples:',len(train))
print('development samples:',len(devel))
print('evaluation samples:',len(evalu))