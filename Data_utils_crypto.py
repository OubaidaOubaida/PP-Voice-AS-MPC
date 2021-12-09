import torch
from torch import Tensor

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import soundfile as sf
from torch.utils import data
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torchvision import datasets, transforms
#import torchvision

import os
import numpy as np
import torch
import torch.nn as nn
#from util_dsp import LinearDCT
from LFCC_pipeline import *


def dct(x, type=2, axis=1, norm='ortho'):
    from scipy.fftpack import dct
    return scipy.fftpack.dct(x=x, type=type, axis=axis, norm=norm)

def split_genSpoof( dir_meta):
    
    d_meta = {}
    attack_id=[]
    utt_text=[]
    utt_file=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    for line in l_meta:
         _, key, att_id, label = line.strip().split(' ')
         attack_id.append(att_id)
         utt_text.append(key)
         d_meta[key] = 1 if label == 'human' else 0
        
    return d_meta,utt_text,attack_id
	
class Dataset_ASVspoof2019_LA(Dataset):
	def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)
               self.nb_time	: integer, the number of timesteps for each mini-batch'''
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            
	def __len__(self):
           
            return len(self.list_IDs)

	def __getitem__(self, index):
            
            self.nb_samp=24000 # 1.5 sec
            
            num_ceps = 30
            low_freq = 0
            high_freq = 8000
            nfilts = 70
            nfft = 1024
            dct_type = 2
            fs=16000
            
            key = self.list_IDs[index]
                
            X, fs = sf.read(self.base_dir+key+'.flac') 
                
            # For fixed length we need to cut or repeat utterance
            nb_time = X.shape[0]
            if nb_time > self.nb_samp:
                X = X[:self.nb_samp]
            else:
                nb_dup = int(self.nb_samp / nb_time) + 1
                X = np.tile(X,(1, nb_dup))[:,:self.nb_samp][0]
                          
            ##------LFCC feature extraction---##
            lfccs = lfcc(sig=X,
                          fs=fs,
                          num_ceps=num_ceps,
                          nfilts=nfilts,
                          nfft=nfft,
                          low_freq=low_freq,
                          high_freq=high_freq,
                          dct_type=dct_type)
            
            
            lfccs=Tensor(lfccs)                             # feature
            y = self.labels[key]                            # target label
            
            return lfccs, y
