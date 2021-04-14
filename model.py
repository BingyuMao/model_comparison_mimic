'''
# Author: Bingyu Mao
# Date: 2021-03-28
# Code dependence: https://github.com/ZhiGroup/pytorch_ehr

# This is the draft for RNN model.

'''

##Basics
import pandas as pd
import numpy as np
import sys, random
import math
try:
    import cPickle as pickle
except:
    import pickle
import string
import re
import os
import time
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings("ignore")

## ML and Stats 
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import sklearn.linear_model  as lm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lightgbm import LGBMClassifier
import statsmodels.api as sm
import patsy
from scipy import stats

## DL Framework
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader

###GPU enabling and device allocation
use_cuda = torch.cuda.is_available()


def preprocess(batch,pack_pad): 
    # Check cuda availability
    if use_cuda:
        flt_typ=torch.cuda.FloatTensor
        lnt_typ=torch.cuda.LongTensor
    else: 
        lnt_typ=torch.LongTensor
        flt_typ=torch.FloatTensor
    mb=[]
    mtd=[]
    lbt=[]
    seq_l=[]
    bsize=len(batch) #number of patients in minibatch
    lp= len(max(batch, key=lambda xmb: len(xmb[-1]))[-1]) #maximum number of visits per patients in minibatch
    llv=0
    for x in batch:
        lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
        if llv < lv:
            llv=lv #max number of codes per visit in minibatch        
    for pt in batch:
        sk,label,ehr_seq_l = pt
        lpx=len(ehr_seq_l) #no of visits in pt record
        seq_l.append(lpx) 
        lbt.append(Variable(flt_typ([[float(label)]])))
        ehr_seq_tl=[]
        time_dim=[]
        for ehr_seq in ehr_seq_l:
            pd=(0, (llv -len(ehr_seq[1])))
            result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
            ehr_seq_tl.append(result)
            time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))

        ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
        lpp= lp-lpx #diffence between max seq in minibatch and cnt of patient visits 
        if pack_pad:
            zp= nn.ZeroPad2d((0,0,0,lpp))
        else: 
            zp= nn.ZeroPad2d((0,0,lpp,0))
        ehr_seq_t= zp(ehr_seq_t)
        mb.append(ehr_seq_t)
        time_dim_v= Variable(torch.stack(time_dim,0))
        time_dim_pv= zp(time_dim_v)
        mtd.append(time_dim_pv)
    lbt_t= Variable(torch.stack(lbt,0))
    mb_t= Variable(torch.stack(mb,0)) 
    if use_cuda:
        mb_t.cuda()
    return mb_t, lbt_t,seq_l, mtd

#Dataloader
def my_collate(batch):
    mb_t, lbt_t,seq_l, mtd =preprocess(batch,pack_pad)
    return [mb_t, lbt_t,seq_l, mtd]

class EHRdataloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, packPadMode = False):
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn
        global pack_pad
        pack_pad = packPadMode

class EHRDataset(Dataset):
    def __init__(self, loaded_list, transform=None, sort = True, model='RNN'):
        """
        Args:
            1) loaded_list from pickled file
            2) data should have the format: pickled, 4 layer of lists, a single patient's history should look like this
                [310062,
                 0,
                 [[[0],[7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]],
                  [[66], [590, 596, 153, 8, 30, 11, 10, 240, 20, 175, 190, 15, 7, 5, 183, 62]],
                  [[455],[120, 30, 364, 153, 370, 797, 8, 11, 5, 169, 167, 7, 240, 190, 172, 205, 124, 15]]]]
                 where 310062: patient id, 
                       0: no heart failure
                      [0]: visit time indicator (first one), [7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]: visit codes.                      
            3)transform (optional): Optional transform to be applied on a sample. Data augmentation related. 
            4)test_ratio,  valid_ratio: ratios for splitting the data if needed.
        """
        self.data = loaded_list 
        if sort: 
                self.data.sort(key=lambda pt:len(pt[2]),reverse=True) 
        self.transform = transform 
              
                                     
    def __getitem__(self, idx, seeDescription = False):
        '''
        Return the patient data of index: idx of a 4-layer list 
        patient_id (pt_sk); 
        label: 0 for no, 1 for yes; 
        visit_time: int indicator of the time elapsed from the previous visit, so first visit_time for each patient is always [0];
        visit_codes: codes for each visit.
        '''
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        
        vistc = np.asarray(sample[2])
        desc = {'patient_id': sample[0], 'label': sample[1], 'visit_time': vistc[:,0],'visit_codes':vistc[:,1]}     
        return sample

    def __len__(self):
        return len(self.data)

class EHREmbeddings(nn.Module):
    def __init__(self, input_size, embed_dim ,hidden_size, n_layers=1,dropout_r=0.1,cell_type='GRU', bii=False, time=False , preTrainEmb='', packPadMode = True):
        super(EHREmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.time=time
        self.preTrainEmb=preTrainEmb
        self.packPadMode = packPadMode
        if bii: 
            self.bi=2 
        else: 
            self.bi=1
            
        if len(input_size)==1:
            self.multi_emb=False
            if len(self.preTrainEmb)>0:
                emb_t= torch.FloatTensor(np.asmatrix(self.preTrainEmb))
                self.embed= nn.Embedding.from_pretrained(emb_t)
                self.in_size= embed_dim
            else:
                input_size=input_size[0]
                self.embed= nn.Embedding(input_size, self.embed_dim,padding_idx=0)
                self.in_size= embed_dim
        else:
            if len(input_size)!=3: 
                raise ValueError('the input list is 1 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1

        if self.time: self.in_size= self.in_size+1 
               
        if self.cell_type == "GRU":
            self.cell = nn.GRU
        elif self.cell_type == "RNN":
            self.cell = nn.RNN
        else:
            raise NotImplementedError

        self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers=self.n_layers, dropout= self.dropout_r, bidirectional=bii, batch_first=True)
        self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()

    def EmbedPatients_MB(self,mb_t, mtd):
        self.bsize=len(mb_t)
        embedded = self.embed(mb_t)
        embedded = torch.sum(embedded, dim=2) 
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: 
                mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb

class EHR_RNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers=1,dropout_r=0.1,cell_type='GRU',bii=False ,time=False, preTrainEmb='',packPadMode = True):
        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, 
                               bii=bii, time=time , preTrainEmb=preTrainEmb, packPadMode=packPadMode)

    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def init_hidden(self):
        
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        if use_cuda: 
            h_0.cuda()
            
        result = h_0
        return result
    
    def forward(self, input, x_lens, mtd):
        x_in  = self.EmbedPatient_MB(input, mtd) 
        if self.packPadMode: 
            x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)   
            output, hidden = self.rnn_c(x_inp) 
        else:
            output, hidden = self.rnn_c(x_in) 
        
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze()

#major model training utilities
def trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = nn.BCELoss()): 
    model.train()
    model.zero_grad()
    output = model(sample,seq_l, mtd)   
    loss = criterion(output, label_tensor)    
    loss.backward()   
    optimizer.step()
    return output, loss.item()

#train with loaders
def trainbatches(mbs_list, model, optimizer, shuffle = True): 
    model.train()
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch

        output, loss = trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = nn.BCELoss())
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
    return current_loss, all_losses 

def calculate_auc(model, mbs_list, which_model = 'RNN', shuffle = True):
    model.eval()
    y_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        output = model(sample, seq_l, mtd)
        y_hat.extend(output.cpu().data.view(-1).numpy())  
        y_real.extend(label_tensor.cpu().data.view(-1).numpy())
         
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat 

#for tracking computational timing
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


#RNN Model training 
def run_dl_model(ehr_model,train_sl,valid_sl,test_sl):
    ## Data Loading
    print (' creating the list of training minibatches')
    print(EHRDataset)
    train = EHRDataset(train_sl,sort= True, model='RNN')
    train_mbs = list(tqdm(EHRdataloader(train, batch_size = 128, packPadMode = True)))
    print (' creating the list of valid minibatches')
    valid = EHRDataset(valid_sl,sort= True, model='RNN')
    valid_mbs = list(tqdm(EHRdataloader(valid, batch_size = 128, packPadMode = True)))
    print (' creating the list of test minibatches')
    test = EHRDataset(test_sl,sort= True, model='RNN')
    test_mbs = list(tqdm(EHRdataloader(test, batch_size = 128, packPadMode = True, shuffle = False)))


    ##Hyperparameters -- Fixed for testing purpose
    epochs = 100
    l2 = 0.0001
    lr = 0.01
    eps = 1e-4
    w_model='RNN'
    optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)   

    ##Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep =[]
    valid_auc_allep =[]
    test_auc_allep=[]  
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(train_mbs, model= ehr_model, optimizer = optimizer)
        avg_loss = np.mean(train_loss)
        train_time = timeSince(start)
        eval_start = time.time()
        Train_auc, y_real_train, y_hat_train  = calculate_auc(ehr_model, train_mbs, which_model = w_model)
        valid_auc, y_real_valid, y_hat_valid  = calculate_auc(ehr_model, valid_mbs, which_model = w_model)
        TestAuc, y_real_test, y_hat_test = calculate_auc(ehr_model, test_mbs, which_model = w_model, shuffle = False) # make sure don't shuffle test set
        eval_time = timeSince(eval_start)
        print ("Epoch: " ,str(ep) ," Train_auc :" , str(Train_auc) , " , Valid_auc : " ,str(valid_auc) ,
               " ,& Test_auc : " , str(TestAuc) ," Avg Loss: " ,str(avg_loss),
               ' , Train Time :' , str(train_time) ,' ,Eval Time :' ,str(eval_time))
        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)
        test_auc_allep.append(TestAuc)

        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAuc
            #save the best model parameters
            best_model = ehr_model

        if ep - bestValidEpoch > 10: break
    print( 'bestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch))
    return train_auc_allep,valid_auc_allep,test_auc_allep


def calculate_cindex(model, mbs_list, which_model = 'RNN', shuffle = True):
    model.eval() 
    e_real =[]
    d_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        output = model(sample, seq_l, mtd)
        y_hat.extend(output.cpu().data.view(-1).numpy()*-1)  
        e, d = label_tensor.squeeze().T
        d_real.extend(d.cpu().data.view(-1).numpy())
        e_real.extend(e.cpu().data.view(-1).numpy())
       
    c_index = concordance_index(d_real, y_hat,e_real)
    return c_index, (d_real,e_real), y_hat

def cox_ph_loss(log_h, label, eps=1e-7):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    """
    events, durations = label.squeeze().T
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

def cox_ph_loss_sorted(log_h, events, eps = 1e-7):
    """Requires the input to be sorted by descending duration time.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

#Survival model training 
def run_surv_model(ehr_model,train_sl,valid_sl,test_sl,wmodel,packpadmode,surv_model):
    ## Data Loading
    print (' creating the list of training minibatches')
    train = EHRDataset(train_sl,sort= True, model='RNN')
    train_mbs = list(tqdm(EHRdataloader(train, batch_size = 256, packPadMode = packpadmode, surv=surv_model )))
    print (' creating the list of valid minibatches')
    valid = EHRDataset(valid_sl,sort= True, model='RNN')
    valid_mbs = list(tqdm(EHRdataloader(valid, batch_size = 256, packPadMode = packpadmode , surv=surv_model)))
    print (' creating the list of test minibatches')
    test = EHRDataset(test_sl,sort= True, model='RNN')
    test_mbs = list(tqdm(EHRdataloader(test, batch_size = 256, packPadMode = packpadmode , surv=surv_model)))

    #Hyperparameters -- Fixed for testing purpose
    epochs = 100
    l2 = 0.0001
    lr = 0.01
    eps = 1e-4
    w_model= wmodel
    optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)   

    #Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep =[]
    valid_auc_allep =[]
    test_auc_allep=[]  
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(train_mbs, model= ehr_model, optimizer = optimizer , loss_fn=cox_ph_loss)
        avg_loss = np.mean(train_loss)
        train_time = timeSince(start)
        eval_start = time.time()
        Train_auc, y_real, y_hat  = calculate_cindex(ehr_model, train_mbs, which_model = w_model)
        valid_auc, y_real, y_hat  = calculate_cindex(ehr_model, valid_mbs, which_model = w_model)
        TestAuc, y_real, y_hat = calculate_cindex(ehr_model, test_mbs, which_model = w_model)
        eval_time = timeSince(eval_start)
        print ("Epoch: " ,str(ep) ," Train_cindex :" , str(Train_auc) , " , Valid_cindex : " ,str(valid_auc) ,
               " ,& Test_cindex : " , str(TestAuc) ," Avg Loss: " ,str(avg_loss),
               ' , Train Time :' , str(train_time) ,' ,Eval Time :' ,str(eval_time))
        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)
        test_auc_allep.append(TestAuc)

        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAuc
     
        if ep - bestValidEpoch >10: break
    print( 'bestValidCindex %f has a TestCindex of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch))
    return train_auc_allep,valid_auc_allep,test_auc_allep