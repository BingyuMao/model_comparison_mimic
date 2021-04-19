'''
# Author: Bingyu Mao
# Date: 2021-03-08
# Code dependence: modified from https://github.com/ZhiGroup/pytorch_ehr/blob/master/Preprocessing/data_preprocessing_v1.py

# This script processes originally extracted data and builds pickled lists including a full list that includes all information for case and controls
# Additionally it outputs pickled list of the following shape
#[[pt1_id,label,[
#                  [[delta_time 0],[list of Medical codes in Visit0]],
#                  [[delta_time between V0 and V1],[list of Medical codes in Visit2]],
#                   ......]],
# [pt2_id,label,[[[delta_time 0],[list of Medical codes in Visit0 ]],[[delta_time between V0 and V1],[list of Medical codes in Visit2]],......]]]
#
# for survival the label is a list [event_label,time_to_event]
#
# Usage: feed this script with Case file and Control files each is just a three columns like pt_id | medical_code | visit_date and execute like:
#
# python preprocessing.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> 
# <classification type use 'surv' for survival, else it will be provide single label> 
# <path and prefix to pts file if available,otherwise use 'NA' to build new one>
# you can optionally activate <case_samplesize> <control_samplesize> based on your cohort definition
# This file will later split the data to training , validation and Test sets of ratio
# Output files include
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# Main output files for the baseline RNN models are <output file>.combined
'''

import sys
from optparse import OptionParser
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import glob

if __name__ == '__main__':

    caseFile= sys.argv[1]
    controlFile= sys.argv[2]
    typeFile= sys.argv[3]
    outFile = sys.argv[4]
    cls_type= sys.argv[5]
    pts_file_pre = sys.argv[6]
    parser = OptionParser()
    (options, args) = parser.parse_args()
 
    debug=False
    time_list = []
    dates_list =[]
    label_list = []
    pt_list = []


    print (" Loading cases and controls" ) 

    ## loading Case
    print('loading cases')
    data_case = pd.read_table(caseFile)
    data_case.columns = ["Pt_id", "ICD", "Time","tte"]

    if cls_type=='surv':
        data_case = data_case[["Pt_id", "ICD", "Time","tte"]]
    else:
        data_case = data_case[["Pt_id", "ICD", "Time"]]
    data_case['Label'] = 1
    print('Case counts: ',data_case["Pt_id"].nunique())

    ## loading Control
    print('loading ctrls')
    data_control = pd.read_table(controlFile)
    data_control.columns = ["Pt_id", "ICD", "Time","tte"]

    if cls_type=='surv':
        data_control = data_control[["Pt_id", "ICD", "Time","tte"]]
    else:
        data_control = data_control[["Pt_id", "ICD", "Time"]]
    data_control['Label'] = 0
    print('Ctrl counts: ',data_control["Pt_id"].nunique())

    
    data_l= pd.concat([data_case,data_control])
    print('total counts: ',data_l["Pt_id"].nunique())   

    ## loading the types

    if typeFile=='NA': 
       types={}
    else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)
 
    full_list=[]
    index_date = {}
    time_list = []
    dates_list =[]
    label_list = []
    pt_list = []
    dur_list=[]
    newVisit_list = []
    count=0

    for Pt, group in data_l.groupby('Pt_id'):
            data_i_c = []
            data_dt_c = []
            for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False):
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
            if len(data_i_c) > 0:
                 # creating the duration in days between visits list, first visit marked with 0        
                    v_dur_c=[]
            if len(data_dt_c)<=1:
                     v_dur_c=[0]
            else:
                     for jx in range (len(data_dt_c)):
                        if jx==0:
                             v_dur_c.append(jx)
                        else:
                            xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                            v_dur_c.append(xx)

            ### Diagnosis recoding
            newPatient_c = []
            for visit in data_i_c:
                      newVisit_c = []
                      for code in visit:
                                    if code in types: newVisit_c.append(types[code])
                                    else:                             
                                          types[code] = len(types)+1
                                          newVisit_c.append(types[code])
                      newPatient_c.append(newVisit_c)

            if len(data_i_c) > 0: ## only save non-empty entries

                  if cls_type=='surv':
                      label_list.append([group.iloc[0]['Label'],group.iloc[0]['tte']])
                  else:
                      label_list.append(group.iloc[0]['Label'])

                  pt_list.append(Pt)
                  newVisit_list.append(newPatient_c)
                  dur_list.append(v_dur_c)

            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)


    pickle.dump(types, open(outFile+'.types', 'wb'), -1)
  
    ### Random split to train ,test and validation sets
    print ("Splitting")

    if pts_file_pre=='NA':
        print('random split')
        dataSize = len(pt_list)
        ind = np.random.permutation(dataSize)
        nTest = int(0.2 * dataSize)
        nValid = int(0.1 * dataSize)
        test_indices = ind[:nTest]
        valid_indices = ind[nTest:nTest+nValid]
        train_indices = ind[nTest+nValid:]
    else:
        print ('loading previous splits')
        pt_train=pickle.load(open(pts_file_pre+'.train', 'rb'))
        pt_valid=pickle.load(open(pts_file_pre+'.valid', 'rb'))
        pt_test=pickle.load(open(pts_file_pre+'.test', 'rb'))
        test_indices = np.intersect1d(pt_list, pt_test,assume_unique=True, return_indices=True)[1]
        valid_indices= np.intersect1d(pt_list, pt_valid,assume_unique=True, return_indices=True)[1]
        train_indices= np.intersect1d(pt_list, pt_train,assume_unique=True, return_indices=True)[1]

    for subset in ['train','valid','test']:
        if subset =='train':
            indices = train_indices
        elif subset =='valid':
            indices = valid_indices
        elif subset =='test':
            indices = test_indices
        else: 
            print ('error')
            break 
        
    ### Create the combined list for the Pytorch RNN
    fset=[]
    print ('Reparsing')
    for pt_idx in range(len(pt_list)):
                pt_sk= pt_list[pt_idx]
                pt_lbl= label_list[pt_idx]
                pt_vis= newVisit_list[pt_idx]
                pt_td= dur_list[pt_idx]
                d_gr=[]
                n_seq=[]
                d_a_v=[]
                for v in range(len(pt_vis)):
                        nv=[]
                        nv.append([pt_td[v]])
                        nv.append(pt_vis[v])                   
                        n_seq.append(nv)
                n_pt= [pt_sk,pt_lbl,n_seq]
                fset.append(n_pt)              

    ### split the full combined set to the same as individual files

    train_set_full = [fset[i] for i in train_indices]
    test_set_full = [fset[i] for i in test_indices]
    valid_set_full = [fset[i] for i in valid_indices]
    ctrfilename=outFile+'.combined.train'
    ctstfilename=outFile+'.combined.test'
    cvalfilename=outFile+'.combined.valid'    
    pickle.dump(train_set_full, open(ctrfilename, 'wb'), -1)
    pickle.dump(test_set_full, open(ctstfilename, 'wb'), -1)
    pickle.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
