'''
# Author: Bingyu Mao
# Date: 2021-02-25

# This is the draft for data pre-process. We will create two types of data file for 
# in hospital mortality prediction and survival analysis:

# Case&control files contain diagnosis, procedures and prescriptions;
# Case&control files contain diagnosis, prescriptions, procedures and demographics.
# For each type we will have two files: one is case contains died patients' information and 
# the other is contorl contains other patients' information.

# Note that this project focuses on the encounter level, so we will use HADM_ID to identify 
# every patient's admissions. Since our focus is on mortality, we will use the last ICU admission of every patient.
'''

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

mimic3_path='raw_data'

#read csv from .gz file
def dataframe_from_csv(path, compression='gzip', header=0, index_col=0):
    return pd.read_csv(path, compression=compression, header=header, index_col=index_col)

def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv.gz'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays

def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv.gz'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv.gz'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats

def add_age_to_icustays(stays):
    stays['AGE'] = (stays['INTIME'].subtract(stays['DOB'])).dt.days/365
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays

def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    return stays

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

#Data loading
ad=read_admissions_table(mimic3_path)
pt=read_patients_table(mimic3_path)
st=read_icustays_table(mimic3_path)

#add diagnoses and prescriptions, procedures table is to get ICD9 code
dg=pd.read_csv('raw_data/DIAGNOSES_ICD.csv.gz', compression='gzip', header=0, index_col=0)
pr=pd.read_csv('raw_data/PROCEDURES_ICD.csv.gz', compression='gzip', header=0, index_col=0)
med=pd.read_csv('raw_data/PRESCRIPTIONS.csv.gz', compression='gzip', header=0, index_col=0)

#merge three tables to one
st=merge_on_subject_admission(st, ad)
st=merge_on_subject(st, pt)

#add age
st=add_age_to_icustays(st)
#add mortality
st=add_inhospital_mortality_to_icustays(st)


#Case data with diagnosis, procedures and prescription
cast=st[st['MORTALITY']== 1] #select case data

casti= cast.groupby('SUBJECT_ID')['INTIME'].max().reset_index()
caad=ad[ad['SUBJECT_ID'].isin(cast['SUBJECT_ID'].drop_duplicates().tolist())]
caad1=pd.merge(casti,caad, right_on='SUBJECT_ID', left_on='SUBJECT_ID')
caad2=caad1[pd.to_datetime(caad1['ADMITTIME']) <= caad1['INTIME']]

#We need to add a column on the dataframe (length of stay) to prepare for survival analysis.
st1=cast[['SUBJECT_ID','LOS']].dropna()
st1['LOS']=st1['LOS'].astype(int)
stca=pd.merge(st1,caad2, on='SUBJECT_ID')
stca=stca[pd.to_datetime(stca['ADMITTIME']) <= stca['INTIME']]
stca=stca.drop_duplicates()

#add diagnoses
cadi=dg[(dg['SUBJECT_ID'].isin(stca['SUBJECT_ID'].drop_duplicates().tolist()))&(dg['HADM_ID'].isin(stca['HADM_ID'].drop_duplicates().tolist()))]
cadi=cadi.merge(stca[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
cadi['DISCHTIME']=pd.to_datetime(cadi['DISCHTIME']).dt.date
cadi=cadi[['SUBJECT_ID','ICD9_CODE','DISCHTIME','LOS']].drop_duplicates()
cadi.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
cadi['Event_CODE']='D_'+cadi['Event_CODE']

#add procedures
capr=pr[(pr['SUBJECT_ID'].isin(stca['SUBJECT_ID'].drop_duplicates().tolist()))&(pr['HADM_ID'].isin(stca['HADM_ID'].drop_duplicates().tolist()))]
capr=capr.merge(stca[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
capr['DISCHTIME']=pd.to_datetime(capr['DISCHTIME']).dt.date
capr=capr[['SUBJECT_ID','ICD9_CODE','DISCHTIME','LOS']].drop_duplicates()
capr.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
capr['Event_CODE']='P_'+ capr['Event_CODE'].astype('str')

#add prescriptions for drug name
camd=med[(med['SUBJECT_ID'].isin(stca['SUBJECT_ID'].drop_duplicates().tolist()))]
camd=camd.merge(stca[['SUBJECT_ID','INTIME','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
camd=camd[pd.to_datetime(camd['STARTDATE'].fillna(camd['ENDDATE']))<(camd['INTIME'])]

camd=camd[['SUBJECT_ID','DRUG_NAME_GENERIC','DRUG','DISCHTIME','LOS']].drop_duplicates()
camd['DRUG_NAME_GENERIC']=camd['DRUG_NAME_GENERIC'].fillna(camd['DRUG'])
camd['DISCHTIME']=pd.to_datetime(camd['DISCHTIME']).dt.date
camd=camd[['SUBJECT_ID','DRUG_NAME_GENERIC','DISCHTIME','LOS']].drop_duplicates()
camd.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
camd['Event_CODE']='M_'+camd['Event_CODE']

ca1=pd.concat([cadi,camd,capr])
ca1=ca1.dropna(subset=['Event_CODE']) #drop the rows with null value
ca1.dropna().to_csv('data/case_dp.csv', sep='\t',index=False)

#Case data with diagnosis, prescription, procedures and demographics
#add age to every adimission
cag=st[st['SUBJECT_ID'].isin(cast['SUBJECT_ID'].drop_duplicates().tolist())] 
#make sure to get all admission information of patients (not only the last admission)
cag1=cag[cag['SUBJECT_ID'].isin(stca['SUBJECT_ID'].drop_duplicates().tolist())&(cag['HADM_ID'].isin(stca['HADM_ID'].drop_duplicates().tolist()))] 
#make sure just add patients have admissions
casta=cag1[['SUBJECT_ID','HADM_ID','AGE','DISCHTIME','LOS']]
casta.drop_duplicates()

cast1=casta[['SUBJECT_ID','AGE','DISCHTIME','LOS']]
cast1['DISCHTIME']=pd.to_datetime(cast1['DISCHTIME']).dt.date
cast1['AGE']=cast1['AGE'].astype(int) #make age to be integer
cast1['AGE']=cast1['AGE'].astype(str) #change float to string
cast1.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
cast1['Event_CODE']='A_'+cast1['Event_CODE']

#add ethnicity to every admission
caste=cag1[['SUBJECT_ID','HADM_ID','ETHNICITY','DISCHTIME','LOS']]
caste.drop_duplicates()

castec=caste[~ caste['ETHNICITY'].str.contains('UNKNOWN')]    #drop all the patients for unknown race
castec=castec[~ castec['ETHNICITY'].str.contains('DECLINED')] #drop all the patients declined to tell race

cast2=castec[['SUBJECT_ID','ETHNICITY','DISCHTIME','LOS']]
cast2['DISCHTIME']=pd.to_datetime(cast2['DISCHTIME']).dt.date
cast2.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
cast2['Event_CODE']='E_'+cast2['Event_CODE']

#add gender to every admission
castg=cag1[['SUBJECT_ID','HADM_ID','GENDER','DISCHTIME','LOS']]
castg.drop_duplicates()

cast3=castg[['SUBJECT_ID','GENDER','DISCHTIME','LOS']]
cast3['DISCHTIME']=pd.to_datetime(cast3['DISCHTIME']).dt.date
cast3.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
cast3['Event_CODE']='G_'+cast3['Event_CODE']

ca2=pd.concat([cadi,camd,capr,cast1,cast2,cast3])
ca2['LOS']=ca2['LOS'].astype(int)
ca2=ca2.dropna(subset=['Event_CODE']) #drop the rows with null value
ca2.dropna().to_csv('data/case_dpd.csv', sep='\t',index=False)


#Control data with diagnosis, procedures and prescription
ctst= st[~ (st['SUBJECT_ID'].isin(cast['SUBJECT_ID'].drop_duplicates().tolist()))]
ctsti= ctst.groupby('SUBJECT_ID')['INTIME'].max().reset_index()
ctad=ad[ad['SUBJECT_ID'].isin(ctst['SUBJECT_ID'].drop_duplicates().tolist())]
ctad1=pd.merge(ctsti,ctad, right_on='SUBJECT_ID', left_on='SUBJECT_ID')
ctad2=ctad1[pd.to_datetime(ctad1['ADMITTIME']) <= ctad1['INTIME']]

st2=ctst[['SUBJECT_ID','LOS']].dropna()
st2['LOS']=st2['LOS'].astype(int)
stct=pd.merge(st2,ctad2, on='SUBJECT_ID')
stct=stct[pd.to_datetime(stct['ADMITTIME']) <= stct['INTIME']]
stct=stct.drop_duplicates()

#add diagnoses
ctdi=dg[(dg['SUBJECT_ID'].isin(stct['SUBJECT_ID'].drop_duplicates().tolist()))&(dg['HADM_ID'].isin(stct['HADM_ID'].drop_duplicates().tolist()))]
ctdi=ctdi.merge(stct[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
ctdi['DISCHTIME']=pd.to_datetime(ctdi['DISCHTIME']).dt.date
ctdi=ctdi[['SUBJECT_ID','ICD9_CODE','DISCHTIME','LOS']].drop_duplicates()
ctdi.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctdi['Event_CODE']='D_'+ctdi['Event_CODE']

#add procedures
ctpr=pr[(pr['SUBJECT_ID'].isin(stct['SUBJECT_ID'].drop_duplicates().tolist()))&(pr['HADM_ID'].isin(stct['HADM_ID'].drop_duplicates().tolist()))]
ctpr=ctpr.merge(stct[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
ctpr['DISCHTIME']=pd.to_datetime(ctpr['DISCHTIME']).dt.date
ctpr=ctpr[['SUBJECT_ID','ICD9_CODE','DISCHTIME','LOS']].drop_duplicates()
ctpr.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctpr['Event_CODE']='P_'+ ctpr['Event_CODE'].astype('str')

#add prescriptions for drug name
ctmd=med[(med['SUBJECT_ID'].isin(stct['SUBJECT_ID'].drop_duplicates().tolist()))]
ctmd=ctmd.merge(stct[['SUBJECT_ID','INTIME','HADM_ID','ADMITTIME','DISCHTIME','LOS']].drop_duplicates(),how='left')
ctmd=ctmd[pd.to_datetime(ctmd['STARTDATE'].fillna(ctmd['ENDDATE']))<(ctmd['INTIME'])]  
ctmd=ctmd[['SUBJECT_ID','DRUG_NAME_GENERIC','DRUG','DISCHTIME','LOS']].drop_duplicates()
ctmd['DRUG_NAME_GENERIC']=ctmd['DRUG_NAME_GENERIC'].fillna(ctmd['DRUG'])
ctmd['DISCHTIME']=pd.to_datetime(ctmd['DISCHTIME']).dt.date
ctmd=ctmd[['SUBJECT_ID','DRUG_NAME_GENERIC','DISCHTIME','LOS']].drop_duplicates()
ctmd.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctmd['Event_CODE']='M_'+ctmd['Event_CODE']

ct1=pd.concat([ctdi,ctmd,ctpr])
ct1=ct1.dropna(subset=['Event_CODE']) #drop the rows with null value
ct1.dropna().to_csv('data/ctrl_dp.csv', sep='\t',index=False)


#Control data with diagnosis, prescription, procedures and demographics
#add age to every adimission
ctg=ctst[ctst['SUBJECT_ID'].isin(stct['SUBJECT_ID'].drop_duplicates().tolist())&(ctst['HADM_ID'].isin(stct['HADM_ID'].drop_duplicates().tolist()))] 
ctsta=ctg[['SUBJECT_ID','HADM_ID','AGE','DISCHTIME','LOS']]
ctsta.drop_duplicates()

ctst1=ctsta[['SUBJECT_ID','AGE','DISCHTIME','LOS']]
ctst1['DISCHTIME']=pd.to_datetime(ctst1['DISCHTIME']).dt.date
ctst1['AGE']=ctst1['AGE'].astype(int) #make age to be integer
ctst1['AGE']=ctst1['AGE'].astype(str) #change float to string
ctst1.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctst1['Event_CODE']='A_'+ctst1['Event_CODE']

#add ethnicity to every admission
ctste=ctg[['SUBJECT_ID','HADM_ID','ETHNICITY','DISCHTIME','LOS']]
ctste.drop_duplicates()

ctstec=ctste[~ ctste['ETHNICITY'].str.contains('UNKNOWN')]    #drop all the patients for unknown race
ctstec=ctstec[~ ctstec['ETHNICITY'].str.contains('DECLINED')] #drop all the patients declined to tell race

ctst2=ctstec[['SUBJECT_ID','ETHNICITY','DISCHTIME','LOS']]
ctst2['DISCHTIME']=pd.to_datetime(ctst2['DISCHTIME']).dt.date
ctst2.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctst2['Event_CODE']='E_'+ctst2['Event_CODE']

#add gender to every admission
ctstg=ctg[['SUBJECT_ID','HADM_ID','GENDER','DISCHTIME','LOS']]
ctstg.drop_duplicates()

ctst3=ctstg[['SUBJECT_ID','GENDER','DISCHTIME','LOS']]
ctst3['DISCHTIME']=pd.to_datetime(ctst3['DISCHTIME']).dt.date
ctst3.columns=['SUBJECT_ID','Event_CODE','DISCHTIME','LOS']
ctst3['Event_CODE']='G_'+ctst3['Event_CODE']

ct2=pd.concat([ctdi,ctmd,ctpr,ctst1,ctst2,ctst3])
ct2=ct2.dropna()#drop the rows with null value
ct2['LOS']=ct2['LOS'].astype(int)
ct2.dropna().to_csv('data/ctrl_dpd.csv', sep='\t',index=False)
