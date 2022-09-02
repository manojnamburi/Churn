from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_theme(style="whitegrid")
from sklearn.model_selection import train_test_split
import time
import pickle
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1
import string
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler,\
                                   Binarizer, KBinsDiscretizer, QuantileTransformer, PowerTransformer,\
                                   PolynomialFeatures, OneHotEncoder, OrdinalEncoder)


# To Connect GCP

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1


class bqConnect:
    
    def __init__(self, cred_json, project_id):
        self.cred_json = cred_json
        self.project_id = project_id
        credentials = service_account.Credentials.from_service_account_file(self.cred_json)
        self.client = bigquery.Client(credentials= credentials, project=self.project_id)
    
    def dry_run(self, query):
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False) 

        query_job = self.client.query(
            (query),
            job_config=job_config,
        )  # Make an API request.
        # A dry run query completes immediately.
        return ("This query will process {} GB.".format(query_job.total_bytes_processed/1000000000))
    
    def fetch_data(self, query):
        return self.client.query(query).to_dataframe()


        

def reduce_cat_cols_levels(df, cat_cols, num_cols):
    
    x = df.copy()
    

    print('ebill_status_desc ======================') 
    print('Before : ', x['ebill_status_desc'].unique())    
    x.loc[(x['ebill_status_desc'].str.upper() ==  'EBILL AND PRINT STATEMENT') | (x['ebill_status_desc'].str.upper() ==  'EBILL ONLY'), 'ebill_status_desc'] = 'Ebill'    
    print('After  : ', x['ebill_status_desc'].unique())
        
    print('\nagent_last_dept ======================') 
    print('Before : ', x['agent_last_dept'].unique()) 
    x.loc[(x['agent_last_dept'].str.contains("retention", case=False)), 'agent_last_dept'] = 'Retention'   
    x.loc[(x['agent_last_dept'].str.contains("sales", case=False)), 'agent_last_dept'] = 'Sales' 
    print('After  : ', x['agent_last_dept'].unique()) 
       
    #------------------------------------------------------   
    print('\n ivr_last_dept ======================') 
    print('Before : ', x['ivr_last_dept'].unique()) 
    x.loc[(x['ivr_last_dept'].str.contains("retention", case=False)), 'ivr_last_dept'] = 'Retention'   
    x.loc[(x['ivr_last_dept'].str.contains("sales", case=False)), 'ivr_last_dept'] = 'Sales' 
    x.loc[(x['ivr_last_dept'].str.upper() ==  'UNKNOWN'), 'ivr_last_dept'] = 'None' 
    print('After  : ', x['ivr_last_dept'].unique())
    
    print('\n ivr_last_interactionReason ======================') 
    print('Before : ', x['ivr_last_interactionReason'].unique()) 
    x.loc[(x['ivr_last_interactionReason'].str.upper() !=  'DATA') & (x['ivr_last_interactionReason'].str.upper() !=  'BILLING') & (x['ivr_last_interactionReason'].str.upper() !=  'VIDEO') & (x['ivr_last_interactionReason'].str.upper() !=  'GENERAL') & \
        (x['ivr_last_interactionReason'].str.upper() !=  'RETENTION') & (x['ivr_last_interactionReason'].str.upper() !=  'SALES') & (x['ivr_last_interactionReason'].str.upper() !=  'APPOINTMENT') & (x['ivr_last_interactionReason'].str.upper() !=  'NONE'), 'ivr_last_interactionReason'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_last_interactionReason'].unique())
    
    print('\n ivr_last_ivrInteractionEndResult ======================') 
    print('Before : ', x['ivr_last_ivrInteractionEndResult'].unique()) 
    x.loc[(x['ivr_last_ivrInteractionEndResult'].str.upper() ==  'NONE'), 'ivr_last_ivrInteractionEndResult'] = 'Success'    
    print('After  : ', x['ivr_last_ivrInteractionEndResult'].unique())
    
    #-------------------------------------------------------
    print('\n ivr_repeat_1_prevInteractionReason ======================') 
    print('Before : ', x['ivr_repeat_1_prevInteractionReason'].unique()) 
    x.loc[(x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'BILLING') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'DATA') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'VIDEO') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'GENERAL') & \
        (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'RETENTION') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'SALES') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'APPOINTMENT') & (x['ivr_repeat_1_prevInteractionReason'].str.upper() !=  'NONE'), 'ivr_repeat_1_prevInteractionReason'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_1_prevInteractionReason'].unique())
    
    print('\n ivr_repeat_2_prevInteractionReason ======================') 
    print('Before : ', x['ivr_repeat_2_prevInteractionReason'].unique()) 
    x.loc[(x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'BILLING') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'DATA') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'VIDEO') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'GENERAL') & \
        (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'RETENTION') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'SALES') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'APPOINTMENT') & (x['ivr_repeat_2_prevInteractionReason'].str.upper() !=  'NONE'), 'ivr_repeat_2_prevInteractionReason'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_2_prevInteractionReason'].unique())
    
    print('\n ivr_repeat_3_prevInteractionReason ======================') 
    print('Before : ', x['ivr_repeat_3_prevInteractionReason'].unique()) 
    x.loc[(x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'BILLING') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'DATA') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'VIDEO') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'GENERAL') & \
        (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'RETENTION') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'SALES') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'APPOINTMENT') & (x['ivr_repeat_3_prevInteractionReason'].str.upper() !=  'NONE'), 'ivr_repeat_3_prevInteractionReason'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_3_prevInteractionReason'].unique())
    
    #----------------------------------------------------------
    print('\n ivr_repeat_1_prevDept ======================') 
    print('Before : ', x['ivr_repeat_1_prevDept'].unique()) 
    x.loc[(x['ivr_repeat_1_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_1_prevDept'] = 'None' 
    x.loc[(x['ivr_repeat_1_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_1_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_1_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_1_prevDept'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_1_prevDept'].unique())
    
    print('\n ivr_repeat_2_prevDept ======================') 
    print('Before : ', x['ivr_repeat_2_prevDept'].unique()) 
    x.loc[(x['ivr_repeat_2_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_2_prevDept'] = 'None' 
    x.loc[(x['ivr_repeat_2_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_2_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_2_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_2_prevDept'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_2_prevDept'].unique())
    
    print('\n ivr_repeat_3_prevDept ======================') 
    print('Before : ', x['ivr_repeat_3_prevDept'].unique()) 
    x.loc[(x['ivr_repeat_3_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_3_prevDept'] = 'None' 
    x.loc[(x['ivr_repeat_3_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_3_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_3_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_3_prevDept'] = 'ALL_OTHER'    
    print('After  : ', x['ivr_repeat_3_prevDept'].unique())
    
     # ---------------------------------------------------------------------------
    print('\n speed_m0 ======================') 
    print('Before : ', x['speed_m0'].unique()) 
    x['speed_m0'] = x['speed_m0'].str.replace('OPTIMUM ','')
    x['speed_m0'] = x['speed_m0'].str.replace('BROADBAND INTERNET ','')
    x['speed_m0'] = x['speed_m0'].str.replace('None','0')
    x['speed_m0'] = x['speed_m0'].str.replace('OOL','25')
    x['speed_m0'] = x['speed_m0'].str.replace('GBPS','000')
    x['speed_m0'] = x['speed_m0'].str.split('/').str[0]
    x['speed_m0'] = x['speed_m0'].str.split('-').str[0]
    x['speed_m0'] = x['speed_m0'].astype('float')
    print('After  : ', x['speed_m0'].unique())
    
    
    print('speed_m1 ======================') 
    print('Before : ', x['speed_m1'].unique()) 
    x['speed_m1'] = x['speed_m1'].str.replace('OPTIMUM ','')
    x['speed_m1'] = x['speed_m1'].str.replace('BROADBAND INTERNET ','')
    x['speed_m1'] = x['speed_m1'].str.replace('None','0')
    x['speed_m1'] = x['speed_m1'].str.replace('OOL','25')
    x['speed_m1'] = x['speed_m1'].str.replace('GBPS','000')
    x['speed_m1'] = x['speed_m1'].str.split('/').str[0]
    x['speed_m1'] = x['speed_m1'].str.split('-').str[0]
    x['speed_m1'] = x['speed_m1'].astype('float')
    print('After  : ', x['speed_m1'].unique())
    
    
    print('speed_m2 ======================') 
    print('Before : ', x['speed_m2'].unique()) 
    x['speed_m2'] = x['speed_m2'].str.replace('OPTIMUM ','')
    x['speed_m2'] = x['speed_m2'].str.replace('BROADBAND INTERNET ','')
    x['speed_m2'] = x['speed_m2'].str.replace('None','0')
    x['speed_m2'] = x['speed_m2'].str.replace('OOL','25')
    x['speed_m2'] = x['speed_m2'].str.replace('GBPS','000')
    x['speed_m2'] = x['speed_m2'].str.split('/').str[0]
    x['speed_m2'] = x['speed_m2'].str.split('-').str[0]
    x['speed_m2'] = x['speed_m2'].astype('float')
    print('After  : ', x['speed_m2'].unique())

    
    print('speed_m3 ======================') 
    print('Before : ', x['speed_m3'].unique()) 
    x['speed_m3'] = x['speed_m3'].str.replace('OPTIMUM ','')
    x['speed_m3'] = x['speed_m3'].str.replace('BROADBAND INTERNET ','')
    x['speed_m3'] = x['speed_m3'].str.replace('None','0')
    x['speed_m3'] = x['speed_m3'].str.replace('OOL','25')
    x['speed_m3'] = x['speed_m3'].str.replace('GBPS','000')
    x['speed_m3'] = x['speed_m3'].str.split('/').str[0]
    x['speed_m3'] = x['speed_m3'].str.split('-').str[0]
    x['speed_m3'] = x['speed_m3'].astype('float')
    print('After  : ', x['speed_m3'].unique())
    
    
    print('\n speed_m4 ======================') 
    print('Before : ', x['speed_m4'].unique()) 
    x['speed_m4'] = x['speed_m4'].str.replace('OPTIMUM ','')
    x['speed_m4'] = x['speed_m4'].str.replace('BROADBAND INTERNET ','')
    x['speed_m4'] = x['speed_m4'].str.replace('None','0')
    x['speed_m4'] = x['speed_m4'].str.replace('OOL','25')
    x['speed_m4'] = x['speed_m4'].str.replace('GBPS','000')   
    x['speed_m4'] = x['speed_m4'].str.split('/').str[0]
    x['speed_m4'] = x['speed_m4'].str.split('-').str[0] 
    x['speed_m4'] = x['speed_m4'].astype('float')
    print('After  : ', x['speed_m4'].unique())
    
    
    x['speed_max_m1m4'] = x[["speed_m1", "speed_m2", "speed_m3", "speed_m4"]].max(axis=1)
    x['speed_change_m4m0'] = np.where(x['speed_m0'] == x['speed_max_m1m4'], 'same' , (np.where(x['speed_m0'] > x['speed_max_m1m4'], 'higher', 'lower')))
    x = x.drop(['speed_m1', 'speed_m2', 'speed_m3','speed_m4','speed_max_m1m4'],axis=1)

    
    x['speed_m0'] = x['speed_m0'].astype('string')
    x["speed_m0"] = x["speed_m0"].str[:-2]
    x.loc[(x['speed_m0'].str.upper() !=  '0') & (x['speed_m0'].str.upper() !=  '25') & (x['speed_m0'].str.upper() !=  '30') & (x['speed_m0'].str.upper() !=  '50') & (x['speed_m0'].str.upper() !=  '100') & (x['speed_m0'].str.upper() !=  '200') & (x['speed_m0'].str.upper() !=  '300') & (x['speed_m0'].str.upper() !=  '400') & (x['speed_m0'].str.upper() !=  '500') & (x['speed_m0'].str.upper() !=  '1000'), 'speed_m0'] = 'ALL_OTHER'
    print("dropped speed_m1 to speed_m4 and created speed_change_m4m0")
    print(x['speed_m0'].unique())
    print(x['speed_change_m4m0'].unique())
    
    #--------------------------------------------------------------
    print('\n max_svod_m1 to max_svod_m3')
    x["max_svod_m1"] = np.where((x["max_svod_m1"].str.upper() == "Y"), 1, 0)
    x["max_svod_m2"] = np.where((x["max_svod_m2"].str.upper() == "Y"), 1, 0)
    x["max_svod_m3"] = np.where((x["max_svod_m3"].str.upper() == "Y"), 1, 0)
    x['max_svod_m1'] = x['max_svod_m1'].astype('int32')
    x['max_svod_m2'] = x['max_svod_m2'].astype('int32')
    x['max_svod_m3'] = x['max_svod_m3'].astype('int32')

    x['max_svod_max_m2m3'] = x[["max_svod_m2", "max_svod_m3"]].max(axis=1)
    x['max_svod_change_m3m1'] = np.where(x['max_svod_m1'] == x['max_svod_max_m2m3'], 'same' , (np.where(x['max_svod_m1'] > x['max_svod_max_m2m3'], 'up', 'down')))
    x = x.drop(['max_svod_m1', 'max_svod_m2', 'max_svod_m3', 'max_svod_max_m2m3'],axis=1)
    print(x['max_svod_change_m3m1'].unique())
    print('Dropped max_svod_m1 to max_svod_m3 and created max_svod_change_m3m1')

    #----------------------------------------------------------------
    print('\n hbo_svod_new_m1 to hbo_svod_new_m3')
    x["hbo_svod_new_m1"] = np.where((x["hbo_svod_new_m1"].str.upper() == "Y"), 1, 0)
    x["hbo_svod_new_m2"] = np.where((x["hbo_svod_new_m2"].str.upper() == "Y"), 1, 0)
    x["hbo_svod_new_m3"] = np.where((x["hbo_svod_new_m3"].str.upper() == "Y"), 1, 0)
    x['hbo_svod_new_m1'] = x['hbo_svod_new_m1'].astype('int32')
    x['hbo_svod_new_m2'] = x['hbo_svod_new_m2'].astype('int32')
    x['hbo_svod_new_m3'] = x['hbo_svod_new_m3'].astype('int32')

    x['hbo_svod_max_m2m3'] = x[["hbo_svod_new_m2", "hbo_svod_new_m3"]].max(axis=1)
    x['hbo_svod_change_m3m1'] = np.where(x['hbo_svod_new_m1'] == x['hbo_svod_max_m2m3'], 'same' , (np.where(x['hbo_svod_new_m1'] > x['hbo_svod_max_m2m3'], 'up', 'down')))
    x = x.drop(['hbo_svod_new_m1', 'hbo_svod_new_m2', 'hbo_svod_new_m3', 'hbo_svod_max_m2m3'],axis=1)
    print(x['hbo_svod_change_m3m1'].unique())
    print('Dropped hbo_svod_new_m1 to hbo_svod_new_m3 and created hbo_svod_change_m3m1')

    #----------------------------------------------------------------
    print('\n stz_enc_svod_m1 to stz_enc_svod_m3')
    x["stz_enc_svod_m1"] = np.where((x["stz_enc_svod_m1"].str.upper() == "Y"), 1, 0)
    x["stz_enc_svod_m2"] = np.where((x["stz_enc_svod_m2"].str.upper() == "Y"), 1, 0)
    x["stz_enc_svod_m3"] = np.where((x["stz_enc_svod_m3"].str.upper() == "Y"), 1, 0)
    x['stz_enc_svod_m1'] = x['stz_enc_svod_m1'].astype('int32')
    x['stz_enc_svod_m2'] = x['stz_enc_svod_m2'].astype('int32')
    x['stz_enc_svod_m3'] = x['stz_enc_svod_m3'].astype('int32')

    x['stz_enc_svod_max_m2m3'] = x[["stz_enc_svod_m2", "stz_enc_svod_m3"]].max(axis=1)
    x['stz_enc_svod_change_m3m1'] = np.where(x['stz_enc_svod_m1'] == x['stz_enc_svod_max_m2m3'], 'same' , (np.where(x['stz_enc_svod_m1'] > x['stz_enc_svod_max_m2m3'], 'up', 'down')))
    x = x.drop(['stz_enc_svod_m1', 'stz_enc_svod_m2', 'stz_enc_svod_m3', 'stz_enc_svod_max_m2m3'],axis=1)
    print(x['stz_enc_svod_change_m3m1'].unique())
    print('Dropped stz_enc_svod_m1 to stz_enc_svod_m3 and created stz_enc_svod_change_m3m1')

    #----------------------------------------------------------------
    print('\n baserev_m0 to baserev_m6')
    x[["baserev_m0","baserev_m1", "baserev_m2",'baserev_m3','baserev_m4','baserev_m5','baserev_m6']] = x[["baserev_m0","baserev_m1", "baserev_m2",'baserev_m3','baserev_m4','baserev_m5','baserev_m6']].astype('str').replace('None','0').astype('float64')
    x['baserev_max_m1m6'] = x[["baserev_m1", "baserev_m2",'baserev_m3','baserev_m4','baserev_m5','baserev_m6']].max(axis=1)
    x['baserev_change_m6m0'] = np.where(x['baserev_m0'] == x['baserev_max_m1m6'], 'same' , (np.where(x['baserev_m0'] > x['baserev_max_m1m6'], 'up', 'down')))
    x = x.drop(['baserev_max_m1m6'],axis=1)
    print('created baserev_change_m6m0')

    #---------------------------------------------------------------
    print('\n curr_video_tier_desc_m0 ======================') 
    print('Before : ', x['curr_video_tier_desc_m0'].unique()) 
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("basic", case=False)), 'curr_video_tier_desc_m0'] = 'BASIC'   
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("economy", case=False)), 'curr_video_tier_desc_m0'] = 'ECONOMY' 
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("core", case=False)), 'curr_video_tier_desc_m0'] = 'CORE'   
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("value", case=False)), 'curr_video_tier_desc_m0'] = 'VALUE' 
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("select", case=False)), 'curr_video_tier_desc_m0'] = 'SELECT'   
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("premier", case=False)), 'curr_video_tier_desc_m0'] = 'PREMIER' 
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("free", case=False)), 'curr_video_tier_desc_m0'] = 'FREE'   
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("bulk", case=False)), 'curr_video_tier_desc_m0'] = 'BULK'  
    x.loc[(x['curr_video_tier_desc_m0'].str.contains("family", case=False)), 'curr_video_tier_desc_m0'] = 'FAMILY' 
    x.loc[(x['curr_video_tier_desc_m0'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m0'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m0'].str.upper() !=  'CORE') \
          & (x['curr_video_tier_desc_m0'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m0'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m0'].str.upper() !=  'PREMIER') \
          # & (x['curr_video_tier_desc_m0'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m0'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m0'].str.upper() !=  'FAMILY') \
          & (x['curr_video_tier_desc_m0'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m0'] = 'ALL_OTHER'
    print('After  : ', x['curr_video_tier_desc_m0'].unique())


    print('curr_video_tier_desc_m1 ======================') 
    print('Before : ', x['curr_video_tier_desc_m1'].unique()) 
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("basic", case=False)), 'curr_video_tier_desc_m1'] = 'BASIC'   
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("economy", case=False)), 'curr_video_tier_desc_m1'] = 'ECONOMY' 
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("core", case=False)), 'curr_video_tier_desc_m1'] = 'CORE'   
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("value", case=False)), 'curr_video_tier_desc_m1'] = 'VALUE' 
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("select", case=False)), 'curr_video_tier_desc_m1'] = 'SELECT'   
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("premier", case=False)), 'curr_video_tier_desc_m1'] = 'PREMIER' 
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("free", case=False)), 'curr_video_tier_desc_m1'] = 'FREE'   
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("bulk", case=False)), 'curr_video_tier_desc_m1'] = 'BULK'  
    x.loc[(x['curr_video_tier_desc_m1'].str.contains("family", case=False)), 'curr_video_tier_desc_m1'] = 'FAMILY' 
    x.loc[(x['curr_video_tier_desc_m1'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m1'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m1'].str.upper() !=  'CORE') \
          & (x['curr_video_tier_desc_m1'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m1'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m1'].str.upper() !=  'PREMIER') \
          # & (x['curr_video_tier_desc_m1'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m1'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m1'].str.upper() !=  'FAMILY') \
                         & (x['curr_video_tier_desc_m1'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m1'] = 'ALL_OTHER'
    print('After  : ', x['curr_video_tier_desc_m1'].unique())


    print('curr_video_tier_desc_m2 ======================') 
    print('Before : ', x['curr_video_tier_desc_m2'].unique()) 
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("basic", case=False)), 'curr_video_tier_desc_m2'] = 'BASIC'   
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("economy", case=False)), 'curr_video_tier_desc_m2'] = 'ECONOMY' 
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("core", case=False)), 'curr_video_tier_desc_m2'] = 'CORE'   
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("value", case=False)), 'curr_video_tier_desc_m2'] = 'VALUE' 
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("select", case=False)), 'curr_video_tier_desc_m2'] = 'SELECT'   
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("premier", case=False)), 'curr_video_tier_desc_m2'] = 'PREMIER' 
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("free", case=False)), 'curr_video_tier_desc_m2'] = 'FREE'   
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("bulk", case=False)), 'curr_video_tier_desc_m2'] = 'BULK'  
    x.loc[(x['curr_video_tier_desc_m2'].str.contains("family", case=False)), 'curr_video_tier_desc_m2'] = 'FAMILY' 
    x.loc[(x['curr_video_tier_desc_m2'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m2'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m2'].str.upper() !=  'CORE') \
          & (x['curr_video_tier_desc_m2'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m2'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m2'].str.upper() !=  'PREMIER') \
          # & (x['curr_video_tier_desc_m2'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m2'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m2'].str.upper() !=  'FAMILY') \
                         & (x['curr_video_tier_desc_m2'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m2'] = 'ALL_OTHER'
    print('After  : ', x['curr_video_tier_desc_m2'].unique())


    print('curr_video_tier_desc_m3 ======================') 
    print('Before : ', x['curr_video_tier_desc_m3'].unique()) 
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("basic", case=False)), 'curr_video_tier_desc_m3'] = 'BASIC'   
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("economy", case=False)), 'curr_video_tier_desc_m3'] = 'ECONOMY' 
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("core", case=False)), 'curr_video_tier_desc_m3'] = 'CORE'   
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("value", case=False)), 'curr_video_tier_desc_m3'] = 'VALUE' 
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("select", case=False)), 'curr_video_tier_desc_m3'] = 'SELECT'   
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("premier", case=False)), 'curr_video_tier_desc_m3'] = 'PREMIER' 
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("free", case=False)), 'curr_video_tier_desc_m3'] = 'FREE'   
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("bulk", case=False)), 'curr_video_tier_desc_m3'] = 'BULK'  
    x.loc[(x['curr_video_tier_desc_m3'].str.contains("family", case=False)), 'curr_video_tier_desc_m3'] = 'FAMILY' 
    x.loc[(x['curr_video_tier_desc_m3'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m3'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m3'].str.upper() !=  'CORE') \
          & (x['curr_video_tier_desc_m3'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m3'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m3'].str.upper() !=  'PREMIER') \
          # & (x['curr_video_tier_desc_m3'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m3'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m3'].str.upper() !=  'FAMILY') \
                         & (x['curr_video_tier_desc_m3'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m3'] = 'ALL_OTHER'
    print('After  : ', x['curr_video_tier_desc_m3'].unique())


    print('curr_video_tier_desc_m4 ======================') 
    print('Before : ', x['curr_video_tier_desc_m4'].unique()) 
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("basic", case=False)), 'curr_video_tier_desc_m4'] = 'BASIC'   
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("economy", case=False)), 'curr_video_tier_desc_m4'] = 'ECONOMY' 
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("core", case=False)), 'curr_video_tier_desc_m4'] = 'CORE'   
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("value", case=False)), 'curr_video_tier_desc_m4'] = 'VALUE' 
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("select", case=False)), 'curr_video_tier_desc_m4'] = 'SELECT'   
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("premier", case=False)), 'curr_video_tier_desc_m4'] = 'PREMIER' 
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("free", case=False)), 'curr_video_tier_desc_m4'] = 'FREE'   
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("bulk", case=False)), 'curr_video_tier_desc_m4'] = 'BULK'  
    x.loc[(x['curr_video_tier_desc_m4'].str.contains("family", case=False)), 'curr_video_tier_desc_m4'] = 'FAMILY' 
    x.loc[(x['curr_video_tier_desc_m4'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m4'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m4'].str.upper() !=  'CORE') \
          & (x['curr_video_tier_desc_m4'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m4'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m4'].str.upper() !=  'PREMIER') \
          # & (x['curr_video_tier_desc_m4'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m4'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m4'].str.upper() !=  'FAMILY') \
                         & (x['curr_video_tier_desc_m4'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m4'] = 'ALL_OTHER'
    print('After  : ', x['curr_video_tier_desc_m4'].unique())


    x = x.replace({'curr_video_tier_desc_m0':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1},
               'curr_video_tier_desc_m1':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
               'curr_video_tier_desc_m2':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
               'curr_video_tier_desc_m3':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
               'curr_video_tier_desc_m4':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 }})
    print('After encoding : ', x['curr_video_tier_desc_m0'].unique())


    x['video_tier_max_m1m4'] = x[["curr_video_tier_desc_m1", "curr_video_tier_desc_m2", "curr_video_tier_desc_m3", "curr_video_tier_desc_m4"]].max(axis=1)
    x['video_tier_change_m4m0'] = np.where(x['curr_video_tier_desc_m0'] == x['video_tier_max_m1m4'], 'same' , (np.where(x['curr_video_tier_desc_m0'] > x['video_tier_max_m1m4'], 'higher', 'lower')))


    x.loc[(x['video_tier_change_m4m0'] != 'same') & (x['curr_video_tier_desc_m0'] == -1), 'video_tier_change_m4m0'] = 'OTHER' 
    x.loc[(x['video_tier_change_m4m0'] != 'same') & (x['video_tier_max_m1m4'] == -1), 'video_tier_change_m4m0'] = 'OTHER' 
    x = x.replace({'curr_video_tier_desc_m0':{0 : 'None', 1 : 'BASIC' , 2 :'ECONOMY' , 3 : 'CORE', 4 : 'VALUE',  5 : 'SELECT', 6 : 'PREMIER', -1 : 'ALL_OTHER'}})
    x = x.drop(['curr_video_tier_desc_m1', 'curr_video_tier_desc_m2', 'curr_video_tier_desc_m3','curr_video_tier_desc_m4','video_tier_max_m1m4'],axis=1)
    print('After decoding : ', x['curr_video_tier_desc_m0'].unique())
    print('Dropped curr_video_tier_desc_m1 to curr_video_tier_desc_m4 and created video_tier_change_m4m0 variable')
    
    #----------------------------------------------------------
    print('\n curr_ov_tier_desc_m0 to curr_ov_tier_desc_m0 ==================================')
    x['ov_tier_change_m4m0'] = np.where((x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m1']) & 
                                      (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m2']) & 
                                      (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m3']) &
                                      (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m4']), 0 , 1)
    x = x.drop(['curr_ov_tier_desc_m0', 'curr_ov_tier_desc_m1', 'curr_ov_tier_desc_m2', 'curr_ov_tier_desc_m3','curr_ov_tier_desc_m4'],axis=1)
    print('created ov_tier_change_m4m0 variable')
    print("dropped curr_ov_tier_desc_m1 to curr_ov_tier_desc_m4")
    
    #---------------------------------------------------------------
    print('\n ool_offer_group_v2_desc_m0 to ool_offer_group_v2_desc_m4 ==========================')    
    x["ool_offer_group_v2_desc_m0"] = np.where((x["ool_offer_group_v2_desc_m0"].str.upper() == "NO OFFER") | (x["ool_offer_group_v2_desc_m0"].str.upper() == "NONE"), 0, 1)
    x["ool_offer_group_v2_desc_m1"] = np.where((x["ool_offer_group_v2_desc_m1"].str.upper() == "NO OFFER") | (x["ool_offer_group_v2_desc_m1"].str.upper() == "NONE"), 0, 1)
    x["ool_offer_group_v2_desc_m2"] = np.where((x["ool_offer_group_v2_desc_m2"].str.upper() == "NO OFFER") | (x["ool_offer_group_v2_desc_m2"].str.upper() == "NONE"), 0, 1)
    x["ool_offer_group_v2_desc_m3"] = np.where((x["ool_offer_group_v2_desc_m3"].str.upper() == "NO OFFER") | (x["ool_offer_group_v2_desc_m3"].str.upper() == "NONE"), 0, 1)
    x["ool_offer_group_v2_desc_m4"] = np.where((x["ool_offer_group_v2_desc_m4"].str.upper() == "NO OFFER") | (x["ool_offer_group_v2_desc_m4"].str.upper() == "NONE"), 0, 1)
    print('ool_offer_group_v2_desc_m0  : ', x['ool_offer_group_v2_desc_m0'].unique())
    print('ool_offer_group_v2_desc_m1  : ', x['ool_offer_group_v2_desc_m1'].unique())
    print('ool_offer_group_v2_desc_m2  : ', x['ool_offer_group_v2_desc_m2'].unique())
    print('ool_offer_group_v2_desc_m3  : ', x['ool_offer_group_v2_desc_m3'].unique())
    print('ool_offer_group_v2_desc_m4  : ', x['ool_offer_group_v2_desc_m4'].unique())
    
    #----------------------------------------------------------------- 
    print('ov_offer_group_v2_desc_m0 to ov_offer_group_v2_desc_m4 ===========================')         
    x["ov_offer_group_v2_desc_m0"] = np.where((x["ov_offer_group_v2_desc_m0"].str.upper() == "NO OFFER") | (x["ov_offer_group_v2_desc_m0"].str.upper() == "NONE"), 0, 1)
    x["ov_offer_group_v2_desc_m1"] = np.where((x["ov_offer_group_v2_desc_m1"].str.upper() == "NO OFFER") | (x["ov_offer_group_v2_desc_m1"].str.upper() == "NONE"), 0, 1)
    x["ov_offer_group_v2_desc_m2"] = np.where((x["ov_offer_group_v2_desc_m2"].str.upper() == "NO OFFER") | (x["ov_offer_group_v2_desc_m2"].str.upper() == "NONE"), 0, 1)
    x["ov_offer_group_v2_desc_m3"] = np.where((x["ov_offer_group_v2_desc_m3"].str.upper() == "NO OFFER") | (x["ov_offer_group_v2_desc_m3"].str.upper() == "NONE"), 0, 1)
    x["ov_offer_group_v2_desc_m4"] = np.where((x["ov_offer_group_v2_desc_m4"].str.upper() == "NO OFFER") | (x["ov_offer_group_v2_desc_m4"].str.upper() == "NONE"), 0, 1)
    print('ov_offer_group_v2_desc_m0  : ', x['ov_offer_group_v2_desc_m0'].unique())
    print('ov_offer_group_v2_desc_m1  : ', x['ov_offer_group_v2_desc_m1'].unique())
    print('ov_offer_group_v2_desc_m2  : ', x['ov_offer_group_v2_desc_m2'].unique())
    print('ov_offer_group_v2_desc_m3  : ', x['ov_offer_group_v2_desc_m3'].unique())
    print('ov_offer_group_v2_desc_m4  : ', x['ov_offer_group_v2_desc_m4'].unique())
    
    #-----------------------------------------------------------------
    print('\n video_offer_group_desc_m0 to video_offer_group_desc_m4 ===========================')  
    x["video_offer_group_desc_m0"] = np.where((x["video_offer_group_desc_m0"].str.upper() == "NO OFFER") | (x["video_offer_group_desc_m0"].str.upper() == "NONE"), 0, 1)
    x["video_offer_group_desc_m1"] = np.where((x["video_offer_group_desc_m1"].str.upper() == "NO OFFER") | (x["video_offer_group_desc_m1"].str.upper() == "NONE"), 0, 1)
    x["video_offer_group_desc_m2"] = np.where((x["video_offer_group_desc_m2"].str.upper() == "NO OFFER") | (x["video_offer_group_desc_m2"].str.upper() == "NONE"), 0, 1)
    x["video_offer_group_desc_m3"] = np.where((x["video_offer_group_desc_m3"].str.upper() == "NO OFFER") | (x["video_offer_group_desc_m3"].str.upper() == "NONE"), 0, 1)
    x["video_offer_group_desc_m4"] = np.where((x["video_offer_group_desc_m4"].str.upper() == "NO OFFER") | (x["video_offer_group_desc_m4"].str.upper() == "NONE"), 0, 1)

    print('video_offer_group_desc_m0  : ', x['video_offer_group_desc_m0'].unique())
    print('video_offer_group_desc_m1  : ', x['video_offer_group_desc_m1'].unique())
    print('video_offer_group_desc_m2  : ', x['video_offer_group_desc_m2'].unique())
    print('video_offer_group_desc_m3  : ', x['video_offer_group_desc_m3'].unique())
    print('video_offer_group_desc_m4  : ', x['video_offer_group_desc_m4'].unique())
      
    #------------------------------------------------------------------
    print('\n service_visit_1_tc_problem_code_desc to service_visit_3_tc_fix_code_desc ==============')     
    x["service_visit_1_tc_problem_code_desc"] = np.where((x["service_visit_1_tc_problem_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_1_tc_problem_code_desc"] = x["service_visit_1_tc_problem_code_desc"].astype('int32')
    x["service_visit_1_tc_fix_code_desc"]     = np.where((x["service_visit_1_tc_fix_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_1_tc_fix_code_desc"]     = x["service_visit_1_tc_fix_code_desc"].astype('int32')
    x["service_visit_2_tc_problem_code_desc"] = np.where((x["service_visit_2_tc_problem_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_2_tc_problem_code_desc"] = x["service_visit_2_tc_problem_code_desc"].astype('int32')
    x["service_visit_2_tc_fix_code_desc"]     = np.where((x["service_visit_2_tc_fix_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_2_tc_fix_code_desc"]     = x["service_visit_2_tc_fix_code_desc"].astype('int32')
    x["service_visit_3_tc_problem_code_desc"] = np.where((x["service_visit_3_tc_problem_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_3_tc_problem_code_desc"] = x["service_visit_3_tc_problem_code_desc"].astype('int32')
    x["service_visit_3_tc_fix_code_desc"]     = np.where((x["service_visit_3_tc_fix_code_desc"].str.upper() == "NONE"), 0, 1)
    x["service_visit_3_tc_fix_code_desc"]     = x["service_visit_3_tc_fix_code_desc"].astype('int32')

    print('service_visit_1_tc_problem_code_desc    : ', x['service_visit_1_tc_problem_code_desc'].unique())
    print('service_visit_1_tc_fix_code_desc        : ', x['service_visit_1_tc_fix_code_desc'].unique())
    print('service_visit_2_tc_problem_code_desc    : ', x['service_visit_2_tc_problem_code_desc'].unique())
    print('service_visit_2_tc_fix_code_desc        : ', x['service_visit_2_tc_fix_code_desc'].unique())
    print('service_visit_3_tc_problem_code_desc    : ', x['service_visit_3_tc_problem_code_desc'].unique())
    print('service_visit_3_tc_fix_code_desc        : ', x['service_visit_3_tc_fix_code_desc'].unique())
      
   
    #-----------------------------------------------------------------------    
    x['cust_tenure_days'] = x['cust_tenure_days'].astype('float64')
    x.loc[(x['cust_tenure_days'] <= 182), 'cust_tenure'] = '0_6mon'
    x.loc[(x['cust_tenure_days'] > 182) & (x['cust_tenure_days'] <= 365), 'cust_tenure'] = '6_12mon'
    x.loc[(x['cust_tenure_days'] > 365) & (x['cust_tenure_days'] <= 730), 'cust_tenure'] = '12_24mon'
    x.loc[(x['cust_tenure_days'] > 730) & (x['cust_tenure_days'] <= 1825), 'cust_tenure'] = '24_60mon'
    x.loc[(x['cust_tenure_days'] > 1825), 'cust_tenure'] = 'morethan_60mon'
    x = x.drop(['cust_tenure_days'], axis=1)
    x['cust_tenure'] = x['cust_tenure'].astype('str').fillna(value='None')

    
    print('created cust_tenure and dropped cust_tenure_days')    
    print('cust_tenure  : ', x['cust_tenure'].unique())

    #------------------------------------------------------------------------
    print('promo_rolloff_count_60day')
    x["promo_rolloff_count_60day"] = np.where(x["promo_rolloff_count_60day"] > 0, 1, x["promo_rolloff_count_60day"])


    #------------------------------------------------------------------------
    print('Adding new feature combinations ================================')
    x = x.drop(['metro_area_desc'],axis=1)
    print('dropped metro_area_desc')

    x['cust_equip_class_speedm0'] = x["cust_equip_class"] + '_' + x["speed_m0"]
    x['tenure_productclass']      = x["cust_tenure"] + '_' + x["product_class_desc"]
    x['tenure_income']            = x["cust_tenure"] + '_' + x["ecohort_income_range_desc"]
    print('created cust_equip_class_speedm0, tenure_productclass, tenure_income')


    x['rateevent_baserevchange'] = np.where((x["rate_event_count_60day"] > 0) & (x["baserev_m0"] - x["baserev_m1"] != 0), 1, 0)
    x = x.drop(['baserev_m0', 'baserev_m1', 'baserev_m2', 'baserev_m3', 'baserev_m4', 'baserev_m5', 'baserev_m6'],axis=1)
    print('created rateevent_baserevchange and dropped baserev_m0 to baserev_m6')

    x['agentcall_and_repeat1'] = np.where((x["agent_last_dept"].str.upper() != 'NONE') & \
                                                             (x["agent_repeat_1_prevDept"].str.upper() != 'NONE') & \
                                                             (x["agent_repeat_2_prevDept"].str.upper() == 'NONE') , 1, 0)
    x['agentcall_and_repeat2'] = np.where((x["agent_last_dept"].str.upper() != 'NONE') & \
                                                                (x["agent_repeat_2_prevDept"].str.upper() != 'NONE') & \
                                                                (x["agent_repeat_3_prevDept"].str.upper() == 'NONE'), 1, 0)
    x['agentcall_and_repeat3'] = np.where((x["agent_last_dept"].str.upper() != 'NONE') & \
                                                             (x["agent_repeat_3_prevDept"].str.upper() != 'NONE'), 1, 0)
    print('created agentcall_and_repeat1, agentcall_and_repeat2, agentcall_and_repeat3')                                                         

    x['servicevisit_once']       = np.where((x["service_visit_1_tc_problem_code_desc"] == 1) & (x["service_visit_2_tc_problem_code_desc"] == 0), 1, 0)
    x['servicevisit_twotimes']   = np.where((x["service_visit_2_tc_problem_code_desc"] == 1) & (x["service_visit_3_tc_problem_code_desc"] == 0), 1, 0)
    x['servicevisit_threetimes'] = np.where((x["service_visit_3_tc_problem_code_desc"] == 1), 1, 0)
    print('created servicevisit_once, servicevisit_twotimes, servicevisit_threetimes')  

    x['agentcall_servicevisit']       = np.where((x["agent_last_dept"].str.upper() != 'NONE') & (x["service_visit_1_tc_problem_code_desc"] == 1), 1, 0)
    print('created agentcall_servicevisit') 

    x["FAILSTAT_OFFLINEEVT2"]        = np.where((x["FAILSTAT_OFFLINEEVT"] > 0), 1, 0)
    x["FAILSTAT_DHCP_MACADDR2"]      = np.where((x["FAILSTAT_DHCP_MACADDR"] > 0), 1, 0)
    x["FAILSTAT_WIFI_REBOOT_COUNT2"] = np.where((x["FAILSTAT_WIFI_REBOOT_COUNT"] > 0), 1, 0)
    x["FAILSTAT_WIFI_QOE_HOME_302"]  = np.where((x["FAILSTAT_WIFI_QOE_HOME_30"] > 0), 1, 0)
    x['agentlastdep_OFFLINEdistress']    = x["agent_last_dept"].astype('str') + '_' + x["FAILSTAT_OFFLINEEVT2"].astype('str') 
    x['agentlastdep_DHCPdistress']       = x["agent_last_dept"].astype('str') + '_' + x["FAILSTAT_DHCP_MACADDR2"].astype('str') 
    x['agentlastdep_WIFIRebootdistress'] = x["agent_last_dept"].astype('str') + '_' + x["FAILSTAT_WIFI_REBOOT_COUNT2"].astype('str') 
    x['agentlastdep_WIFIQoEdistress']    = x["agent_last_dept"].astype('str') + '_' + x["FAILSTAT_WIFI_QOE_HOME_302"].astype('str') 
    x = x.drop(['FAILSTAT_OFFLINEEVT2', 'FAILSTAT_DHCP_MACADDR2','FAILSTAT_WIFI_REBOOT_COUNT2','FAILSTAT_WIFI_QOE_HOME_302'],axis=1)
    print('created agentlastdep_OFFLINEdistress, agentlastdep_DHCPdistress, agentlastdep_WIFIRebootdistress, agentlastdep_WIFIQoEdistress')

    #-----------------------------------------------------------
    # return final cat cols and num cols
    all_df_cols_after  = x.columns 
    cat_cols_dropped   = list(set(cat_cols) - set(all_df_cols_after))
    num_cols_dropped   = list(set(num_cols) - set(all_df_cols_after))
    
    new_cols_added = ['baserev_change_m6m0','speed_change_m4m0', 'max_svod_change_m3m1', 'hbo_svod_change_m3m1', 'stz_enc_svod_change_m3m1', 'video_tier_change_m4m0', 'ov_tier_change_m4m0', 'cust_tenure',
                      'cust_equip_class_speedm0', 'tenure_productclass', 'tenure_income', #'archetype_oolpromoleft','headend_metroarea',
                      'agentlastdep_OFFLINEdistress', 'agentlastdep_DHCPdistress', 'agentlastdep_WIFIRebootdistress', 'agentlastdep_WIFIQoEdistress']

    new_cat_cols   = list(set(cat_cols) - set(cat_cols_dropped)) + new_cols_added
    
    new_num_cols = ['service_visit_1_tc_problem_code_desc', 'service_visit_1_tc_fix_code_desc', 'service_visit_2_tc_problem_code_desc', 'service_visit_2_tc_fix_code_desc',
                      'service_visit_3_tc_problem_code_desc', 'service_visit_3_tc_fix_code_desc',
                      'ov_tier_change_m4m0', 'ool_offer_group_v2_desc_m0', 'ool_offer_group_v2_desc_m1', 'ool_offer_group_v2_desc_m2', 'ool_offer_group_v2_desc_m3', 'ool_offer_group_v2_desc_m4', 
                      'ov_offer_group_v2_desc_m0', 'ov_offer_group_v2_desc_m1', 'ov_offer_group_v2_desc_m2', 'ov_offer_group_v2_desc_m3', 'ov_offer_group_v2_desc_m4',
                      'video_offer_group_desc_m0', 'video_offer_group_desc_m1', 'video_offer_group_desc_m2', 'video_offer_group_desc_m3', 'video_offer_group_desc_m4',
                      'servicevisit_once', 'servicevisit_twotimes', 'servicevisit_threetimes',
                      'agentcall_and_repeat1', 'agentcall_and_repeat2', 'agentcall_and_repeat3', 'rateevent_baserevchange','agentcall_servicevisit']

    final_num_cols = list(set(num_cols) - set(num_cols_dropped)) + new_num_cols                     
    # final_num_cols = num_cols + new_num_cols
    
    final_cat_cols = list(set(new_cat_cols) - set(new_num_cols))
    
    return {'df': x, 'final_cat_cols' : final_cat_cols, 'final_num_cols': final_num_cols, 'new_num_cols_created': new_num_cols}




def RareCategoryEncoder(df, features_relevant, category_min_pct=0.01, category_max_count=20):
    
    # 1. Create Catetory Mapping Dictionary
    # - Total numer of categories = self.category_max_count
    # - Min pct of selected categories >= self.category_min_pct
    category_mapper_         = {}

    for fe in features_relevant:
        mapping              =  df[fe].value_counts(normalize=True).iloc[:category_max_count]
        print(mapping)
        category_mapper_[fe] =  mapping[mapping >= category_min_pct].index
            
    tmp_df             = df.copy()
    
    for fe in features_relevant:
        tmp_df[fe] =  np.where(tmp_df[fe].isin(category_mapper_[fe]), tmp_df[fe], 'ALL_OTHER')    
    
    X_transformed      = tmp_df
    
    return X_transformed  




def preprocess_num_cols(dfr, num_cols, replace_neg_values = False , remove_constant_features = True, lower_quant = 0, upper_quant = 0.995):

    df = dfr.copy()

    flag_cols = [col for col in df if col.startswith('flag')]
    service_cols = [col for col in df if col.startswith('service_visit')]
    relevant_cols = list(set(num_cols) - set(flag_cols) - set(service_cols))
    
    # Replace max values with upper_quant percentile
    df[relevant_cols] = df[relevant_cols].astype('float64').apply(lambda x: x.clip(x.quantile(lower_quant), x.quantile(upper_quant)))
    print('1. Replaced max values with lower_quant and upper_quant percentiles')
    
    # Replace negative values with 0
    if replace_neg_values == True:
        x = df[relevant_cols]
        x[x < 0] = 0
        df[relevant_cols] = x
        print('2. Replaced negative values with 0')    


    const_cols_removed = []
    # Remove features with constatnt values  
    if remove_constant_features == True:                      
        cols_before = df[num_cols].columns          
        df = df.loc[:,df.apply(pd.Series.nunique) != 1]
        all_cols_after  = df.columns 
        const_cols_removed = list(set(cols_before) - set(all_cols_after))
        print('3. Constant features removed: ', const_cols_removed)    
   
    # Replace missing values with 0
    num_cols_remain = list(set(num_cols) - set(const_cols_removed))
    df[num_cols_remain] = df[num_cols_remain].astype('str').replace('nan','0').astype('float64')
    print('4. Replaced missing values with 0') 
    

    return {'df':df, 'const_cols_removed':const_cols_removed, 'num_cols_remain':num_cols_remain}




def get_feature_name(df_X, y=None, prefix=None, suffix=None):
    '''
    Purpose
    ---------------
    A custom utility function to provide appropriate feature names to transformed data
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df)
    y: a pandas series that represent a churn status
        - default: y=None
    prefix: a string that is appended as prefix to the feature names of df_X
        - default: prefix=None
    suffix: a string that is appended as suffix to the feature names of df_X
        - default: suffix=None

    Returns
    ---------------
    feature_Name: a list of transformed feature names
    '''

    feature_Name     = df_X.columns.tolist()
    
    if (prefix is not None) & (suffix is not None):
        feature_Name = [prefix + '_' + fe + '_' + suffix for fe in df_X.columns]
    elif (prefix is not None):   
        feature_Name = [prefix + '_' + fe for fe in df_X.columns]
    elif (suffix is not None):   
        feature_Name = [fe + '_' + suffix for fe in df_X.columns]

    return feature_Name





class UniversalCategoryEncoder(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom UNIVERSAL encoder to encode CATEGORICAL features with a selected encoding method
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
        - default: y=None
    encoding_method: ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']
        - One must choose one of the listed encoding methods.
        - 'ohe': OneHotEonding
            - Generate 0/1 binary variable for every category of CATEGORICAL features.
        - 'pct': 
            - Replace category with its corresponding %.
        - 'count':
            - Replace category with its corresponding count.
        - 'ordinal':
            - y should be given.
            - Replace category with its order of average value of target y.
        - 'y_mean':
            - y should be given.
            - Replace category with its corresponding average value of target y.            
        - 'y_log_ratio':
            - y should be given.
            - Replace category with its corresponding log(p(Churner)/p(Non-Churner)).
            - p(Churner) = 0 be set to p(Churner) = 0.00001 to avoid -inf.
        - 'y_ratio':        
            - y should be given.    
            - Replace category with its corresponding (p(Churner)/p(Non-Churner)).
    prefix: a string to add a prefix to all column names
        - default: prefix=None
    suffix: a string to add a suffix to all column names
        - default: suffix=None

    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    category_mapper_: 
        - a data dictionary of mapping features to corresponding categories after encoding

    References
    ---------------        
    Feature Engine: 
        - https://pypi.org/project/feature-engine/
        - https://github.com/solegalli/feature_engine/blob/master/feature_engine/categorical_encoders.py
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary

        
    # Need to specify grouping variables that will be used in feature aggregation.
    # features_geo_cardinal = a list of high cardinal/geographic features that can be used as grouping variables
    # features_geo_cardinal = ['census', 'cleansed_city', 'cleansed_zipcode', 'ecohort_code', \
    #                          'clust', 'corp', 'fta_desc', 'house', 'hub', 'trunk', 'node', \
    #                          'geo', 'geo2', 'geo3']

    def __init__(self, encoding_method, prefix=None, suffix=None):
        if encoding_method not in ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']:
            raise ValueError("ONE encoding method should be choosen from ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio'].")

        if encoding_method in ['ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']:
            print(f"'{encoding_method}' encoding requires target y.")

        self.encoding_method = encoding_method
        self.prefix          = prefix
        self.suffix          = suffix
        

    def fit(self, X, features_CAT_, y=None):
        # 0. Select Relevant Features
        # features_relevant = list(set(X.columns) - (set(self.features_irrelevant)))
        # features_relevant.sort()
        # features_CAT_     = X[features_relevant].select_dtypes(include=[object]).columns.tolist()

        # Create Catetory Mapping Dictionary
        category_mapper_             = {}

        if self.encoding_method      == 'ohe':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts().index

        elif self.encoding_method    == 'pct':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts(normalize=True).to_dict()

        elif self.encoding_method    == 'count':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts().to_dict()

        elif self.encoding_method    == 'ordinal':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                mapping              = tmp_df.groupby(fe)['y'].mean().sort_values(ascending=True).index
                category_mapper_[fe] = {j:i for i, j in enumerate(mapping, 0)}

        elif self.encoding_method    == 'y_mean':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                category_mapper_[fe] = tmp_df.groupby(fe)['y'].mean().sort_values(ascending=False).to_dict()

        elif self.encoding_method    == 'y_log_ratio':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                tmp_stat             = tmp_df.groupby(fe)['y'].mean()
                tmp_stat             = pd.concat([tmp_stat, 1-tmp_stat], axis=1)
                tmp_stat.columns     = ['p1', 'p0']
                # Assign a small value, .00001 if p = 0
                tmp_stat['p1']       = np.where(tmp_stat['p1'] <= 0, .00001, tmp_stat['p1'])
                tmp_stat['p0']       = np.where(tmp_stat['p0'] <= 0, .00001, tmp_stat['p0'])
                category_mapper_[fe] = (np.log(tmp_stat.p1/tmp_stat.p0)).to_dict()

        elif self.encoding_method    == 'y_ratio':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                tmp_stat             = tmp_df.groupby(fe)['y'].mean()
                tmp_stat             = pd.concat([tmp_stat, 1-tmp_stat], axis=1)
                tmp_stat.columns     = ['p1', 'p0']
                # Assign a small value, .00001 if p = 0
                tmp_stat['p0']       = np.where(tmp_stat['p0'] <= 0, .00001, tmp_stat['p0'])
                category_mapper_[fe] = (tmp_stat.p1/tmp_stat.p0).to_dict()

        self.features_CAT_           = features_CAT_
        self.category_mapper_        = category_mapper_

        return self


    def transform(self, X, y=None):
        # Note: A custom utility function, 'get_feature_name' is used.        
        tmp_df             = X[self.features_CAT_]

        if self.encoding_method == 'ohe':
            for fe in self.features_CAT_:
                for cat in self.category_mapper_[fe]:
                    tmp_df[f'{fe}_{cat}'] = np.where(tmp_df[fe] == cat, 1, 0)
            # Drop the original features for 'ohe' only.
            tmp_df.drop(self.features_CAT_, axis=1, inplace=True)
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)

        elif self.encoding_method == 'y_log_ratio':
            # Note: log(0) = - inf --> replace Null with a very small prob such as .00001.
            for fe in self.features_CAT_:
                tmp_df[fe] = tmp_df[fe].map(self.category_mapper_[fe]).fillna(np.log(.00001/(1-.00001)))
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)    

        else:
            for fe in self.features_CAT_:
                tmp_df[fe] = tmp_df[fe].map(self.category_mapper_[fe]).fillna(0)
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)    

        X_transformed = tmp_df

        X_other = X.loc[:, ~X.columns.isin(self.features_CAT_)]
        df_full = pd.concat([X_transformed,X_other], axis = 1)

        return {'X_transformed': X_transformed, 'X_full': df_full}