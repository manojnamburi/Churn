# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:47:03 2019

@author: syadalam


Script for defining helper functions.
"""
from datetime import datetime
import pandas as pd
import numpy as np
from config_dict import config_dict
from google.cloud import storage
import os
###############################################################################
#Function:  Log to console and to file
###############################################################################


def log(*args,**kwargs):
    ##Print to console
    print(*args,**kwargs)
    print(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'))
    ##Print to file
    with open("voluntary_churn_log_" + datetime.now().strftime("%Y%m%d")+'.log',"a") as f:  # appends to file and closes it when finished
        print(file=f,*args,**kwargs)
        print(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'),file=f)

##################################################################################################################################
###Function to get the 6 month string from the date entered for the train and test dates
##################################################################################################################################
def get_6m_list(rundt):
    a = int(rundt) 
    z = []
    for i in range(6):
        if(i < 1):
            if(a % 100 == 12):
                z.append(a + 89)
            else:
                z.append(a+1)
            
        else:
            if(z[i-1] % 100 == 12):
                z.append(z[i-1] + 89)   
            else:
                z.append(z[i-1] + 1)
                
    list_6m =  list(map(str,z))
    return list_6m
    
##################################################################################################################################
##Function to strip leading zeroes to get house and cust
##################################################################################################################################
def rev_lpad(input_string,c):
    try:
        
        while (input_string[0] == c):
            input_string = input_string.lstrip(c)
    except Exception as e:
        input_string = 0
        log('*'*10+'\nGeneration of corp house cust logic failed with Exception: '+ str(e))
    return input_string


##################################################################################################################################
##Function to get the corp house cust for a list
##################################################################################################################################
def get_chc(hhid_list):
    try:
        scrubdf = pd.DataFrame(
                {'corp': pd.Series(hhid_list).apply(lambda x: str(x)[0:4]),
                 'house': pd.Series(hhid_list).apply(lambda x: rev_lpad(str(x)[4:10],'0')),
                 'cust' : pd.Series(hhid_list).apply(lambda x: rev_lpad(str(x)[10:],'0'))
                 })
    except Exception as e:
        scrubdf = None
        log('*'*10+'\nGetting corp house cust from the list has failed with exception: '+ str(e))
    return scrubdf

##################################################################################################################################
##Function to connect to the Google Cloud infrastructure
##################################################################################################################################
def dbConnect():
    log('*'*10+'\nAdding the necessary dependencies to environment to connect to Google Platform')
    ##Comment this and config file when using service account in local machine/server. Keep it as is for using default GCP credentials of the Compute machine
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=config_dict['credentials_file']
    os.environ['BUCKET'] = config_dict['BUCKET']
    os.environ['PROJECT'] = config_dict['PROJECT']
    
    
    
    try:
        res = 0
        ##Check if the environment variables were correctly instantiated
        if os.environ['BUCKET'] == config_dict['BUCKET']:
            res = 1
            if os.environ['PROJECT'] == config_dict['PROJECT']:
                res = 2
                if os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == config_dict['credentials_file']:
                    res = 3
        else:
            res = 0
      
    except Exception as e:
        log('Failed to instantiate environment variables due to exception: '+str(e))
       
    
    return res

##################################################################################################################################
###Download from GCS and read the object
##################################################################################################################################

def download_model_objs(file_path,input_str):
    """
    Description:
        Method to download the objects needed for Voluntary churn modelling from Google Storage. 
        Specify the path relative the bucket and give the same prefix to all the objects for this to work.
    
    Parameters:
        file_path (str): Path of the files relative to the bucket specified in the config_dict['BUCKET'] parameter
        input_str (str): Prefix of the file names that is common to all the objects needed
        
    Returns:
        No return
    """
    ##Invoke the GCS client to read the objects 
    gcs_client = storage.Client(config_dict['PROJECT'])
    ##Retrieve the bucket object
    gcs_bucket = gcs_client.get_bucket(config_dict['BUCKET'])
    ##Get the blobs that match with the pattern entered
    blobs = gcs_bucket.list_blobs(prefix = file_path+'/'+input_str, delimiter = '/')
    ##Retrieve the objects
    files_to_be_read = [blob.name for blob in blobs]
    for f in files_to_be_read:
        log('Retrieving file: '+ f)
        tempblob = gcs_bucket.blob(f)
        temp_file_name = 'temp_'+ f.replace(file_path+'/','') ##Replace the folder name in the file name to just keep the name of the file
        tempblob.download_to_filename(temp_file_name)
    
    
    
    