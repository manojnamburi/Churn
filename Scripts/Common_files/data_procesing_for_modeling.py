# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:07:21 2018

@author: vguntupa
"""

import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd
#import pandas_gbq as gbq

from attribute_dictionary import attribute_dict
from imputation_dictionary import attribute_imputer_dict

import voluntary_prediction_helpers as vph
########################################################################################################
#Imputation & all other data preperation good stuff for modeling
########################################################################################################
def process_data(df,dataset):
    '''
    Takes dataframe as an input and performs the standard clean up and binarizes to make the feature set fit for typical scikit models to ingest
    '''
    df_data = df
    
    vph.log('*'*10 + "\nData processing for %s dataset starts now: "%dataset, flush=True)
    #vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)

    #Put attribute column-names in a list
    list_col_names = list(df_data.iloc[0:,])
    
    #Seperate categorical and numerical attributes and also the target variable
    cat_attributes     = [x for x in list_col_names if (attribute_dict[x] == 'CAT')]
    num_attributes     = [x for x in list_col_names if (attribute_dict[x] == 'NUM')]
    target_attribute   = [x for x in list_col_names if (attribute_dict[x] == 'TARGET')]
    
    #Let's impute any whitespaces with NaNs:
    df_data = df_data.replace(r'^\s*$', np.nan, regex=True) #careful to not replace whitespaces in between characters
    
    #Let's see how many NaNs there are:
    vph.log('*'*10 + '\nNaNs in attributes:\n' + '*'*10, flush=True)
    vph.log(df_data.isnull().sum(), flush=True)
    
    #Impute ALL attributes
    for each in list_col_names:
        df_data[each].fillna(attribute_imputer_dict[each], inplace=True)
    
    #Let's drop records with any NaNs
    df_data = df_data.dropna(how='any')
    
    #Let's see again how many NaNs there are (should be zero):
    vph.log('*'*10 + '\nNaNs in attributes after imputation and dropping NaNs (after dropna="any"):\n' + '*'*10, flush=True)
    vph.log(df_data.isnull().sum(), flush=True)
    vph.log('*'*10 + '\nNumber of records after NaNs removal: ' + str(len(df_data)) +'\n'+ '*'*10, flush=True)
    
    #Final list of numeric variables ingested to model:
    num_attributes_model_ip = [each for each in num_attributes if each in df_data.columns.values]
    df_data[num_attributes_model_ip] = df_data[num_attributes_model_ip].astype(float)
    
    #Let's dummy data for CAT attributes first and then concat with NUM attributes. First, let's splice categorical data into it's own DF.
    df_data_cat_to_be_dummied = df_data[cat_attributes] 
    
    #Make sure all th categorical features are explicitly casted as a string/object
    df_data_cat_to_be_dummied = df_data_cat_to_be_dummied.astype(str)
    
    #Let's dummy categorical attributes
    df_data_cat_dummied = pd.get_dummies(df_data_cat_to_be_dummied, prefix=cat_attributes)
    
    del df_data_cat_to_be_dummied

    
    
    #Let's keep the class label aside and concat it later NUM and dummied-CAT features later
    if dataset=='Test' or dataset == 'Train':
        class_label_df = df_data['status']
        #Let's bring together 1) class variable, 2) numerical data, and 3) dummied categorical data, to create the final dataset
        df_data_model_ip = pd.concat([class_label_df, df_data[num_attributes_model_ip], df_data_cat_dummied], axis=1)
    elif dataset == 'Infer':
        df_data_model_ip = pd.concat([df_data[num_attributes_model_ip], df_data_cat_dummied], axis=1)
    
    
    
    #Let's columns that have zero variance. Helps avoid divide by zero error during standardization and shrinks featureset for good.
    cols_with_variance_zero = [each_col for each_col in df_data_model_ip.columns.values if df_data_model_ip[each_col].var() == 0]
    vph.log('*'*10)
    vph.log('These columns have zero variance, so dropping them: \n', flush=True)
    for each in cols_with_variance_zero: vph.log('-',each)
    remaining_cols = [x for x in df_data_model_ip.columns.values if x not in cols_with_variance_zero]
    df_data_model_ip = df_data_model_ip[remaining_cols]
    
    vph.log('*'*10 + "\nData processing for %s dataset finished. "%dataset, flush=True)
    #vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
    
    return df_data_model_ip



########################################################################################################
#Plot correlations and output correlation scores only for unique combinations
########################################################################################################
# def correlation_scores_and_plot(df,dataset):
#     '''
#     Checking to see what the correlation looks like among model input features
#     '''
#     corr = df.corr()#.abs() #let's get correlations
    
#     #Plot
#     fig, ax = plt.subplots(figsize=(20, 30))
#     ax.matshow(corr)
#     plt.xticks(range(len(corr.columns)), corr.columns)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.show()
#     plt.savefig('Correlations.png', format='png')
#     plt.close()
    
#     #Correlation scores
#     corr1 = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)) #Let's convert the lower triangle to all nans
#     x = pd.DataFrame(corr1.unstack()) #convert the corr. matrix in to a table
#     x.reset_index(inplace=True) #make value in multi-level index applicable to all rows
#     x = x.dropna()
#     x = x[x.iloc[:,0] != x.iloc[:,1]] #remove diagonal elements with correlation = 1
#     x = x.sort_values(by=0, kind="quicksort", na_position = 'last') #sort by correlation values
#     name = dataset+"_dataset_all_attributes_correlations_ordered.csv" #file to write results to
#     x.to_csv(name,index=False,header=['Column_1','Column_2','Correlation_Score']) #write to CSV
#     target_corr_file_name = dataset+"_dataset_target_attribute_correlations_ordered.csv"
#     vph.log('*'*10 + "\nCorrelations plot and correlation scores for unique combinations of input features of %s dataset generated. "%dataset, flush=True)
#     vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
#     res_df = df.drop("status", axis=1).apply(lambda x: x.corr(df['status'])).reset_index()
#     res_df.columns = ['Attribute','corr_score']
#     res_df = res_df.sort_values('corr_score',ascending = False)
#     pd.DataFrame(res_df).to_csv(target_corr_file_name)
#     vph.log('*'*10 + '\nCorrelations with the target variable for %s dataset generated.'%dataset, flush = True)

def correlation_scores_and_plot(df,dataset):
    '''
    Checking to see what the correlation looks like among model input features
    '''
    #df = df_data_model_ip
    col_list = pd.Series(df.columns)
    col_list = list(set(col_list.apply(lambda x: str(x).split('vg58rj')[0])))
    cat_cols = [col for col in attribute_dict if attribute_dict[col]=='CAT']
    cat_cols = [col for col in cat_cols if col in col_list]  #Get cols only in the data
    
    exclude_cols = []
    
    ###Remove columns with greater than 100 categories to avoid the code to break
    for col in cat_cols:
        count_unique = sum(pd.Series(df.columns).apply(lambda x: str(x).startswith(col)))
        if ((count_unique>=100) or (count_unique==1)):
            exclude_cols.append(col)
    
    for col in exclude_cols:
        df.drop(df.columns[pd.Series(df.columns).apply(lambda x: str(x).startswith(col))], inplace = True, axis = 1)
    
    corr = df.corr()#.abs() #let's get correlations
    
    #Plot
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    plt.savefig('Correlations.png', format='png')
    plt.close()
    
    #Correlation scores
    corr1 = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)) #Let's convert the lower triangle to all nans
    x = pd.DataFrame(corr1.unstack()) #convert the corr. matrix in to a table
    x.reset_index(inplace=True) #make value in multi-level index applicable to all rows
    x = x.dropna()
    x = x[x.iloc[:,0] != x.iloc[:,1]] #remove diagonal elements with correlation = 1
    x = x.sort_values(by=0, kind="quicksort", na_position = 'last') #sort by correlation values
    name = dataset+"_dataset_all_attributes_correlations_ordered.csv" #file to write results to
    x.to_csv(name,index=False,header=['Column_1','Column_2','Correlation_Score']) #write to CSV
    target_corr_file_name = dataset+"_dataset_target_attribute_correlations_ordered.csv"
    vph.log('*'*10 + "\nCorrelations plot and correlation scores for unique combinations of input features of %s dataset generated. "%dataset, flush=True)
    vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
    res_df = df.drop("status", axis=1).apply(lambda x: x.corr(df['status'])).reset_index()
    res_df.columns = ['Attribute','corr_score']
    res_df = res_df.sort_values('corr_score',ascending = False)
    pd.DataFrame(res_df).to_csv(target_corr_file_name)
    vph.log('*'*10 + '\nCorrelations with the target variable for %s dataset generated.'%dataset, flush = True)


##########################################################################################################
#Get value counts of variables and write to csv
##########################################################################################################
def get_value_counts(df_data,dataset):
    cat_values_df = pd.DataFrame(columns = ['Attribute','status','Attribute_level','Count'])
    list_col_names = [x for x in list(df_data.iloc[0:,]) if x not in ['corp','house','chc_id']]
    cat_attributes     = [x for x in list_col_names if (attribute_dict[x] == 'CAT')]
    num_attributes = [x for x in list_col_names if (attribute_dict[x] == 'NUM')]
    #col = cat_attributes[5]
    for col in cat_attributes:
        val_df = pd.DataFrame(df_data.groupby([col,'status']).agg({'chc_id':pd.Series.nunique})).reset_index()
        val_df = val_df.rename(index = str,columns = {col:'Attribute_level','chc_id':'Count'})
        val_df.loc[:,'Attribute'] = col
        val_df = val_df.sort_values('status')
        cat_values_df = cat_values_df.append(val_df)
        cat_values_df = cat_values_df.sort_values(['Attribute','status'])
        cat_values_df = cat_values_df[['Attribute','status','Attribute_level','Count']]
    cat_values_df.to_csv(str(dataset+'_cat_distributions.csv'),index=False)
    vph.log('*'*10 + '\n Finished writing the cat distributions to csv for '+dataset,flush = True)
    #vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
    num_values_df = pd.DataFrame(columns = ['Attribute','status','mean','sd','median','min','max'])
    #col = num_attributes[5]
    for col in num_attributes:
        df_data[col] = pd.to_numeric(df_data[col])
        val_df = pd.DataFrame(df_data.groupby(['status']).agg({col:[np.mean,np.std,np.median,np.min,np.max]})).reset_index()
        val_df.columns = ['status','mean','sd','median','min','max']
        val_df.loc[:,'Attribute'] = col
        num_values_df = num_values_df.append(val_df)
        num_values_df = num_values_df.sort_values(['Attribute','status'])
        num_values_df = num_values_df[['Attribute','status','mean','sd','median','min','max']]
    num_values_df.to_csv(str(dataset+'_num_distributions.csv'),index=False)
    vph.log('*'*10 + '\n Finished writing the num distributions to csv for '+dataset,flush = True)
    #vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)