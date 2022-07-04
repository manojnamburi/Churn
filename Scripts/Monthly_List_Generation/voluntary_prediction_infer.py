##Read in the dependencies
import datetime

#import DATAFRAME_TO_REDSHIFT_TABLE_V2 as pd2rs
#def df_to_rs(source_df,key,dest_schema,dest_table,drop) (drop takes True or False, i.e., to drop existing table or just append it to )
#import boto3

#from churn_model_prep import model_prep
#vph.log('*'*10 + "\nDone importing the list of dummied categories that are available in both TRAIN & TEST data.", flush=True)
#import psycopg2
import string
import numpy as np
import pandas as pd
#import pandas_gbq as gbq





import data_procesing_for_modeling
from Class_GeneralUtilitiesNonTF import GeneralUtilitiesNonTF

#import output_automation_bq as oa
import sys

import voluntary_prediction_helpers as vph

##Method to score the test/infer data 

def score_data(model,test_data,scaler,train_cols,project_id,dataset,bucket,auth_file,test_month,run_type='Infer',train_identifier='',\
               *args,**kwargs):
    """Method to score the voluntary data. Takes in all the necessary model parameters and writes the output to either Infer/Test prediction tables. If test, also generates the model run statistics and other metrics of the model. """
    vph.log('*'*10 + "\n{} start time:".format(run_type), flush=True)
    ##Generate the identifier string that would be appended to the final prediction results to be stored in a single table.
    ui_identifier_str = '_'.join([train_identifier,run_type,str(test_month)])
    ##Also get the time stamp string to store the time the score method was invoked.
    time_run = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    ##Utility object instantiation
    util_obj = GeneralUtilitiesNonTF(project_id = project_id,dataset = dataset,bucket_name = bucket,json_path = auth_file)
    
    ##Check to see if there are any duplicate chc_id
    if len(test_data)!= len(test_data['chc_id'].unique()):
        vph.log('*'*10+"\nDuplicate chc_ids found.")
        sys.exit(1)
    else:
        vph.log('*'*10+"\nNo duplicate ids found.")

    #data_procesing_for_modeling.get_value_counts(test_data,'Test')
    #test_data.to_csv('test.csv',index=False)
    #Let's make index of the dataframe as corp_house_cust
    test_data.index = test_data['chc_id']

    #Sort by hhid
    test_data = test_data.sort_index()

    #Let's remove redundant columns
    ###Changing the sql to not read in the redundant columns
    del test_data['chc_id']

    ########################################################################################################
    #Imputation & all other data preperation good stuff for modeling
    ########################################################################################################
    df_data_model_ip = data_procesing_for_modeling.process_data(test_data, dataset=run_type)
    ##Add status column to the final columns in case it is test
    if run_type == 'Test':
        test_cols = ['status'] + train_cols ##TBD change this to work with any target variable
    elif run_type == 'Infer':
        test_cols = train_cols.copy()
    
    ########################################################################################################
    #XGBOOST does not like special characters in column names. Let's clean that up.
    ########################################################################################################
    #Make sure column names contain only letters, and numbers. no special characters.
    #But we found out that in some cases removing a + sign or space sometimes resulted in duplicate categories, so allowing them
    #But XGB does not like these charactes, so we will replace them with following strings: 'PLUS' and 'SPACE'
    validchars = string.ascii_letters + string.digits + str('+') + str(' ')
    original_names = [] #will use to hold the original attribute-categories in dummied format
    clean_names = [] #will use to hold transformed attribute-category names suited for XGBoost model
    for each in df_data_model_ip.columns.values: #for each dummied column
        temp1 = ''.join(x for x in each if x in validchars) #retain only characters defined in validchars
        temp2 = [x if x != '+' else 'PLUS' for x in temp1] #replace the character '+' with the string PLUS
        temp = [x if x != ' ' else 'SPACE' for x in temp2] #replace the character ' ' with the string SPACE
        temp = ''.join(x for x in temp) #join all the characters into one word
        clean_names.append(temp)
        original_names.append(each)
    
    df_data_model_ip.columns = clean_names
    ##########################################################################################
    ##Matching the test data columns to train
    ##########################################################################################
    ##Restrict the categories to only those that are used to train the model
    #train_cols.insert(0,'status')
    temp_cols = [each for each in df_data_model_ip.columns if each in test_cols]# limit to only attribute-category combinations available in both TRAIN & TEST
    df_data_model_ip = df_data_model_ip[temp_cols]
    
    if len(df_data_model_ip.columns) < len(train_cols):
        ##For all the categories present in the model but not in the infer, fill 0
        temp_cols2 = [each for each in train_cols if each not in df_data_model_ip.columns]
        temp_arr = np.zeros([df_data_model_ip.shape[0],len(temp_cols2)])
        temp_df = pd.DataFrame(temp_arr,columns = temp_cols2,index=df_data_model_ip.index)
        vph.log('*'*10+"\nNumber of columns added to the final test data: " + str(len(temp_cols2)))
    
        ##Store the columns that were missing from the Infer data
        with open('Missing_Categories_{}_{}.txt'.format(run_type,train_identifier), 'w') as f:
            for item in temp_cols2:
                f.write("%s\n" % item) 
        
        ##Combine the temp_df created above to the df_data_model_ip
        df_data_model_ip = pd.concat([df_data_model_ip,temp_df],axis = 1)
        
        ##Check again to see if now the infer columns match the train columns
        if len(df_data_model_ip.columns) < len(train_cols):
            vph.log('*'*10+"\nThe length of the columns still do not match. Please check the code/data and try again.")
            sys.exit(2)
    ##################################################################################
    ##Start preparation for model testing/infer
    ##################################################################################
    if run_type == 'Infer':
        X_test = df_data_model_ip 
    elif run_type == 'Test':
        X_test = df_data_model_ip.iloc[:,1:]#all rows, second column through:last column
        y_test = df_data_model_ip[['status']]

    test_index = X_test.index
    test_cols = X_test.columns
    ##Cast the entire preprocessed test to numeric and apply the scaler object that was fit on the train data
    #####TBD: Change this piece of code to add any other transformers that were applied to train data to the test data
    X_test = X_test.apply(pd.to_numeric)
    #X_test = X_test.astype(float)
    X_test = scaler.transform(X_test)
    ##Create a data frame for X_test with index and columns as previous X_test
    X_test = pd.DataFrame(X_test, index = test_index)
    X_test.columns = test_cols
    ##Order the columns as per the training columns to be scored
    
    X_test = X_test[train_cols]

    #Output correlations among input features
    #data_procesing_for_modeling.correlation_scores_and_plot(X_train, dataset='Train')

    vph.log('*'*10 + "\nModel testing starts now.",flush=True)
    
    #Number of columns in the test set
    vph.log('*'*10+'\n Number of columns in the test set: '+str(X_test.shape[1]), flush = True)

    #Testing
    clf_predict       = model.predict(X_test) #predict test data using built model
    clf_predict_prob  = model.predict_proba(X_test) #predict test data and gettheir probability scores
    clf_model_classes = list(model.classes_) #probabilities are in the order of class labels in "classes_", so getting these labels and their order

    #Let's create a dataframe to hold prediction records
    y_test_prediction_df = pd.DataFrame({'predicted_label': clf_predict})
    #Let's make the index of this same as y_test, because it is for the same records
    y_test_prediction_df.index = X_test.index
    #Let's also include the original test labels in this dataframe, so we have everything in the same place
    if run_type == 'Test':
        y_test_prediction_df['original_labels'] = y_test
    #Let's create a column with probability 
    #STEP1: Let's get probabilities for each class
    temp_df = pd.DataFrame(clf_predict_prob, columns=clf_model_classes)
    #STEP2: Pick the highest probability value
    y_test_prediction_df['prediction_probability'] = temp_df.max(axis=1).values
    #del temp_df
    vph.log('*'*10 + "\nModel testing complete.",flush=True)
   

    temp_temp = y_test_prediction_df
    '''
    #Change the predicted label based on custom threshold determined as ideal for our classifier:
    custom_predict = pd.DataFrame(index = y_test_prediction_df.index, columns = ['custom_predicted']) 
    #Fill '0', i.e., predicted as 'ACTIVE'
    custom_predict = custom_predict.fillna(0)
    #Change '0' to '1', i.e., from 'ACTIVE' to 'NON-PAY' if original predicted label is 1 and associated probability score is what's given here:
    custom_predict.loc[(y_test_prediction_df['predicted_label'] == 1) & (y_test_prediction_df['prediction_probability'] > 0.59)] = 1

    #Swap the new predicted label with old version that was determined by the default threshold 0.5
    temp_temp['predicted_label'] = custom_predict['custom_predicted']
    '''
    #temp_temp['corp_house_cust'] = y_test_prediction_df.index
    temp_temp.insert(0,'corp_house_cust',y_test_prediction_df.index)
    
    final_output_df = temp_temp #The dataframe to write to BQ
    
    ##Break the corp_house_cust into individual columns
    final_output_df.loc[:,'corp'] = final_output_df.loc[:,'corp_house_cust'].apply(lambda x: str(x)[0:4])
    final_output_df.loc[:,'house'] = final_output_df.loc[:,'corp_house_cust'].apply(lambda x: vph.rev_lpad(str(x)[4:10],'0'))
    final_output_df.loc[:,'cust'] = final_output_df.loc[:,'corp_house_cust'].apply(lambda x: vph.rev_lpad(str(x)[10:],'0'))
    
    ##Get the list of columns that are going to go into the final table
    test_columns = list(final_output_df.columns)
    test_columns = test_columns[-3:] + test_columns[:-3]
    test_columns.insert(0,'test_identifier')
    test_columns.insert(0,'time_run')
    ##Remove the corp_house_cust column
    test_columns.remove('corp_house_cust')
    ##Add the identifier and the time run to the data
    final_output_df.loc[:,'time_run'] = time_run
    
    final_output_df.loc[:,'test_identifier'] = ui_identifier_str
    ##Rearrange the columns
    final_output_df = final_output_df[test_columns]
    
    final_results_table = 'voluntary_prediction_{}'.format(str.lower(run_type))

    key = 'model_results/'+str.lower(run_type)+'/'+ui_identifier_str+'.csv' #The key/file to which the dataframe will be dumped in BQ

    dest_table = final_results_table#Destination table name

    #def df_to_rs(source_df,key,dest_schema,dest_table,drop) (drop takes True or False, i.e., to drop existing table or just append it to )
    vph.log('*'*10+'\nThe dest table for test predictions is: '+dest_table)
    #pd2rs.df_to_rs(source_df,key,dest_schema,dest_table,True)
    util_obj.df_to_gcp(final_output_df,key,dest_table,False)
    
    return ui_identifier_str,final_output_df
    
    
        
    