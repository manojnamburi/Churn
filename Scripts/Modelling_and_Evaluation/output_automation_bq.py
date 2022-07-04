from datetime import datetime
import numpy as np
import pandas as pd
#import DATAFRAME_TO_REDSHIFT_TABLE_V2 as pd2rs
import sys
import os
#import boto3
from Class_GeneralUtilitiesNonTF import GeneralUtilitiesNonTF
import matplotlib.pyplot as plt
from google.cloud import storage

def create_output(key,Segment,n_var,n_iter,Description,test_month,atts,ui_identifier,train_churn,test_churn,bq_table,vol_view):
    print('*'*10+"\nStarting generation of model output statistics", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    ##Create a connection to the GCP
    bucket = 'alticeusa-am'
    project_id = 'alticeusa-am'
    dataset = 'poc'
    auth_file = 'alticeusa-am-b639e404289b.json' 
    
    gcs_client = storage.Client(project= project_id)
    
    ##File locations
    lift_file_location = 'model_results/lift_analysis/'
    image_path = 'model_results/churn_model_images/'
    output_log_path = 'model_results/churn_output_logs/'
    atts_location = 'model_results/churn_model_attributes/'
    
    #Instantiate the util obj that will be used to interact with BQ
    try:
        util_obj = GeneralUtilitiesNonTF(project_id = project_id,dataset = dataset,bucket_name = bucket,json_path = auth_file)
        print('*'*10+"\nInstantiated the utility object to interact with the Google Cloud Platform",flush = True)
        print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    except Exception as e:
        print('*'*10+"\nFailed to instantiate the utility object due to exception: " + str(e))
        
    output_dict = {}
    output_dict['Unique_Identifier'] = ui_identifier
    output_dict['Output_File_Name'] = key
    output_dict['Segment'] = Segment
    output_dict['No_of_variables'] = n_var
    output_dict['No_of_iterations'] = n_iter
    output_dict['Description'] = Description
    output_dict['Train_data_actives'] = train_churn.loc[0,'chc_id'].item()
    output_dict['Train_data_churners'] = train_churn.loc[1,'chc_id'].item()
    output_dict['Test_data_actives'] = test_churn.loc[0,'chc_id'].item()
    output_dict['Test_data_churners'] = test_churn.loc[1,'chc_id'].item()
    
    ##Calculate the actual test churn rate
    avg_churn_rate_actual_test = float(test_churn.loc[1,'chc_id'].item())/(float(test_churn.loc[1,'chc_id'].item()) + float(test_churn.loc[0,'chc_id'].item()))
    avg_churn_rate_actual_train = float(train_churn.loc[1,'chc_id'].item())/(float(train_churn.loc[1,'chc_id'].item()) + float(train_churn.loc[0,'chc_id'].item()))


    ts_string = str(datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S'))
    
    a = int(test_month) 
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
    list_3m = list(map(str,z[0:3]))
    str_3m = "('" + "','".join(list_3m) + "')"
    str_6m = "('" + "','".join(list_6m) + "')"
    
    print('*'*10 + "\nCalculating the conversion metrics for the following 3 months "+str_3m+" and 6 months: " + str_6m)
    
    print('*'*10 + "\nStarting the Conversion metrics calculation.", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    ##Monthly 
    monthly_conversion ="""select count(*) hh_count, 
    CASE WHEN prediction_probability >= 0.5 and prediction_probability < 0.6 THEN 'GTE_0.5 and LT_0.6 Count' 
    WHEN prediction_probability >= 0.6 and prediction_probability < 0.7 THEN 'GTE_0.6 and LT_0.7 Count' 
    WHEN prediction_probability >= 0.7 and prediction_probability < 0.8 THEN 'GTE_0.7 and LT_0.8 Count' 
    WHEN prediction_probability >= 0.8 and prediction_probability < 0.9 THEN 'GTE_0.8 and LT_0.9 Count' 
    WHEN prediction_probability >= 0.9 THEN 'GTE_0.9 Count'
    END prediction_probability,
    'Counts' pending_disco_month,
    1 rank_order
    from {bq_table} a 
    where predicted_label in (1)
    group by 2,3
    
    UNION DISTINCT
    
    select count(*) hh_count,
    CASE WHEN prediction_probability >= 0.5 and prediction_probability < 0.6 THEN 'GTE_0.5 and LT_0.6 Pending Disconnect' 
    WHEN prediction_probability >= 0.6 and prediction_probability < 0.7 THEN 'GTE_0.6 and LT_0.7 Pending Disconnect' 
    WHEN prediction_probability >= 0.7 and prediction_probability < 0.8 THEN 'GTE_0.7 and LT_0.8 Pending Disconnect' 
    WHEN prediction_probability >= 0.8 and prediction_probability < 0.9 THEN 'GTE_0.8 and LT_0.9 Pending Disconnect' 
    WHEN prediction_probability >= 0.9 THEN 'GTE_0.9 Pending Disconnect'
    END prediction_probability
    ,dt as pending_disco_month,
    2 rank_order
    from {bq_table} a
    JOIN (select distinct CAST(CONCAT(CAST(corp AS STRING), LPAD(house,6,'0'),LPAD(cust,2,'0')) as STRING) corp_house_cust, 
    min(dt) dt 
    from {view} 
    where (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') 
    and dt in {str_6m}
    group by 1)b ON a.corp_house_cust = b.corp_house_cust 
    where predicted_label in (1) 
    group by 2,3 
    
    UNION DISTINCT   
    
    select count(*) hh_count,
    CASE WHEN prediction_probability >= 0.5 and prediction_probability < 0.6 THEN 'GTE_0.5 and LT_0.6 Voluntary Disconnect' 
    WHEN prediction_probability >= 0.6 and prediction_probability < 0.7 THEN 'GTE_0.6 and LT_0.7 Voluntary Disconnect' 
    WHEN prediction_probability >= 0.7 and prediction_probability < 0.8 THEN 'GTE_0.7 and LT_0.8 Voluntary Disconnect' 
    WHEN prediction_probability >= 0.8 and prediction_probability < 0.9 THEN 'GTE_0.8 and LT_0.9 Voluntary Disconnect' 
    WHEN prediction_probability >= 0.9 THEN 'GTE_0.9 Voluntary Disconnect'
    END prediction_probability
    ,dt as pending_disco_month,
    3 rank_order
    from {bq_table} a
    JOIN (select distinct CAST(CONCAT(CAST(corp AS STRING), LPAD(house,6,'0'),LPAD(cust,2,'0')) as STRING) corp_house_cust, min(dt) dt 
    from {view} 
    where drform = 'VOLUNTARY' 
    and dt in {str_6m} 
    group by 1)b ON a.corp_house_cust = b.corp_house_cust where predicted_label in (1) 
    group by 2,3 
    
    UNION DISTINCT
    
    select count(*) hh_count, 
    'GTE_0.7 Count' prediction_probability,
    'Counts' pending_disco_month,
    1 rank_order
    from {bq_table} a 
    where predicted_label in (1)
    and prediction_probability >=0.7
    group by 2,3
    
    UNION DISTINCT
    
    select count(*) hh_count, 
    'GTE_0.8 Count' prediction_probability,
    'Counts' pending_disco_month,
    1 rank_order
    from {bq_table} a 
    where predicted_label in (1)
    and prediction_probability >=0.8
    group by 2,3
    
    UNION DISTINCT
    
    select count(*) hh_count,
    'GTE_0.7 Pending Disconnect' prediction_probability
    ,dt as pending_disco_month,
    4 rank_order
    from {bq_table} a
    JOIN (select distinct CAST(CONCAT(CAST(corp AS STRING), LPAD(house,6,'0'),LPAD(cust,2,'0')) as STRING) corp_house_cust, min(dt) dt 
    from {view} 
    where (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE')
    
    and dt in {str_6m}
    group by 1)b 
    ON a.corp_house_cust = b.corp_house_cust 
    where predicted_label in (1) 
    and prediction_probability >=0.7
    group by 2,3 
    
    UNION DISTINCT
    
    select count(*) hh_count,
    'GTE_0.8 Pending Disconnect' prediction_probability
    ,dt as pending_disco_month,
    4 rank_order
    from {bq_table} a
    JOIN (select distinct CAST(CONCAT(CAST(corp AS STRING), LPAD(house,6,'0'),LPAD(cust,2,'0')) as STRING) corp_house_cust, min(dt) dt 
    from {view} 
    where (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE')
    
    and dt in {str_6m}
    group by 1)b 
    ON a.corp_house_cust = b.corp_house_cust 
    where predicted_label in (1) 
    and prediction_probability >=0.8
    group by 2,3 
    
    order by 4,3 asc,2;
    """.format(view = vol_view, str_6m = str_6m, bq_table = bq_table)
    print(monthly_conversion)
    
    ##Monthly conversion df creation. Creating an empty df to get all possible rows and filling the ones missing with 0.
    
    conversion_monthly_def = pd.DataFrame(columns = ['prediction_probability'])
    
    monthly_df_rows = []

    monthly_df_rows.extend(["""GTE_0.5 and LT_0.6 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.6 and LT_0.7 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.7 and LT_0.8 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.7 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.8 and LT_0.9 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.8 Count(Counts)"""])
    monthly_df_rows.extend(["""GTE_0.9 Count(Counts)"""])

    monthly_df_rows.extend([ """GTE_0.5 and LT_0.6 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.6 and LT_0.7 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.7 and LT_0.8 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.7 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.8 and LT_0.9 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.8 Pending Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.9 Pending Disconnect("""+ x + """)"""  for x in list_6m])

    monthly_df_rows.extend([ """GTE_0.5 and LT_0.6 Voluntary Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.6 and LT_0.7 Voluntary Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.7 and LT_0.8 Voluntary Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.8 and LT_0.9 Voluntary Disconnect("""+ x + """)"""  for x in list_6m])
    monthly_df_rows.extend([ """GTE_0.9 Voluntary Disconnect("""+ x + """)"""  for x in list_6m]) 


    ui_list_6m = ['month_'+str(i) for i in range(1,7)]

    ui_monthly_rows = []

    ui_monthly_rows.extend(["""GTE_0.5 and LT_0.6 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.6 and LT_0.7 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.7 and LT_0.8 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.7 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.8 and LT_0.9 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.8 Count(Counts)"""])
    ui_monthly_rows.extend(["""GTE_0.9 Count(Counts)"""])

    ui_monthly_rows.extend([ """GTE_0.5 and LT_0.6 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.6 and LT_0.7 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.7 and LT_0.8 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.7 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.8 and LT_0.9 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.8 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.9 Pending Disconnect("""+ x + """)"""  for x in ui_list_6m])

    ui_monthly_rows.extend([ """GTE_0.5 and LT_0.6 Voluntary Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.6 and LT_0.7 Voluntary Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.7 and LT_0.8 Voluntary Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.8 and LT_0.9 Voluntary Disconnect("""+ x + """)"""  for x in ui_list_6m])
    ui_monthly_rows.extend([ """GTE_0.9 Voluntary Disconnect("""+ x + """)"""  for x in ui_list_6m]) 





    conversion_monthly_def = conversion_monthly_def.append(pd.DataFrame({'prediction_probability':monthly_df_rows}))
    conversion_monthly_def.set_index('prediction_probability',inplace=True)
    
    ##Clean up the names for bigquery. Remove all special characters and replace with _
    monthly_df_rows = [str(row).translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "}) for row in monthly_df_rows]

    ui_monthly_rows = [str(row).translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "}) for row in ui_monthly_rows]

    ##Create a dict object to rename the monthly columns
    ui_monthly_dict = dict(zip(monthly_df_rows,ui_monthly_rows))

    ##Get the output df which is the result of the above query
    try:
        output_df = util_obj.read_gbq_lite(monthly_conversion)
        if output_df is None:
            print('*'*10 + "\nFailed to fetch the conversion metrics from the bq. Please check the query",flush = True)
            sys.exit(1)
        else:
            print('*'*10 + "\nSuccessfully fetched the conversion data.", flush = True)
            print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    except Exception as e:
        print('*'*10 + "\nException occurred while fetching the conversion data: "+ str(e))
    
    ##Get 3m and 6m conversion metrics
    output_3m_df = output_df.loc[output_df['pending_disco_month'].isin(list_3m)].groupby('prediction_probability').agg({'hh_count':sum}).reset_index()
    output_3m_df['prediction_probability'] = output_3m_df['prediction_probability'] + ' (3m CR)'
    
    output_6m_df = output_df.loc[output_df['pending_disco_month'].isin(list_6m)].groupby('prediction_probability').agg({'hh_count':sum}).reset_index()
    output_6m_df['prediction_probability'] = output_6m_df['prediction_probability'] + ' (6m CR)'
    
    ##Get the monthly conversion df by joining the empty df and the result of output df
    output_df['prediction_probability'] = output_df['prediction_probability'] + '(' + output_df['pending_disco_month'] + ')'
    #output_df.set_index('prediction_probability',inplace = True)
    del output_df['pending_disco_month']
    
    conversion_monthly_df = conversion_monthly_def.join(output_df.set_index('prediction_probability'),rsuffix = '_res',how = 'left',sort = False)
    conversion_monthly_df['hh_count'].fillna(0,inplace = True)
    conversion_monthly_df.reset_index(inplace = True)
    
    
    
    ##Concat all 3 to get the final df
    final_output_df = pd.concat([conversion_monthly_df[['prediction_probability','hh_count']],output_3m_df,output_6m_df]).reset_index(drop = True)
    
    final_output_df['prediction_probability'] = final_output_df['prediction_probability'].apply(lambda x: str(x).translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "}))
    
    ##Create a dictionary object of the conversion dictionary
    conversion_dict = dict(zip(final_output_df.prediction_probability, final_output_df.hh_count))
    
    ##Add the conversion dict to the output dict
    output_dict.update(conversion_dict)
    
    
    
    
    print('*'*10 + "\nCompleted the conversion metrics calculation",flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    print('*'*10 + "\nStarting model performance metric calculation",flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    ##Read in the bq_table for calculation of metrics and for lift metrics
    pred_df_sql = """
    SELECT * from {}
    """.format(bq_table)
    y_test_prediction_df = util_obj.read_gbq_lite(pred_df_sql)
    
    ##Correct the prediction_probability for AUC and lift
    y_test_prediction_df0 = y_test_prediction_df.loc[y_test_prediction_df['predicted_label']==0,:]
    y_test_prediction_df1 = y_test_prediction_df.loc[y_test_prediction_df['predicted_label']==1,:]

    y_test_prediction_df0.loc[:,'prediction_probability'] = 1- y_test_prediction_df0['prediction_probability']

    y_test_prediction_df = pd.concat([y_test_prediction_df0,y_test_prediction_df1])
    
    ##Metric calculations
    tp = len(y_test_prediction_df[(y_test_prediction_df['original_labels'] == 1) & \
                              (y_test_prediction_df['predicted_label'] == 1)])
    
    tn = len(y_test_prediction_df[(y_test_prediction_df['original_labels'] == 0) & \
                              (y_test_prediction_df['predicted_label'] == 0)])

    fp = len(y_test_prediction_df[(y_test_prediction_df['original_labels'] == 0) & \
                              (y_test_prediction_df['predicted_label'] == 1)])

    fn = len(y_test_prediction_df[(y_test_prediction_df['original_labels'] == 1) & \
                              (y_test_prediction_df['predicted_label'] == 0)])
    print ('*'*10 + "\nTest statistics.",flush=True)  
    try:
        tpr = tp/(tp+fn)
    except ZeroDivisionError as err:
        print('tpr:', err)
        tpr = 0.0

    try:    
        tnr = tn/(tn+fp)
    except ZeroDivisionError as err:
        print('tnr:', err)
        tnr = 0.0

    try:    
        fpr = fp/(tn+fp)
    except ZeroDivisionError as err:
       print('fpr:', err,flush = True)
       fpr = 0.0
    
    # accuracy = (tpr+tnr)/2

    accuracy = (tp + tn)/(tp+tn+fp+fn)

    balanced_accuracy = (tpr+tnr)/2 

    try:
        precision_metric = tp/(tp+fp)
    except ZeroDivisionError as err:
        print('precision_metric:', err)
        precision_metric = 0.0
    
    try:        
        recall_metric = tp/(tp+fn)
    except ZeroDivisionError as err:
        print('recall_metric:', err) 
        recall_metric = 0.0
        
    #MCC:
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'])
    mcc_weighted = matthews_corrcoef(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'])#, sample_weight = weights)
    
    #F1 Score:
    from sklearn.metrics import f1_score
    regular_f1_score = f1_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], average=None) #average='macro' averages both without weighing claas imbalance
    weighted_avg_f1_score = f1_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], average='weighted')

    #Precision
    from sklearn.metrics import precision_score
    precision_ACTIVE = precision_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], pos_label = 0, average = 'binary')
    precision_NON_PAY = precision_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], pos_label = 1, average = 'binary')
    weighted_avg_precision = precision_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], average='weighted')

    #Recall
    from sklearn.metrics import recall_score
    recall_ACTIVE = recall_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], pos_label = 0, average = 'binary')
    recall_NON_PAY = recall_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], pos_label = 1, average = 'binary')
    weighted_avg_recall = recall_score(y_test_prediction_df['original_labels'], y_test_prediction_df['predicted_label'], average='weighted')

    #AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test_prediction_df['original_labels'],y_test_prediction_df['prediction_probability'])

    f_active = 2*(precision_ACTIVE*recall_ACTIVE)/(recall_ACTIVE + precision_ACTIVE)
    f_churn = 2*(precision_NON_PAY*recall_NON_PAY)/(recall_NON_PAY + precision_NON_PAY)
    
    metrics_names = [
    'TPR',
    'TNR',
    'FPR',
    'Accuracy',
    'Balanced_Accuracy',
    'AUC',
    'Precision_ACTIVE',
    'Precision_CHURN',
    'Precision_Weighted_Avg',
    'Recall_ACTIVE',
    'Recall_CHURN',
    'Recall_Weighted_Avg',
    'F1_ACTIVE',
    'F1_CHURN',
    'F1_Weighted_Avg',
    'MCC']



    metrics_list = [float(tpr),
    float(tnr),
    float(fpr),
    float(accuracy),
    float(balanced_accuracy),
    float(auc),
    float(precision_ACTIVE),
    float(precision_NON_PAY),
    float(weighted_avg_precision),
    float(recall_ACTIVE),
    float(recall_NON_PAY),
    float(weighted_avg_recall),
    float(f_active),
    float(f_churn),
    float(weighted_avg_f1_score),
    float(mcc)]

    metrics_dict = dict(zip(metrics_names,metrics_list))
    
    print('*'*10 + "\nCalculation of the model performance metrics complete.", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    output_dict.update(metrics_dict)
    ##Get the output df from the output dict
    ##Convert the metrics types as well as some beautification to reduce the precision of the metrics
    output_df_temp = pd.DataFrame(list(output_dict.items()))
    output_df_temp.columns = ['Metric','Value']
    output_df = output_df_temp.set_index('Metric').T
    output_df[list(output_df.columns)] = output_df[list(output_df.columns)].apply(pd.to_numeric,errors='ignore',downcast='integer')
    metrics_cols = list(metrics_dict.keys())
    output_df[metrics_cols] = output_df[metrics_cols].apply(lambda x:round(x,4))
    
    final_output_cols = [
            'Unique_Identifier',
            'Output_File_Name',
            'Segment',
            'Train_data_actives',
            'Train_data_churners', 
            'Test_data_actives',
            'Test_data_churners',
            'No_of_variables',
            'No_of_iterations',
            'Description',
            'TPR',
            'TNR',
            'FPR',
            'Accuracy',
            'Balanced_Accuracy',
            'AUC',
            'Precision_ACTIVE',
            'Precision_CHURN',
            'Precision_Weighted_Avg',
            'Recall_ACTIVE',
            'Recall_CHURN',
            'Recall_Weighted_Avg',
            'F1_ACTIVE',
            'F1_CHURN',
            'F1_Weighted_Avg',
            'MCC']
    
    final_output_cols.extend(monthly_df_rows)
    output_df = output_df[final_output_cols]

    ##Rename the output columns with month 1 - 6 instead of the actual months
    output_df = output_df.rename(index = str, columns = ui_monthly_dict)
    
    #print(output_df.dtypes)
    print('#'*10+'\nOutput DF\n'+'#'*10)
    print(output_df.T,flush=True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

    
    output_df.to_csv("Output_current_run.csv", index = False)
    #write the atts to a new table and csv
    atts_df = pd.DataFrame({'unique_identifier':ui_identifier,'Attributes':atts})
    
    print('*'*10 + "\nUploading the model performance images (AUC and PRC) to GCS bucket.Please check for images in "+ image_path,flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    #####################################################################
    ##Image upload
    #####################################################################
    bucket = gcs_client.get_bucket('alticeusa-am')
    
    AUC_file_name = image_path + ui_identifier + '_AUC.png'
    PRC_file_name = image_path + ui_identifier + '_PRC.png'
    
    blob = bucket.blob(AUC_file_name)
    blob.upload_from_filename('AUC.png')
    
    blob = bucket.blob(PRC_file_name)
    blob.upload_from_filename('PRC.png')
    print('*'*10 + "\nLift analysis starts now.", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    ######################################################################
    ##Lift analysis
    ######################################################################
    # df_data = y_test_prediction_df
    # df_data.index = df_data['corp_house_cust']
    # del df_data['corp_house_cust']
    # df_data = df_data.sort_values('prediction_probability',ascending = False)
    # ##Cut the prediction probability by decile
    # df_data['decile'] = pd.qcut(df_data['prediction_probability'], 10, labels=False,duplicates='drop') + 1
    # lift_table=df_data.reset_index().groupby('decile', as_index=False).agg({'prediction_probability':[np.min,np.max],'original_labels':np.mean,'corp_house_cust':'count'})
    # lift_table.columns = ['decile','min_prob','max_prob','conv_3m','count_subs']
    
    # lift_table['avg_churn_rate_actual']=np.mean(lift_table.conv_3m)
    # lift_table['lift']=lift_table.conv_3m/lift_table.avg_churn_rate_actual
    
    # lift_table.loc[:,'Unique_Identifier'] = ui_identifier
    
    # lift_cols = list(lift_table.columns)
    # ##Move the unique_identifier to the first location
    # lift_cols.insert(0, lift_cols.pop())
    
    # lift_table = lift_table[lift_cols]
    
    # ##Write the lift table to file
    # lift_table.to_csv('Lift_Table_'+ts_string+'.csv',index=False)
    
    # print('*'*10 + "\nUploading the lift images to GCS bucket at location: "+lift_file_location, flush = True)
    # print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    # ##Plot the lift chart
    # lt = lift_table.plot(x='decile',y='lift', kind='bar')
    # fig = lt.get_figure()
    # fig.savefig('lift_all_deciles.png')
    # ##TBD: Add labels to the bars
    
    # ##Save the lift image to bucket
    # ALL_LIFT_file_name = image_path + ui_identifier + '_Lift_Overall.png'
    # blob = bucket.blob(ALL_LIFT_file_name)
    # blob.upload_from_filename('lift_all_deciles.png')
    
    ##Get top 10000 for deciling 
    ##TBD:parameterize the 10000
    
    top_n = 10000
   
    y_test_prediction_df1 = y_test_prediction_df1.sort_values('prediction_probability',ascending = False).reset_index(drop = True)

    ##Pick top 10000 or max of y_pred = 1 to get top decile df
    if y_test_prediction_df1.shape[0] < top_n:
        top_n = y_test_prediction_df1.shape[0]
        


    top_decile_df = y_test_prediction_df1.loc[0:top_n-1,:]
    
    top_decile_df.index = top_decile_df['corp_house_cust']
    del top_decile_df['corp_house_cust']
    #del top_decile_df['decile']
    
    #top_decile_df = top_decile_df.sort_values('prediction_probability',ascending = False)
    top_decile_df['decile'] = pd.qcut(top_decile_df['prediction_probability'], 10, labels=False,duplicates='drop') + 1
    lift_table_top_dec=top_decile_df.reset_index().groupby('decile', as_index=False).agg({'prediction_probability':[np.min,np.max],'original_labels':np.mean,'corp_house_cust':'count'})
    lift_table_top_dec.columns = ['decile','min_prob','max_prob','conv_3m','count_subs']
    lift_table_top_dec.loc[:,'avg_churn_rate_actual'] = avg_churn_rate_actual_test

    lift_table_top_dec['lift']=lift_table_top_dec.conv_3m/lift_table_top_dec.avg_churn_rate_actual
    lift_table_top_dec.loc[:,'Unique_Identifier'] = ui_identifier
    
    ##Send the ui_identifier to the start of the df
    lift_cols = list(lift_table_top_dec.columns)
    lift_cols.insert(0, lift_cols.pop())
    
    lift_table_top_dec = lift_table_top_dec[lift_cols]
    
    lift_table_top_dec.to_csv('Lift_top_decile_'+ts_string+'.csv',index=False)
    
    lt = lift_table_top_dec.plot(x='decile',y='lift', kind='bar')
    fig = lt.get_figure()
    fig.savefig('lift_top_deciles.png')
    
    ##Save the lift image to bucket
    TOP_LIFT_file_name = image_path + ui_identifier + '_Lift_top_decile.png'
    blob = bucket.blob(TOP_LIFT_file_name)
    blob.upload_from_filename('lift_top_deciles.png')
    
    print('*'*10 + "\nCompleted the lift analysis.", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    
    print('*'*10 + "\nUploading the dfs to respective tables.", flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    ###Upload the output df to bigquery
    output_key = output_log_path + 'Output_'+ts_string+'.csv'
    util_obj.df_to_gcp(output_df,output_key,'churn_output_log',False)
    
    ##Upload the lift table to bigquery
    #lift_key = lift_file_location + 'Lift_All_deciles_' + ts_string + '.csv'
    #util_obj.df_to_gcp(lift_table,lift_key,'lift_all_deciles',False)
    
    ##Upload the top decile to bigquery
    lift_top_key = lift_file_location + 'Lift_Top_deciles_' + ts_string + '.csv'
    util_obj.df_to_gcp(lift_table_top_dec,lift_top_key,'lift_top_decile',False)
    
    ##Upload the attributes df to bigquery
    atts_key = atts_location + 'Model_Attributes_' + ts_string + '.csv'
    util_obj.df_to_gcp(atts_df,atts_key,'churn_model_attributes',False)
    
    print('*'*10 + '\nOutput log generation complete.', flush = True)
    print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
    print('*'*30,flush = True)
    
    return
    


    
    