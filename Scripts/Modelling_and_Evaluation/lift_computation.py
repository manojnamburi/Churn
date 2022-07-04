# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:27:04 2021

@author: syadalam
"""
import pandas as pd
import numpy as np

##################################################################################################################################
###Compute the lift of Voluntary Churn model
##################################################################################################################################

##Compute lift from the predicted df generated from output_automation
def compute_lift(prediction_df):
    ##Compute the actual churn rate for the test data
    avg_churn_rate_actual_test = prediction_df.loc[prediction_df['original_labels'] == 1,:].shape[0]/prediction_df.shape[0]
    prediction_df.index = prediction_df['corp_house_cust']
    del prediction_df['corp_house_cust']
    ##Decile the prediction_df and compute the lift and gain metrics
    prediction_df.loc[:,'decile'] = pd.qcut(prediction_df['prediction_probability'], 10, labels=False,duplicates='drop') + 1
    
    lift_table=prediction_df.reset_index().groupby('decile', as_index=False).agg({'prediction_probability':[np.min,np.max],'original_labels':[lambda x: 100*np.mean(x),np.sum],'corp_house_cust':'count'})
    lift_table.columns = ['decile','min_prob','max_prob','conv_3m','count_churners','count_subs']
    lift_table.loc[:,'avg_churn_rate_actual'] = 100*avg_churn_rate_actual_test
    ##Sort the deciles in decreasing order of probability for the cumulative lift and gains
    lift_table = lift_table.sort_values('max_prob',ascending = False).reset_index(drop = True)
    ##Lift is the ratio between the decile churn rate and the overall churn rate
    lift_table.loc[:,'lift']=lift_table.conv_3m/lift_table.avg_churn_rate_actual
    
    ##Compute the cumulative lift and gains metrics
    ##Reference: https://www.listendata.com/2014/08/excel-template-gain-and-lift-charts.html
    
    lift_table.loc[:,'cumu_churners'] = pd.DataFrame.cumsum(lift_table['count_churners'])
    lift_table.loc[:,'cumu_subs'] = pd.DataFrame.cumsum(lift_table['count_subs'])
    lift_table.loc[:,'Gain'] = (lift_table['cumu_churners']*100)/max(lift_table['cumu_churners'])
    lift_table.loc[:,'cumu_subs_p'] = (lift_table['cumu_subs']*100)/max(lift_table['cumu_subs'])
    lift_table.loc[:,'cumu_lift'] = lift_table['Gain']/lift_table['cumu_subs_p']
    
    ##Format the gain and other columns as percentages
    format_mapping={'lift': '{:.2f}', 'cumu_lift': '{:.2f}', 'Gain': '{:.2f}%','cumu_subs_p': '{:.2f}%','avg_churn_rate_actual': '{:.2f}%','conv_3m': '{:.2f}%'}
    
    for key, value in format_mapping.items():
        lift_table.loc[:,key] = lift_table[key].apply(value.format)
    
    return lift_table


##For all subs and top decile
def get_all_lift(bq_table,util_obj):
    
    ##Read the results table from BQ
    pred_df_sql = """
                   SELECT * from {}
                   """.format(bq_table)
    y_test_prediction_df = util_obj.read_gbq_lite(pred_df_sql)
    
    ##Seperate out the 1s and 0s
    y_test_prediction_df1 = y_test_prediction_df.loc[y_test_prediction_df['predicted_label']==1,:]
    
    ##Reverse the predicted probability of the 0s as the sklearn outputs the probability as confidence in the label
    y_test_prediction_df0 = y_test_prediction_df.loc[y_test_prediction_df['predicted_label']==0,:]
    y_test_prediction_df0.loc[:,'prediction_probability'] = 1 - y_test_prediction_df0['prediction_probability']
    
    ##Combine the two 0 and 1 predition dfs
    y_test_prediction_df = pd.concat([y_test_prediction_df1,y_test_prediction_df0])
    
    y_test_prediction_df = y_test_prediction_df.sort_values('prediction_probability',ascending = False).reset_index(drop = True)
   
    ##Compute the lift metrics for entire data and top decile
    top_decile_df = y_test_prediction_df
    top_decile_df.loc[:,'decile'] = pd.qcut(top_decile_df['prediction_probability'], 10, labels=False,duplicates='drop') + 1
    
    top_decile_df = top_decile_df.loc[top_decile_df['decile']==10,:]
    del top_decile_df['decile']
    
    all_subs_lift = compute_lift(y_test_prediction_df)
    top_decile_lift = compute_lift(top_decile_df)
    
    
    
    return all_subs_lift,top_decile_lift