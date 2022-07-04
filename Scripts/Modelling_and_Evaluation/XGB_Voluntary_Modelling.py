# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:44:04 2018

@author: syadalam
"""

from datetime import datetime
print('*'*10 + "\nProgram start time:", flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
#import DATAFRAME_TO_REDSHIFT_TABLE_V2 as pd2rs
#def df_to_rs(source_df,key,dest_schema,dest_table,drop) (drop takes True or False, i.e., to drop existing table or just append it to )
#import boto3

#from churn_model_prep import model_prep
#print('*'*10 + "\nDone importing the list of dummied categories that are available in both TRAIN & TEST data.", flush=True)
#import psycopg2
import os
import string
import numpy as np
import pandas as pd
#import pandas_gbq as gbq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##From version 0.23 onwards this is deprecated
#from sklearn.externals import joblib

import joblib

from attribute_dictionary import attribute_dict
from imputation_dictionary import attribute_imputer_dict
import data_procesing_for_modeling

from Class_GeneralUtilitiesNonTF import GeneralUtilitiesNonTF

import output_automation_bq as oa
import sys

##Adding the use of create data instead of using pre-created tables
import voluntary_prediction_create_data as vcd
import voluntary_prediction_helpers as vph
from config_dict import config_dict

#import post_modelling_processing_v2 as pmp

#import boto3

#For automation of conversion rates
#Update the parameters mentioned below for each model



Segment = 'OPTION3'

n_var = int(sys.argv[2])
n_iter = 50
train_month = int(sys.argv[1])
#test_month = int(sys.argv[2]) #this is used to calculate conversion months dynamically
target_var = 'status'

train_str = datetime.strptime(str(train_month),"%Y%m")
train_str = datetime.strftime(train_str,'%b%y').lower()

#test_str = datetime.strptime(str(test_month),"%Y%m")
#test_str = datetime.strftime(test_str,'%b%y').lower()

#Description = 'Voluntary Pending Disconnect, Train Data = {} Actives and churners from next three months, Test Data = {} Actives and churners from next three months, feature selection = XGB top {}, imputation = default_dictionary, hyper-params = bayesian 50 samples, Comments = optimized depth and colsample values to avoid overfit. Including false movers. Updated test logic. Used NTM view. No intersection.'.format(train_str,test_str,n_var)

# table_name = 'YST_OPTION3'
# train_data = table_name + '_' + 'TRAIN_NTM_OCT18'
# test_data = table_name + '_' + 'TEST_NTM_FEB19'
feature_imp_file = 'Feature_Ranking_Using_Gain_{}_NTM_{}_nogeo.csv'.format(Segment,train_str)

seg_list = ['Segment1','Segment2','Segment3','FIOS_ONT_G1_4','FIOS_ONT_G4_8','FIOS_COMP_G1_4']
View = 'v_vol_churn_tab'

####Read in the feature selection csv and pick the n_vars 
fs_data = pd.read_csv(feature_imp_file)
all_atts = list(fs_data['Attribute'].unique())
topn_features = all_atts[0:n_var]
#top_atts_cat = list(topn_features['Attribute_Category'].unique())
query_vars = str(target_var+','+'chc_id,cust,') + str(",".join(topn_features))

####Generate a unique_identifier to use across all the outputs the model will generate
#UI Identifier for model
##Model name = Algo used + Segment + nvar + train date
model_name = '_'.join(['XGB',Segment.replace(' ','_'),str(n_var),'vars','train',str(train_month)])
##Add a timestamp to make it unique
ui_identifier_str = str(model_name+'_'+str(datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')))



# ########################################################################################################
# #Connect to Redshift DR
# ########################################################################################################
# database = 'Redshift PROD-DR | SPSS'

# print ('*'*10 + "\nHold on to your pants amigo -- connecting to %s via psycopg2." %(database), flush=True)
# print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
# try:
#     cnxn = psycopg2.connect(database='cvrsamp',host='rsamp1d.czfsgsdwh1wy.us-east-1.redshift.amazonaws.com',port='5450',user='syadalam',password='Sai@1234')
#     curs = cnxn.cursor()
#     print ('*'*10 + "\nConnected to %s!"%(database), flush=True)
#     print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
# except:
#     print('*'*10 + "\nDid not connect to %s -- something fishy is going on with your connection." %(database), flush=True)
#     print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
#######################################################################################################
#Declare BigQuery Variables
#######################################################################################################
bucket = 'alticeusa-am'
project_id = 'alticeusa-am'
dataset = 'poc'
auth_file = 'alticeusa-am-b639e404289b.json'
#Instantiate the util obj that will be used to interact with BQ
util_obj = GeneralUtilitiesNonTF(project_id = project_id,dataset = dataset,bucket_name = bucket,json_path = auth_file)

########################################################################################################
##Model preparation
########################################################################################################

#final_columns = model_prep(query_vars,train_data,test_data,Segment)



##################################################################################################################################
##TRAIN DATA SET UP
##################################################################################################################################
# sql_data = str(""" 
              
# select {} FROM {}.{}
# WHERE Segment = '{}';

# """.format(query_vars,dataset,train_data,Segment))

#df_data = util_obj.read_gbq(sql_data)
##Leveraging create data function
final_df = pd.DataFrame()
for seg in seg_list:
    seg_data = vcd.create_voluntary_prediction_data(bucket,project_id,dataset,auth_file,seg,View,train_month,'Train',topn_features)
##Check if this is an empty dataframe and end the run if it is
    if seg_data.shape[0] == 0:
        print('*'*10+"\nEmpty dataframe returned from create data method. Please check the data creation.")
        sys.exit(4)
    else:
        final_df = final_df.append(seg_data)

final_df = final_df.set_index('chc_id')
df_data = final_df[~final_df.index.duplicated(keep='first')].copy()
df_data = df_data.reset_index(drop = False)



print('*'*10 + '\nDone importing data. Number of records: ' + str(len(df_data)), flush=True)

attributes = topn_features

#data_procesing_for_modeling.get_value_counts(df_data,'Train')
#df_data.to_csv('train.csv',index=False)

##Check to see if there are any duplicate chc_id
if len(df_data)!= len(df_data['chc_id'].unique()):
    print('*'*10+"\nDuplicate chc_ids found.")
    sys.exit(1)
else:
    print('*'*10+"\nNo duplicate ids found.")
    


#Let's make index of the dataframe as corp_house_cust
df_data.index = df_data['chc_id']

df_data = df_data.sort_index()
#Let's remove redundant columns
###Changing the sql to not read in the redundant columns
del df_data['chc_id']
########################################################################################################
#Attribute quirks
########################################################################################################
#Unfortunately, some NUM columns are being interpreted as string, let's cast them back as numeric
try:df_data['cust'] = pd.to_numeric(df_data['cust'], errors='coerce')
except:print('')

try:df_data['roll_off_lift_m1'] = pd.to_numeric(df_data['roll_off_lift_m1'], errors='coerce')
except:print('')
try:df_data['roll_off_lift_m2'] = pd.to_numeric(df_data['roll_off_lift_m2'], errors='coerce')
except:print('')
try:df_data['roll_off_lift_m3'] = pd.to_numeric(df_data['roll_off_lift_m3'], errors='coerce')
except:print('')
try:df_data['roll_off_lift_m4'] = pd.to_numeric(df_data['roll_off_lift_m4'], errors='coerce')
except:print('')

try:df_data['vidpromo_mthsleft_m1'] = pd.to_numeric(df_data['vidpromo_mthsleft_m1'], errors='coerce')
except:print('')
try:df_data['vidpromo_mthsleft_m2'] = pd.to_numeric(df_data['vidpromo_mthsleft_m2'], errors='coerce')
except:print('')
try:df_data['vidpromo_mthsleft_m3'] = pd.to_numeric(df_data['vidpromo_mthsleft_m3'], errors='coerce')
except:print('')
try:df_data['vidpromo_mthsleft_m4'] = pd.to_numeric(df_data['vidpromo_mthsleft_m4'], errors='coerce')
except:print('')
########################################################################################################
#Imputation & all other data preperation good stuff for modeling
########################################################################################################
df_data_model_ip = data_procesing_for_modeling.process_data(df_data, dataset='Train')
# temp_cols = [each for each in df_data_model_ip.columns if each in final_columns]# limit to only attribute-category combinations available in both TRAIN & TEST
# df_data_model_ip = df_data_model_ip[temp_cols]
#data_procesing_for_modeling.correlation_scores_and_plot(df_data_model_ip, dataset='Train')

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
########################################################################################################
#Class Balancing for TRAIN data only
########################################################################################################
X_train = df_data_model_ip.iloc[:,1:] #all rows, second column through:last column
#y = df_data.iloc[:,[0]] #first column as class label. let's not save a series: df_data.iloc[:,[0]]
y_train = df_data_model_ip[['status']]
train_columns = list(X_train.columns)
train_index = X_train.index
joblib.dump(train_columns,'train_columns_{}.pkl'.format(ui_identifier_str))
train_churn = y_train.reset_index(drop=False).groupby('status').agg('count')

#Standardization:
#X_train = (X_train - X_train.mean().values) / X_train.std().values

from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler().fit(X_train)
joblib.dump(std_scale,'standard_scaler_{}.pkl'.format(ui_identifier_str))
X_train = std_scale.transform(X_train)
X_train = pd.DataFrame(X_train,index = train_index)
X_train.columns = train_columns
print ('*'*10 + "\nDone setting up train data for model.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

print ('*'*10 + "\nModel training starts now.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

#Train columns
print('*'*10 + "\nNumber of training columns going into the model: "+ str(X_train.shape[1]), flush =True)


#Get the value counts in the train data


#Let's build an XGB model optimized by Bayesian Hyperparameter tuning
import xgboost
from xgboost import XGBClassifier
from skopt import BayesSearchCV #Damn thing seems to break with scikit-learn 0.20.1. Worked before, so rolled back scikit-learn to 0.19.2 on server and it works now.
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report



hyper_parm_optimizer = BayesSearchCV(

estimator =  XGBClassifier(random_state = 25,
                     verbose = True,
                     n_jobs = 28,
                     objective = 'binary:logistic',
                     eval_metric = ['auc'],
                     silent = 0,
                     tree_method = 'approx'),

search_spaces = {
        'learning_rate': (0.01, 1.0),# 'log-uniform'),
        'min_child_weight': (0, 5),#, 10),
        'max_depth': (3, 5),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0),# 'uniform'),
        'colsample_bytree': (0.01, 0.5),# 'uniform'),
        'colsample_bylevel': (0.01, 0.5),# 'uniform'),
        'reg_lambda': (1e-9, 1000),# 'log-uniform'),
        'reg_alpha': (1e-9, 1.0),# 'log-uniform'),
        'gamma': (1e-9, 0.5),# 'log-uniform'),
        'n_estimators': (10, 100),#
        'scale_pos_weight': (len(y_train[y_train.iloc[:,0] == 0])/len(y_train[y_train.iloc[:,0] == 1]),len(y_train[y_train.iloc[:,0] == 0])/len(y_train[y_train.iloc[:,0] == 1]) + 8)
            },
        
    scoring = 'roc_auc',
    n_jobs = -1,
    verbose = True,

cv =  StratifiedShuffleSplit(
        random_state = 25,
        n_splits = 3,
        train_size = 0.8),
        
     #https://scikit-optimize.github.io/
#    n_iter = 50,   
    n_iter = 50,   
    refit = True,
    random_state = 25)

#Function to print parameter-combination results
def check_status(optim_result):
        
    #Store models tested thus far in a DF
    all_models = pd.DataFrame(hyper_parm_optimizer.cv_results_)    
    
    #Store best parameters    
    best_params = pd.Series(hyper_parm_optimizer.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(hyper_parm_optimizer.best_score_, 4),
        hyper_parm_optimizer.best_params_
    ))
    
    #Save model results
    #clf_name = hyper_parm_optimizer.estimator.__class__.__name__
    all_models.to_csv("{}_cv_results.csv".format(ui_identifier_str))
    

clf_model = hyper_parm_optimizer.fit(X_train, y_train, callback = check_status) #https://scikit-optimize.github.io/#skopt.BayesSearchCV | def fit(self, X, y=None, groups=None, callback=None)

run_results_df = pd.read_csv("{}_cv_results.csv".format(ui_identifier_str))
run_results_df = run_results_df.sort_values(by = 'mean_test_score',ascending=False).reset_index()

#Use this after the hyperparameters are taken from the log file of the Experiment script that runs Bayesian hyperparameter optimization (above piece of code)
clf =  XGBClassifier(random_state = 25,
                     seed = 25, #it's confusing if this thing is still relavent with random_state now available
                     verbose = True,
                     n_jobs = -1,
                     objective = 'binary:logistic',
                     eval_metric = ['auc','aucpr'],
                     silent = 0,
                     tree_method = 'exact',#'approx',
                     colsample_bylevel = run_results_df.loc[0,'param_colsample_bylevel'],
                     colsample_bytree = run_results_df.loc[0,'param_colsample_bytree'],
                     gamma = run_results_df.loc[0,'param_gamma'],
                     learning_rate = run_results_df.loc[0,'param_learning_rate'],
                     max_delta_step = run_results_df.loc[0,'param_max_delta_step'].astype('int'),
                     max_depth = run_results_df.loc[0,'param_max_depth'].astype('int'),
                     min_child_weight = run_results_df.loc[0,'param_min_child_weight'].astype('int'),
                     n_estimators = run_results_df.loc[0,'param_n_estimators'].astype('int'),
                     reg_alpha = run_results_df.loc[0,'param_reg_alpha'],
                     reg_lambda = run_results_df.loc[0,'param_reg_lambda'],
                     scale_pos_weight = run_results_df.loc[0,'param_scale_pos_weight'],
                     subsample = run_results_df.loc[0,'param_subsample'])#0.92)
'''
clf =  XGBClassifier(random_state = 25,
                     seed = 25, #it's confusing if this thing is still relavent with random_state now available
                     verbose = True,
                     n_jobs = -1,
                     objective = 'binary:logistic',
                     eval_metric = ['auc','aucpr'],
                     silent = 0,
                     tree_method = 'exact',#'approx',
                     colsample_bylevel = 0.5,
                     colsample_bytree = 0.5,
                     gamma = 0.5,
                     learning_rate = 0.212,
                     max_delta_step = 0,
                     max_depth = 3,
                     min_child_weight = 1,
                     n_estimators = 100,
                     reg_alpha = 0.480,
                     reg_lambda = 325.43,
                     scale_pos_weight = 65.10,
                     subsample = 0.439)
'''
clf_model = clf.fit(X_train, y_train)
print ('*'*10 + "\nModel training complete.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

########################################################################################################
#Get Feature Importances
########################################################################################################
#Let's get feature_importances of each attribute:
#for each in zip(X.columns.values,clf_model.feature_importances_.ravel()):print(each) #this wouldn't print in order of coeff value. let's fix that:
#Let's put feature_importances and their names in a list and sort by coeffs in descending order:
feature_importances_list = [each for each in zip(X_train.columns.values,clf_model.feature_importances_.ravel())]
feature_importances_list = list(sorted(feature_importances_list, key = lambda x: x[1], reverse=True))
for each in feature_importances_list: print(each)


###########################################################
#Training Metrics##########################################
###########################################################

#Testing
clf_predict       = clf_model.predict(X_train) #predict test data using built model
clf_predict_prob  = clf_model.predict_proba(X_train) #predict test data and gettheir probability scores
clf_model_classes = list(clf_model.classes_) #probabilities are in the order of class labels in "classes_", so getting these labels and their order

#Let's create a dataframe to hold prediction records
y_train_prediction_df = pd.DataFrame({'predicted_label': clf_predict})
#Let's make the index of this same as y_test, because it is for the same records
y_train_prediction_df.index = y_train.index
#Let's also include the original test labels in this dataframe, so we have everything in the same place
y_train_prediction_df['original_labels'] = y_train
#Let's create a column with probability 
#STEP1: Let's get probabilities for each class
temp_df = pd.DataFrame(clf_predict_prob, columns=clf_model_classes)
#STEP2: Pick the highest probability value
y_train_prediction_df['prediction_probability'] = temp_df.max(axis=1).values
#del temp_df
print ('*'*10 + "\nModel testing complete.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

#Let's setup a confusion matrix to beter interpret modeling results
#NON-PAY is +ve & ACTIVE is -ve (not using sklearn.metrics.confusion_matrix; just for the heck of it)
#NOTE: order of columns in metrics.confusion_matrix would be as if it were in clf_model.classes_

tp = len(y_train_prediction_df[(y_train_prediction_df['original_labels'] == 1) & \
                              (y_train_prediction_df['predicted_label'] == 1)])
    
tn = len(y_train_prediction_df[(y_train_prediction_df['original_labels'] == 0) & \
                              (y_train_prediction_df['predicted_label'] == 0)])

fp = len(y_train_prediction_df[(y_train_prediction_df['original_labels'] == 0) & \
                              (y_train_prediction_df['predicted_label'] == 1)])

fn = len(y_train_prediction_df[(y_train_prediction_df['original_labels'] == 1) & \
                              (y_train_prediction_df['predicted_label'] == 0)])
    
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


#Let's get some more metrics:
'''
#Let's get sample weights first:
weights = np.ones(y_test_prediction_df['original_labels'].shape[0]) #initialize an array of 1s

for each in y_test_prediction_df['original_labels'].value_counts().iteritems(): #update the weights array -> for each distinct class, weight = count of samples from that class divided by count of all samples from all classes
    #print (each[0], each[1])
    weights[y_test_prediction_df['original_labels'].ravel() == each[0]] *= (each[1]/float(len(y_test_prediction_df['original_labels'])))
'''
#MCC:
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'])
mcc_weighted = matthews_corrcoef(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'])#, sample_weight = weights)

#F1 Score:
from sklearn.metrics import f1_score
regular_f1_score = f1_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], average=None) #average='macro' averages both without weighing claas imbalance
weighted_avg_f1_score = f1_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], average='weighted')

#Precision
from sklearn.metrics import precision_score
precision_ACTIVE = precision_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], pos_label = 0, average = 'binary')
precision_NON_PAY = precision_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], pos_label = 1, average = 'binary')
weighted_avg_precision = precision_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], average='weighted')

#Recall
from sklearn.metrics import recall_score
recall_ACTIVE = recall_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], pos_label = 0, average = 'binary')
recall_NON_PAY = recall_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], pos_label = 1, average = 'binary')
weighted_avg_recall = recall_score(y_train_prediction_df['original_labels'], y_train_prediction_df['predicted_label'], average='weighted')


print('*'*10+'\n Training model fit statistics')
print ('--TPR: %.3f'%tpr)
print('Â¦')
print ('--TNR: %.3f'%tnr)
print('Â¦')
print ('--Precision: %.3f'%precision_metric)
print('Â¦')
print ('--Precision_ACTIVE: %.3f'%precision_ACTIVE)
print('Â¦')
print ('--Precision_NON_PAY: %.3f'%precision_NON_PAY)
print('Â¦')
print ('--Precision_Weighted_Avg: %.3f'%weighted_avg_precision)
print('Â¦')
print ('--Recall: %.3f'%recall_metric)
print('Â¦')
print ('--Recall_ACTIVE: %.3f'%recall_ACTIVE)
print('Â¦')
print ('--Recall_NON_PAY: %.3f'%recall_NON_PAY)
print('Â¦')
print ('--Recall_Weighted_Avg: %.3f'%weighted_avg_recall)
print('Â¦')
print ('--Balanced Accuracy: %.3f'%balanced_accuracy)
print('Â¦')
print ('--MCC: %.3f'%mcc)
print('Â¦')
print ('--F1 Score: %s = %.3f, %s = %.3f'%(clf_model.classes_[0], regular_f1_score[0], clf_model.classes_[1], regular_f1_score[1]))
print('Â¦')
print ('--F1 Score Weighted Average: %.3f'%weighted_avg_f1_score)

#classification_report(y_test_prediction_df['original_labels'],y_test_prediction_df['predicted_label'])

#Let's also get AUC printed:
print ('*'*10 + "\nROC Curve & AUC:", flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
#Man, scikit is so non-uniform:
#It handled labels in predictor variable well without the need to encode strings to integers, but
#But it doesn't do that for roc_auc_score & roc_curve methods! So doing it manually:
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train.values.ravel()) #let's fit to map classes in dependent variable to integers
#le.classes_ #ok, it captured both classes
le.transform #transforming
#Alright, looks like we can plot now:
y_pred_proba = clf_model.predict_proba(X_train)[::,1]
fpr, tpr, _ = roc_curve(le.transform(y_train.values.ravel()),  y_pred_proba)
auc = roc_auc_score(le.transform(y_train.values.ravel()), y_pred_proba)
plt.plot(fpr,tpr,label='XGB | Area = %0.2f' %auc, color='g')
plt.plot([0, 1], [0, 1],'--',label='Chance', color='r')
plt.xlim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 4)
#plt.box(False)
plt.show()
plt.savefig('AUC_train.png', format='png')
plt.close()

#Let's also get Precision-Recall Plot:
print ('*'*10 + "\nPrecision-Recall Plot.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(le.transform(y_train.values.ravel()),  y_pred_proba)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve') #: AP={0:0.2f}'.format(precision_recall_curve.average_precision))
#plt.box(False)
plt.show()
plt.savefig('PRC_train.png', format='png')
plt.close()

#Generate classifier metrics, using scikit's out-of-box functions (therefore, the appended '_f' to variable names):
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_train, clf_model.predict(X_train)).ravel()
print('TP: %d FP: %d' %(tp_f,fp_f))
print('FN: %d TN: %d' %(fn_f,tn_f))
print(classification_report(y_train, clf_model.predict(X_train)))
try: del df_data
except: print('*'*10 + '\nNothing to delete')

try: del df_data_cat_dummied
except: print('*'*10 + '\nNothing to delete')

try: del df_data_model_ip
except: print('*'*10 + '\nNothing to delete')

try: del class_label_df
except: print('*'*10 + '\nNothing to delete')

try: del X_train
except: print('*'*10 + '\nNothing to delete')

try: del y_train
except: print('*'*10 + '\nNothing to delete')
########################################################################################################
#STEP 3: WRITE RESULTS BACK TO REDSHIFT
########################################################################################################
temp_temp = y_train_prediction_df
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
temp_temp['corp_house_cust'] = y_train_prediction_df.index

source_df = temp_temp #The dataframe to write to Redshift

#Let's use the name of the script and datetime as the key (file in S3) and also the table name (in Redshift)
import sys
import os
x = sys.argv
y = os.path.basename(x[0]).strip('.py')

key = 'model_results/'+ui_identifier_str+'.csv' #The key/file to which the dataframe will be dumped to BQ

dest_table = ui_identifier_str#Destination table name
#def df_to_rs(source_df,key,dest_schema,dest_table,drop) (drop takes True or False, i.e., to drop existing table or just append it to )
print('*'*10+'The dest table for train predictions is'+dest_table)
#pd2rs.df_to_rs(source_df,key,dest_schema,dest_table,True)
util_obj.df_to_gcp(source_df,key,dest_table,True)
#source_df.to_csv(key,index = False)

# Pickle model to file
pkl_name = str(dest_table)+'.pkl'
try:
    joblib.dump(clf_model, pkl_name) 
    print('*'*10+"\nPickled the model object to: " + pkl_name)
    print('*'*10+"\nJoblib Version used to pickle: " + joblib.__version__)
    print('*'*10+"\nXGBoost Version used: " + xgboost.__version__)
   
    
except Exception as e:
    print('*'*10+"\nException occurred while pickling: " + str(e))
    