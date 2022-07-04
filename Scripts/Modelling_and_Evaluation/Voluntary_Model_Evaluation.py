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
from config_dict_test import config_dict
import glob

##Get the train and test month params
train_month = int(sys.argv[1])
test_month = int(sys.argv[2]) #this is used to calculate conversion months dynamically

train_str = datetime.strptime(str(train_month),"%Y%m")
train_str = datetime.strftime(train_str,'%b%y').lower()

test_str = datetime.strptime(str(test_month),"%Y%m")
test_str = datetime.strftime(test_str,'%b%y').lower()

target_var = 'status'
##Segment definition
Segment = 'OPTION3'
n_var = int(sys.argv[3])
n_iter = 50

seg_list = ['Segment1','Segment2','Segment3','FIOS_ONT_G1_4','FIOS_ONT_G4_8','FIOS_COMP_G1_4']
View = 'v_vol_churn_tab'

Description = 'Voluntary Pending Disconnect, Train Data = {} Actives and churners from next three months, Test Data = {} Actives and churners from next three months, feature selection = XGB top {}, imputation = default_dictionary, hyper-params = bayesian 50 samples, Comments = optimized depth and colsample values to avoid overfit. Including false movers. Updated test logic. Used NTM view. No intersection.'.format(train_str,test_str,n_var)

####Generate a unique_identifier to use across all the outputs the model will generate
#UI Identifier for model
##Model name = Algo used + Segment + nvar + train date
model_name = '_'.join(['XGB',Segment.replace(' ','_'),str(n_var),'vars','train',str(train_month)])
##Add a timestamp to make it unique
ui_identifier_str = str(model_name+'_'+str(datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')))

##Current model uses top 100 features. Uncomment to pass it as a command-line argument.
#n_var = int(sys.argv[2])




###############################################################################
##Common params for the infer run
###############################################################################
Bucket = config_dict['BUCKET']
Project_id = config_dict['PROJECT']
Dataset = config_dict['dataset']
Auth_file = config_dict['credentials_file']

###############################################################################
##Set the params for BQ connection
###############################################################################
res = vph.dbConnect()

###############################################################################
##Instantiate the util object to write the final target list
###############################################################################
util_obj = GeneralUtilitiesNonTF(project_id = Project_id,dataset = Dataset,bucket_name = Bucket,json_path = Auth_file)


vph.log('*'*10+'\nSteps in environment variables: \n1.Bucket \n2.Project \n3.Credentials file')

if res==0:
    vph.log('\nFailed to instantiate the environment variables for connecting to Google Environment. Failed at step '+ str(res+1))
    sys.exit(2)
else:
    vph.log('\nInstantiated the necessary variables for connecting to the Google Platform')

####################################################################################
##Retrieve the model objects from the storage
####################################################################################
##Location on Google storage specified relative to the bucket specified in the config_dict. Pattern: gs://bucket_name/voluntary_production_artifacts/Segment/train_month
obj_location = '/'.join([config_dict['voluntary_obj_folder'],Segment,str(train_month)])
try:
    vph.download_model_objs(obj_location,'voluntary_prediction')
except Exception as e:
    vph.log('*'*10+"\nCouldnt download the needed objects for the infer run due to an exception: " + str(e))
fs_file = glob.glob("temp_*feature_ranking*")[0]
model_obj = glob.glob("temp_*model*")[0]
training_cols_obj = glob.glob("temp_*train_columns*")[0]
scaler_obj = glob.glob("temp_*standard_scaler*")[0]

##Checks to ensure the objects are not empty due to no matching pattern
if (not fs_file) or (not model_obj) or (not training_cols_obj) or (not scaler_obj):
    vph.log('*'*10+"\nOne of the object needed for modelling is missing. Please check and try again")
    sys.exit(2)
else:
    vph.log('*'*10+"\nSuccessfully fetched the necessary objects and files for running the infer process")

########################################################################################
##Set up the attributes to be read in as part of infer
########################################################################################
fs_data = pd.read_csv(fs_file)
all_atts = list(fs_data['Attribute'].unique())
topn_features = all_atts[0:n_var]
attributes = topn_features

###################################################################################################################
###Data read and preparation for the test
###################################################################################################################
vph.log('*'*10+"\nStarting the data preparation process for the infer process")

##Master churn view for voluntary churn. Ensure it has chc_id
View = config_dict['vol_churn_view']

##Read the train data to get the number of churners
##Instantiate the final_df object
final_df = pd.DataFrame()
for seg in seg_list:
    seg_data = vcd.create_voluntary_prediction_data(Bucket,Project_id,Dataset,Auth_file,seg,View,train_month,'Train')
    ##Check if this is an empty dataframe and end the run if it is
    if seg_data.shape[0] == 0:
        vph.log('*'*10+"\nEmpty dataframe returned from create data method. Please check the data creation.")
        sys.exit(4)
    else:
        final_df = final_df.append(seg_data)

##Remove the duplicates from the read data
final_df = final_df.set_index('chc_id')
train_data = final_df[~final_df.index.duplicated(keep='first')].copy()

vph.log('*'*10+"\nNumber of unique subs:" + str(train_data.shape[0]))
##Reset the index and remove the final_df object
del final_df

##Drop the cust
train_data = train_data.drop('cust',axis = 1)

##Count of churners
train_churn = train_data.reset_index(drop=False).groupby('status').agg('count')

##Read in all the model objects
##Currently needs 3 objects other than the Feature selection csv - model, scaler and the training columns in the form of pickle
Model = joblib.load(model_obj)
Scaler = joblib.load(scaler_obj)
Train_cols = joblib.load(training_cols_obj)

##################################################################################################################################
##TEST DATA SET UP
##################################################################################################################################

###Set up the test data

final_df = pd.DataFrame()
for seg in seg_list:
    seg_data = vcd.create_voluntary_prediction_data(Bucket,Project_id,Dataset,Auth_file,seg,View,test_month,'Train',topn_features)
##Check if this is an empty dataframe and end the run if it is
    if seg_data.shape[0] == 0:
        print('*'*10+"\nEmpty dataframe returned from create data method. Please check the data creation.")
        sys.exit(4)
    else:
        final_df = final_df.append(seg_data)

final_df = final_df.set_index('chc_id')
df_data = final_df[~final_df.index.duplicated(keep='first')].copy()
df_data = df_data.reset_index(drop = False)

# sql_data = str(""" 
               
# select {} FROM {}.{}
# WHERE Segment = '{}';

# """.format(query_vars,dataset,test_data,Segment))
# df_data = util_obj.read_gbq(sql_data)
print('*'*10 + '\nDone importing test data. Number of records: ' + str(len(df_data)), flush=True)

##Check to see if there are any duplicate chc_id
if len(df_data)!= len(df_data['chc_id'].unique()):
    print('*'*10+"\nDuplicate chc_ids found.")
    sys.exit(1)
else:
    print('*'*10+"\nNo duplicate ids found.")

#data_procesing_for_modeling.get_value_counts(df_data,'Test')
#df_data.to_csv('test.csv',index=False)
#Let's make index of the dataframe as corp_house_cust
df_data.index = df_data['chc_id']

#Sort by hhid
df_data = df_data.sort_index()

#Let's remove redundant columns
###Changing the sql to not read in the redundant columns
#del df_data['corp']
#del df_data['house']
del df_data['chc_id']
########################################################################################################
#Attribute quirks
########################################################################################################
#Unfortunately, some NUM columns are being interpreted as string, let's cast them back as numeric
try:df_data['cust'] = pd.to_numeric(df_data['cust'], errors='coerce')
except: print('')
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
df_data_model_ip = data_procesing_for_modeling.process_data(df_data, dataset='Test')
# temp_cols = [each for each in df_data_model_ip.columns if each in final_columns]# limit to only attribute-category combinations available in both TRAIN & TEST
# df_data_model_ip = df_data_model_ip[temp_cols]



#data_procesing_for_modeling.correlation_scores_and_plot(df_data_model_ip, dataset='Test')
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
##########################################################################################################
###Prepare test data for the model - Categoricals handling
##########################################################################################################
test_cols = ['status'] + Train_cols
temp_cols = [each for each in df_data_model_ip.columns if each in test_cols]# limit to only attribute-category combinations available in both TRAIN & TEST
df_data_model_ip = df_data_model_ip[temp_cols]

if len(df_data_model_ip.columns) < len(Train_cols):
    ##For all the categories present in the model but not in the infer, fill 0
    temp_cols2 = [each for each in Train_cols if each not in df_data_model_ip.columns]
    temp_arr = np.zeros([df_data_model_ip.shape[0],len(temp_cols2)])
    temp_df = pd.DataFrame(temp_arr,columns = temp_cols2,index=df_data_model_ip.index)
    print('*'*10+"\nNumber of columns added to the final test data: " + str(len(temp_cols2)))

    ##Store the columns that were missing from the Infer data
    with open('Missing_Categories_Infer_{}.txt'.format(ui_identifier_str), 'w') as f:
        for item in temp_cols2:
            f.write("%s\n" % item) 
    
    ##Combine the temp_df created above to the df_data_model_ip
    df_data_model_ip = pd.concat([df_data_model_ip,temp_df],axis = 1)

        


########################################################################################################
#Modeling Ahoy!
########################################################################################################
#Class label is the first column, so split dependent and independent attributes accordingly
X_test = df_data_model_ip.iloc[:,1:] #all rows, second column through:last column
#y = df_data.iloc[:,[0]] #first column as class label. let's not save a series: df_data.iloc[:,[0]]
y_test = df_data_model_ip[['status']]

##Order the X_test like the train data
X_test = X_test[Train_cols]

test_cols = list(X_test.columns)
test_index = X_test.index
test_churn = y_test.reset_index(drop = False).groupby('status').agg('count')

#Standardization:
#X_test = (X_test - X_test.mean().values) / X_test.std().values

#X_test = X_test.astype(float)
X_test = Scaler.transform(X_test)

X_test = pd.DataFrame(X_test,index = test_index) ##Resetting the index as Standardscaler returns a np array
X_test.columns = test_cols
#Output correlations among input features
#data_procesing_for_modeling.correlation_scores_and_plot(X_train, dataset='Train')

print ('*'*10 + "\nModel testing starts now.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

#Number of columns in the test set
print('*'*10+'\n Number of columns in the test set: '+str(X_test.shape[1]), flush = True)

#Testing
clf_predict       = Model.predict(X_test) #predict test data using built model
clf_predict_prob  = Model.predict_proba(X_test) #predict test data and gettheir probability scores
clf_model_classes = list(Model.classes_) #probabilities are in the order of class labels in "classes_", so getting these labels and their order

#Let's create a dataframe to hold prediction records
y_test_prediction_df = pd.DataFrame({'predicted_label': clf_predict})
#Let's make the index of this same as y_test, because it is for the same records
y_test_prediction_df.index = y_test.index
#Let's also include the original test labels in this dataframe, so we have everything in the same place
y_test_prediction_df['original_labels'] = y_test
#Let's create a column with probability 
#STEP1: Let's get probabilities for each class
temp_df = pd.DataFrame(clf_predict_prob, columns=clf_model_classes)
#STEP2: Pick the highest probability value
y_test_prediction_df['prediction_probability'] = temp_df.max(axis=1).values
#del temp_df
print ('*'*10 + "\nModel testing complete.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)

#Let's setup a confusion matrix to beter interpret modeling results
#NON-PAY is +ve & ACTIVE is -ve (not using sklearn.metrics.confusion_matrix; just for the heck of it)
#NOTE: order of columns in metrics.confusion_matrix would be as if it were in clf_model.classes_

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
print ('--F1 Score: %s = %.3f, %s = %.3f'%(Model.classes_[0], regular_f1_score[0], Model.classes_[1], regular_f1_score[1]))
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
le.fit(y_test.values.ravel()) #let's fit to map classes in dependent variable to integers
#le.classes_ #ok, it captured both classes
le.transform #transforming
#Alright, looks like we can plot now:
y_pred_proba = Model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(le.transform(y_test.values.ravel()),  y_pred_proba)
auc = roc_auc_score(le.transform(y_test.values.ravel()), y_pred_proba)
#fig_AUC = plt.figure()
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
plt.savefig('AUC.png', format='png')
plt.close()

#Let's also get Precision-Recall Plot:
print ('*'*10 + "\nPrecision-Recall Plot.",flush=True)
print(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),flush=True)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(le.transform(y_test.values.ravel()),  y_pred_proba)
#fig_PRC = plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve') #: AP={0:0.2f}'.format(precision_recall_curve.average_precision))
#plt.box(False)
plt.show()
plt.savefig('PRC.png', format='png')
plt.close()

#Generate classifier metrics, using scikit's out-of-box functions (therefore, the appended '_f' to variable names):
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test, Model.predict(X_test)).ravel()
print('TP: %d FP: %d' %(tp_f,fp_f))
print('FN: %d TN: %d' %(fn_f,tn_f))
print(classification_report(y_test, Model.predict(X_test)))

f_active = 2*(precision_ACTIVE*recall_ACTIVE)/(recall_ACTIVE + precision_ACTIVE)
f_churn = 2*(precision_NON_PAY*recall_NON_PAY)/(recall_NON_PAY + precision_NON_PAY)
try:
    tpr = tp/(tp+fn)
except ZeroDivisionError as err:
   print('tpr:', err,flush = True)
   tpr = 0.0

try:    
    tnr = tn/(tn+fp)
except ZeroDivisionError as err:
    print('tnr:', err,flush = True)
    tnr = 0.0
    
try:    
    fpr = fp/(tn+fp)
except ZeroDivisionError as err:
   print('fpr:', err,flush = True)
   fpr = 0.0
'''
########################################################################################################
#Explore model classification metric by varying classification threshold
########################################################################################################
#Set threshold to different value from default:
#Let's loop through a bunch of thresholds:
for each_threshold in range(55,60,1):#range(50, 105, 5): 
    each_threshold = each_threshold/100
    
    #First, let's create a dummy array to hold modified predictor labels (modified in the sense, labels as a result of varying classifier threshold)
    custom_predict = pd.DataFrame(index = y_test_prediction_df.index, columns = ['custom_predicted']) 
    #Fill '0', i.e., predicted as 'ACTIVE'
    custom_predict = custom_predict.fillna(0)
    #Change '0' to '1', i.e., from 'ACTIVE' to 'NON-PAY' if original predicted label is 1 and associated probability score is what's given here:
    custom_predict.loc[(y_test_prediction_df['predicted_label'] == 1) & (y_test_prediction_df['prediction_probability'] > each_threshold)] = 1
    #Generate classifier metrics, for this custom threshold:
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test_prediction_df['original_labels'], custom_predict['custom_predicted']).ravel()
    
    print('*'*10 + '\nFor the classifier threshold: %f' %each_threshold)
    print('TP: %d FP: %d' %(tp_c,fp_c))
    print('FN: %d TN: %d' %(fn_c,tn_c))
    print(classification_report(y_test_prediction_df['original_labels'], custom_predict)) 
'''
########################################################################################################
#STEP 3: WRITE RESULTS BACK TO REDSHIFT
########################################################################################################
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
temp_temp['corp_house_cust'] = y_test_prediction_df.index

source_df = temp_temp #The dataframe to write to Redshift

#Let's use the name of the script and datetime as the key (file in S3) and also the table name (in Redshift)
import sys
import os
x = sys.argv
y = os.path.basename(x[0]).strip('.py')

key = 'model_results/'+ui_identifier_str+'_test_'+str(test_month)+'.csv' #The key/file to which the dataframe will be dumped in BQ
dest_schema = 'public' #Destination table schema
dest_table = ui_identifier_str+'_test_'+str(test_month)#Destination table name
redshift_table = dest_schema + '.' + dest_table
#def df_to_rs(source_df,key,dest_schema,dest_table,drop) (drop takes True or False, i.e., to drop existing table or just append it to )
print('*'*10+'The dest table for test predictions is'+dest_table)
#pd2rs.df_to_rs(source_df,key,dest_schema,dest_table,True)
util_obj.df_to_gcp(source_df,key,dest_table,True)

# Calling create_output function for generating output log
oa.create_output(key,Segment,n_var,n_iter,Description,test_month,attributes,ui_identifier_str,train_churn,test_churn,str(Dataset+'.'+dest_table),str(Dataset+"."+View))