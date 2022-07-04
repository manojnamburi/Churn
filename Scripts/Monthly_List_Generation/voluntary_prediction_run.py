##Imports needed
import pandas as pd
#from sklearn.externals import joblib
import joblib ##From recent versions sklearn.joblib has been deprecated
from datetime import datetime
import sys
import re
import glob

##Modules needed
import voluntary_prediction_create_data as vcd
import voluntary_prediction_infer as vi
import voluntary_prediction_helpers as vph
from Class_GeneralUtilitiesNonTF import GeneralUtilitiesNonTF

from config_dict import config_dict

vph.log('*'*10 + "\nStarting voluntary infer:", flush=True)


##Steps in the script for each segment in the list/single segment:
##1.Read in the infer data using the data prep module
##2.Prepare the data using the data_processing_for_modelling module
##3.Read in the model objects saved in the storage
##4.Apply the objects using the infer method

#Input the list of segments for while the infer will be run. Specified as a string in the format '[List of segments separated by ',']'
if len(sys.argv)>3:
    input_str = sys.argv[3]
    cleaned_str = re.sub('\[|\]','',input_str)
    seg_list = [s for s in cleaned_str.split(',')]
else:
    seg_list = ['Segment1','Segment2','Segment3','FIOS_ONT_G1_4','FIOS_ONT_G4_8','FIOS_COMP_G1_4']


##Run type for infer/test runs
run_type = 'Infer'

##Month used to train the model
##Used to determine the directory from which the objects are fetched
if len(sys.argv)>=2:
    train_month = int(sys.argv[1])
else:
    train_month = 201810 ##Defaulting to the last used train date 

##Month for which the infer/test is being run
if run_type == 'Infer':
    
    if len(sys.argv)>=3:
        run_month = int(sys.argv[2])
        vph.log('*'*10+'\nRun date received in sys.argv: '+str(run_month))
        if run_month > int(datetime.strftime(datetime.now(),'%Y%m')):
            vph.log('*'*10+'\nIncorrect date entered. Please enter again. Value cannot be greater than the current month.')
            sys.exit(3)
    else:
        if datetime.now().month==1:
            run_month = int(datetime.strftime(datetime.now(),format='%Y%m')) - 100 + 11 ##Only at the start of the year, get the previous year and last month as run_month
        else:
            run_month = int(datetime.strftime(datetime.now(),format='%Y%m')) - 1
        vph.log(str('*'*10+'\nInsufficient number of arguments specified for the production month. Defaulting to one month before the current month ('+str(run_month)+')'))

elif run_type == 'Test':
    if len(sys.argv)>=2:
        run_month = int(sys.argv[1])
        if run_month > int(datetime.strftime(datetime.now(),'%Y%m')):
            vph.log('*'*10+'\nIncorrect date entered. Please enter again. Value cannot be greater than the current month.')
            sys.exit(3)
    else:
        vph.log(str('*'*10+'\nInsufficient number of arguments specified for test. Failing the process'))
        sys.exit(2)
else:
    vph.log('*'*10+"\nNot a valid run type specified. Failing the process")
    sys.exit(2)

    
##Current model uses top 100 features. Uncomment to pass it as a command-line argument.
#n_var = int(sys.argv[2])
n_var = 100

##Segment definition
Segment = 'OPTION3'

###############################################################################
##Common params for the infer run
###############################################################################
Bucket = config_dict['BUCKET']
Project_id = config_dict['PROJECT']
Dataset = config_dict['dataset']
Auth_file = config_dict['credentials_file']

###############################################################################
##Instantiate the util object to write the final target list
###############################################################################
util_obj = GeneralUtilitiesNonTF(project_id = Project_id,dataset = Dataset,bucket_name = Bucket,json_path = Auth_file)

###############################################################################
##Set the params for BQ connection
###############################################################################
res = vph.dbConnect()

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


###################################################################################################################
###Data read and preparation for the infer
###################################################################################################################
vph.log('*'*10+"\nStarting the data preparation process for the infer process")

##Master churn view for voluntary churn. Ensure it has chc_id
View = config_dict['vol_churn_view']


##Instantiate the final_df object
final_df = pd.DataFrame()
for seg in seg_list:
    seg_data = vcd.create_voluntary_prediction_data(Bucket,Project_id,Dataset,Auth_file,seg,View,run_month,run_type,topn_features)
    ##Check if this is an empty dataframe and end the run if it is
    if seg_data.shape[0] == 0:
        vph.log('*'*10+"\nEmpty dataframe returned from create data method. Please check the data creation.")
        sys.exit(4)
    else:
        final_df = final_df.append(seg_data)

##Remove the duplicates from the read data
final_df = final_df.set_index('chc_id')
infer_data = final_df[~final_df.index.duplicated(keep='first')].copy()

vph.log('*'*10+"\nNumber of unique subs:" + str(infer_data.shape[0]))
##Reset the index and remove the final_df object
del final_df
infer_data = infer_data.reset_index()

##Read in all the model objects
##Currently needs 3 objects other than the Feature selection csv - model, scaler and the training columns in the form of pickle
Model = joblib.load(model_obj)
Scaler = joblib.load(scaler_obj)
Train_cols = joblib.load(training_cols_obj)

###################################################################################################################
##Run the infer process
###################################################################################################################
##Format: Algorithm used_Segment_Train month_nvar
train_identifier = '{}_{}_train_{}_{}_vars'.format('XGB',Segment,str(train_month),str(n_var))
vph.log('*'*10+"\nStarting the infer process. Train string:" + train_identifier)
##The ui_string is the identifier generated after infer. The output df is the scored data. Pattern: train_identifier_Infer_run_month Example: XGB_OPTION3_201810_100_Infer_201902
ui_string,final_output_df = vi.score_data(Model,infer_data,Scaler,Train_cols,Project_id,Dataset,Bucket,Auth_file,run_month,run_type,train_identifier)

##Use the final_output_df to get the top 30000 target. 
ntarget = 30000
target_columns = ['time_run','run_month','corp','house','cust','corp_house_cust']
target_table = 'voluntary_target_list'

###Get the month string to prepare the filename
str_date = datetime.strptime(str(run_month),"%Y%m")
str_date = datetime.strftime(str_date,'%b%y').lower()


key = 'model_results/target/'+'voluntary_target_list_'+str_date+'.csv' ##File to which the df will be written to in the storage

vph.log('*'*10 + "\nFinal file name and location:" + key)
##Get the top ntarget customers who are predicted to churn
target_df = final_output_df.loc[final_output_df['predicted_label']==1,:]
target_df = target_df.sort_values(by = 'prediction_probability',ascending = False)
target_df = target_df.head(ntarget)
target_df['house'] = target_df['house'].apply(lambda x: "{0:0>6}".format(x))
target_df['cust']  = target_df['cust'].apply(lambda x: "{0:0>2}".format(x))
target_df['corp_house_cust'] = target_df['corp'].astype(str) + target_df['house'].astype(str) + target_df['cust'].astype(str)
target_df['run_month'] = str(run_month)
##Restrict the target_df to only the target columns
target_df = target_df.loc[:,target_columns]
util_obj.df_to_gcp(target_df,key,target_table,False)

