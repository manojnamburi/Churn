import pandas as pd
import numpy as np
import datetime
from attribute_dictionary import attribute_dict
from imputation_dictionary import attribute_imputer_dict
import sys
import string
import voluntary_prediction_helpers as vph


def select_features(train_data,n_iter=5):
    ##Lets make sure there are no duplicate chc_ids
    if len(train_data) != len(train_data['chc_id'].unique()):
        vph.log('*'*10+"\nDuplicates are present in your data")
        sys.exit(2)
    else:
        vph.log('*'*10+"\nNo duplicates found.")

    train_data['chc_id'] = train_data['chc_id'].astype(str)

    #train_data['chc_id'] = train_data['chc_id'].astype(object)
    #Let's make index of the dataframe as corp_house_cust
    train_data.index = train_data['chc_id']

    #Sorting the dataframe by chc_id
    train_data = train_data.sort_index()

    #Let's remove redundant columns
    del train_data['corp']
    del train_data['house']
    del train_data['chc_id']
    ########################################################################################################
    #Attribute quirks
    ########################################################################################################
    #Unfortunately, some NUM columns are being interpreted as string, let's cast them back as numeric
    train_data['cust'] = pd.to_numeric(train_data['cust'], errors='coerce')

    train_data['roll_off_lift_m1'] = pd.to_numeric(train_data['roll_off_lift_m1'], errors='coerce')
    train_data['roll_off_lift_m2'] = pd.to_numeric(train_data['roll_off_lift_m2'], errors='coerce')
    train_data['roll_off_lift_m3'] = pd.to_numeric(train_data['roll_off_lift_m3'], errors='coerce')
    train_data['roll_off_lift_m4'] = pd.to_numeric(train_data['roll_off_lift_m4'], errors='coerce')

    train_data['vidpromo_mthsleft_m1'] = pd.to_numeric(train_data['vidpromo_mthsleft_m1'], errors='coerce')
    train_data['vidpromo_mthsleft_m2'] = pd.to_numeric(train_data['vidpromo_mthsleft_m2'], errors='coerce')
    train_data['vidpromo_mthsleft_m3'] = pd.to_numeric(train_data['vidpromo_mthsleft_m3'], errors='coerce')
    train_data['vidpromo_mthsleft_m4'] = pd.to_numeric(train_data['vidpromo_mthsleft_m4'], errors='coerce')


    ####################Removing all the NPD columns that are not present in the View in DB#################
    dict_keys = pd.Series(list(attribute_dict.keys()))

    df_cols = train_data.columns.to_series()

    res = list(dict_keys[dict_keys.isin(df_cols)==False])


    for key in res:
        attribute_dict.pop(key,None)
        attribute_imputer_dict.pop(key,None)
    #
    #count = 0
    #for val in list(attribute_dict.values()):
    #    if val=='DNU':
    #        count = count +1

    ########################################################################################################
    #Cleanup and Imputation
    ########################################################################################################
    def process_data(df, cols_to_retain):
        """
        - Takes as input - 1) dataframe and 2) if the columns to process are 'NUM' only or 'CAT' only or 'ALL'
        - Imputes whitespaces with NaN
        - Imputes missing values
        - Drops any records with missing values post imputation
        - Dummies the columns for typical scikit models to ingest, if there are 'CAT' columns
        - Return the DF after these steps, and also returns: 
        - Create a DF with names: Attribute, Attribute-Category without prefix_sep, and Attribute-Category with prefix_sep
        """
        df_data = df
        
        vph.log('*'*10 + "\nData processing and cleanup starts now.", flush=True)
        vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)

        #Put attribute column-names in a list
        list_col_names = list(df_data.iloc[0:,])
        
        #Seperate categorical and numerical attributes and also the target variable
        cat_attributes     = [x for x in list_col_names if (attribute_dict[x] == 'CAT')]
        num_attributes     = [x for x in list_col_names if (attribute_dict[x] == 'NUM')]
        target_attribute   = ['status']#[x for x in list_col_names if (attribute_dict[x] == 'TARGET')]
        
        #Let's keep the class label aside and concat it later NUM and dummied-CAT features later
        class_label_df = df_data[[target_attribute[0]]]
        
        #Retain only NUM columns
        if cols_to_retain == 'NUM':
            cols = num_attributes
            df_data = df_data[cols]
            vph.log('*'*10 + "\nThis Feature Selection run is only for NUM attributes - a total of " + str(len(cols)), flush=True)
            vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
        
        #Retain only CAT columns    
        if cols_to_retain == 'CAT':
            cols = cat_attributes
            df_data = df_data[cols]
            vph.log('*'*10 + "\nThis Feature Selection run is only for CAT attributes - a total of " + str(len(cols)), flush=True)
            vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
            
        #Retain ALL columns    
        if cols_to_retain == 'ALL':
            cols = cat_attributes + num_attributes
            df_data = df_data[cols]
            vph.log('*'*10 + "\nThis Feature Selection run is for ALL attributes - a total of " + str(len(cols)), flush=True)
            vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
        
        #Let's impute any whitespaces with NaNs:
        df_data = df_data.replace(r'^\s*$', np.nan, regex=True) #careful to not replace whitespaces in between characters
        
        #Let's see how many NaNs there are:
        vph.log('*'*10 + '\nNaNs in attributes:\n' + '*'*10, flush=True)
        vph.log(df_data.isnull().sum(), flush=True)
        
        #Impute attributes:
        for each in cols:
            df_data[each].fillna(attribute_imputer_dict[each], inplace=True)
        
        #Let's drop records with any NaNs
        df_data = df_data.dropna(how='any')
        
        #Let's see again how many NaNs there are (should be zero):
        vph.log('*'*10 + '\nNaNs in attributes after imputation and dropping NaNs (after dropna="any"):\n' + '*'*10, flush=True)
        vph.log(df_data.isnull().sum(), flush=True)
        vph.log('*'*10 + '\nNumber of records after NaNs removal: ' + str(len(df_data)) +'\n'+ '*'*10, flush=True)
        
        def feature_names_mapping (df_data_model_ip, prefix_sep):
            """
            #A function that will save,
                1) original attribute name,
                2) attribute-category without prefix_sep for dummying, but will use '_' as seperator between attribute & category 
                3) attribute-category with    prefix_sep for dummying
            and return these three as three columns of a DataFrame
            """
            from collections import OrderedDict
            temp_dict = OrderedDict() #will use to store 1) and 2) 
            original_cols = [] #will use to store 3)
            for each_col in df_data_model_ip.columns: #for each column in the already-dummied DF (i.e., if CATs exisit in the DF)
                if prefix_sep in str(each_col): #if the prefix_sep is in the column name
                    k,v = each_col.split(prefix_sep) #split on prefix_sep and assign them to a Key and Value
                    if k in temp_dict.keys(): #if the key already exists
                        temp_dict[k].append(str(k)+'_'+str(v)) #append k_v in a string format to the values list of that key
                    else: #if the key does not exist
                        temp_dict[k] = [] #create the key and assign an empty list as the value
                        temp_dict[k].append(str(k)+'_'+str(v)) #and then append k_v to the values list
                else: #if prefix_sep is not in the column name (in the case of NUM attributes)
                    temp_dict[each_col] = [each_col] #make the key and value the same
                original_cols.append(each_col)
            
            #Make a DF Attribute and Attribute-Category, without prefix_sep)    
            col_names_mapping_df = pd.DataFrame([(k, each_vL) for (k, vL) in temp_dict.items() for each_vL in vL], columns=['Attribute', 'Attribute_Category'])
            #Add another column with Attribute-Category, with prefix_sep
            col_names_mapping_df['Attribute_Category_with_Prefix_Sep'] = original_cols
            
            return col_names_mapping_df 
            
        
        #Retain only NUM columns
        if cols_to_retain == 'NUM':
            df_data_model_ip = pd.concat([class_label_df, df_data], axis=1) #concat class label and NUM columns
        
        #Retain only CAT columns:
        if cols_to_retain == 'CAT':
            df_data_cat_to_be_dummied = df_data.astype(str)  #Make sure all the categorical features are explicitly casted as a string/object
            df_data_cat_dummied = pd.get_dummies(df_data_cat_to_be_dummied, prefix=cols, prefix_sep='vg58rj')  #Let's dummy categorical attributes
            del df_data_cat_to_be_dummied
            df_data_model_ip = pd.concat([class_label_df, df_data_cat_dummied], axis=1) #concat class label and CAT columns, now dummied
            cat_nuniques = df_data.apply(pd.Series.nunique)
            cat_nuniques.to_csv('Categorical_Nuniques.csv',header=['Unique_values'],index=True, index_label = 'Attribute')
            vph.log('*'*10 + "\nDone writing to csv no. of unique categories of CAT attributes.", flush=True)
            vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
                
        #Retain ALL columns:
        if cols_to_retain == 'ALL':
            num_attributes_model_ip = [each for each in num_attributes if each in df_data.columns.values] #Final list of numeric variables ingested to model
            df_data[num_attributes_model_ip] = df_data[num_attributes_model_ip].astype(float)
            df_data_cat_to_be_dummied = df_data[cat_attributes] #Let's dummy data for CAT attributes first and then concat with NUM attributes. First, let's splice categorical data into it's own DF.
            df_data_cat_to_be_dummied = df_data_cat_to_be_dummied.astype(str) #Make sure all th categorical features are explicitly casted as a string/object
            df_data_cat_dummied = pd.get_dummies(df_data_cat_to_be_dummied, prefix=cat_attributes, prefix_sep='vg58rj') #Let's dummy categorical attributes
            del df_data_cat_to_be_dummied
            df_data_model_ip = pd.concat([class_label_df, df_data[num_attributes_model_ip], df_data_cat_dummied], axis=1) #Let's bring together 1) class variable, 2) numerical data, and 3) dummied categorical data, to create the final dataset
            
        #Let's drop columns that have zero variance. Helps avoid divide by zero error during standardization and shrinks featureset for good.
        #cols_with_variance_zero = [each_col for each_col in df_data_model_ip.columns.values if df_data_model_ip[each_col].var() == 0]
        cols_with_variance_zero = []
        for each_col in df_data_model_ip.columns.values:
            #vph.log(each_col)
            if df_data_model_ip[each_col].var() == 0:
                cols_with_variance_zero.append(each_col)
        vph.log('*'*10)
        vph.log('These columns have zero variance, so dropping them: \n', flush=True)
        for each in cols_with_variance_zero: vph.log('-',each)
        remaining_cols = [x for x in df_data_model_ip.columns.values if x not in cols_with_variance_zero]
        df_data_model_ip = df_data_model_ip[remaining_cols]
        
        #Make a function call to  feature_names_mapping(DF,prefix_sep)
        col_names_mapping_df = feature_names_mapping(df_data_model_ip, 'vg58rj')
        
        vph.log('*'*10 + "\nData processing for %s attributes has finished. "%cols_to_retain, flush=True)
        vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
        
        return df_data_model_ip, col_names_mapping_df

    df_data_model_ip, col_names_mapping_df = process_data(train_data, cols_to_retain='ALL')
    del train_data

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
        temp = [x if x != ' ' else 'SPACE' for x in temp1] #replace the character ' ' with the string SPACE
        temp = ''.join(x for x in temp) #join all the characters into one word
        clean_names.append(temp)
        original_names.append(each)

    df_data_model_ip.columns = clean_names
    #Storing column names pre and post this transformation in a DF
    xgb_names_df = pd.DataFrame([(a,b) for a,b in zip(original_names,clean_names)], columns=['Attribute_Category_with_Prefix_Sep','Attribute_Category_XGBoost'])
    vph.log('*'*10 + "\nDone converting feature names to XGBoost's liking.", flush=True)
    vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)
    '''
    #QC - to compare original and transformed names
    comp = [] #a list that holds clean name and original name tuple
    comp_dict = {} #a dictionary that will hold key as clean name and value as original name

    for each in range(len(clean_names)):
        z = str(clean_names[each])+'\t'+str(original_names[each])
        comp_dict[clean_names[each]] = str(original_names[each])
        #vph.log(z)
        comp.append(z)

    with open('compare.txt', 'w') as pen:
        for each in comp:
            pen.write("%s\n" %each)
    '''
    ########################################################################################################
    #Modeling - Let's do a Nested CV
    ########################################################################################################
    y = df_data_model_ip.iloc[:,0] #all rows, first column
    X = df_data_model_ip.iloc[:,1:] #all rows, second column through:last column
    X = X.reset_index(drop = False)
    ########################################################################################################
    #Modeling - Fit Model
    ########################################################################################################
    ###10 fold CV for FS
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_iter,shuffle=True,random_state = 25)
    skf.get_n_splits(X, y)

    feature_ranking_Kfold_df = pd.DataFrame(columns = ['Attribute_Category_XGBoost','Ranking_Metric_Gain'])

    fit_count = 1
    for train_index, test_index in skf.split(X, y):
        #vph.log(train_index)
        #vph.log(test_index)
        vph.log('*'*10+'\nDoing model fit for '+str(fit_count)+' split'+'\n'+'*'*10,flush = True)
        fit_count = fit_count+1
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        smpl_wght = y_train[y_train == 0]/y_train[y_train == 1]
        vph.log('*'*10+'\nSum of y_train:'+str(np.sum(y_train)))
        vph.log('*'*10+'\n Sum of y_test:' + str(np.sum(y_test)))
        X_train.set_index('chc_id',inplace = True)
        model = XGBClassifier(n_jobs = -1, objective = 'binary:logistic', eval_metric = ['auc'], sample_weight = smpl_wght, verbose=True,random_state= 25, seed = 25,colsample_bylevel=1,colsample_bytree=1,subsample=1,n_estimators=200)
        model.fit(X_train,y_train)
        feature_ranking = model.get_booster().get_score(importance_type = 'gain')
        feature_ranking = sorted(feature_ranking.items(), key = lambda kv: kv[1], reverse=True)
        feature_ranking_df = pd.DataFrame(feature_ranking, columns = ['Attribute_Category_XGBoost','Ranking_Metric_Gain'])
        feature_ranking_Kfold_df = feature_ranking_Kfold_df.append(feature_ranking_df,ignore_index = True)
        feature_ranking_Kfold_df = feature_ranking_Kfold_df.groupby('Attribute_Category_XGBoost').agg({'Ranking_Metric_Gain':np.mean}).reset_index()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25,stratify = y) #split into train and test
    #df_data_model_ip_dmatrix = DMatrix(data=X_train,label=y_train) #convert into DMatrix format for move XGB love
    #smpl_wght = y_train[y_train == 0]/y_train[y_train == 1]
    #vph.log('*'*10+'\nSum of y_train:'+str(np.sum(y_train)))
    #vph.log('*'*10+'\n Sum of y_test:' + str(np.sum(y_test)))
    #model = XGBClassifier(n_jobs = -1, objective = 'binary:logistic', eval_metric = ['auc'], sample_weight = smpl_wght, verbose=True,random_state= 25, seed = 25,colsample_bylevel=1,colsample_bytree=1,subsample=1)
    #model.fit(X_train, y_train)
    

    vph.log('*'*10 + "\nDone fitting model on TRAIN data.", flush=True)
    vph.log(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'), flush=True)

    ########################################################################################################
    #Feature Importance - Let's get this in all possible ways it can be is defined
    ########################################################################################################
    '''
    model.get_booster().get_score(importance_type = 'weight')      : the number of times a feature is used to split the data across all trees.
    model.get_booster().get_score(importance_type = 'gain')        : the average gain across all splits the feature is used in.
    model.get_booster().get_score(importance_type = 'cover')       : the average coverage across all splits the feature is used in.
    model.get_booster().get_score(importance_type = 'total_gain')  : the total gain across all splits the feature is used in.
    model.get_booster().get_score(importance_type = 'total_cover') : the total coverage across all splits the feature is used in.
    model.feature_importances_                                     : same as model.get_booster().get_score(importance_type='weight'), but each value is divided by the sum of all values.

    list_of_dicts = []
    for each_importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
        temp = model.get_booster().get_score(importance_type = each_importance_type)
        list_of_dicts.append(temp)

    yowza = pd.DataFrame(list_of_dicts)
        
        #vph.log(len(model.get_booster().get_score(importance_type = each_importance_type)))
        #feature_ranking_df[each_importance_type] = model.get_booster().get_score(importance_type = each_importance_type)
        
    #feature_ranking_df['weight_normalized'] = model.feature_importances_.ravel()
        
    feature_importances_list = [each for each in zip(X_train.columns.values,model.feature_importances_.ravel())]
    feature_importances_list = list(sorted(feature_importances_list, key = lambda x: x[1], reverse=True))
    for each in feature_importances_list: vph.log(each)
    #Plot Feature Importance
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()
    '''

    #feature_ranking = model.get_booster().get_score(importance_type = 'gain')
    #feature_ranking = sorted(feature_ranking.items(), key = lambda kv: kv[1], reverse=True)
    #feature_ranking_df = pd.DataFrame(feature_ranking, columns = ['Attribute_Category_XGBoost','Ranking_Metric_Gain'])

    #Merge DataFrames to get column names and corresponding Feature Importance Metric
    z_temp = col_names_mapping_df.merge(xgb_names_df, on='Attribute_Category_with_Prefix_Sep', how='left')
    feature_ranking_final_df = z_temp.merge(feature_ranking_Kfold_df, on='Attribute_Category_XGBoost', how='left')
    feature_ranking_final_df['Ranking_Metric_Gain'] = feature_ranking_final_df['Ranking_Metric_Gain'].fillna(0.0)

    #Sort by the Ranking Metric Gain descending
    feature_ranking_final_df = feature_ranking_final_df.sort_values(by=['Ranking_Metric_Gain'],ascending=False)

    return feature_ranking_final_df