"""
Author: Taesun Kim
Date:   4/8/2019

Title: Create Custom Transformers to Primarily Conduct Pre-Processing

Purpose:
    1. Make Custom Transformers to Conduct Pre-Processing 
       Which Are Compatible with General Model Building Process and sklearn API

Custom Transformers That Comprise:
    1. 'Use_DefaultDataType'
        - Convert dtypes into default dtypes.
    2. 'Use_DefaultImputer'
        - Use default imputation values.
    3. 'Remove_MissingFeatures'
        - Remove features with high missing percentage (e.g. 99%)
    4. 'Remove_ConstantFeatures'
        - Remove features with a single unique value.
    5. 'Remove_CorrelatedFeatures'
        - Remove features highly correlated with other features (e.g. abs(Corr) >= 0.95).
    6. 'Remove_DuplicateFeatures'
        - Remove features with duplicate columns.
    7. 'Summarize_Features'
        - Append full summary statistics to Custom Transformer attributes.

Update History:
    Update on 4/8/2019
    1. 'features_irrelevant' list
        - ‘account_pend_disco_frq’ is newly included in 'features_irrelevant' list.
        - See the email, 'Update in Dictionary' on 4/3/2019 from Venu Guntupalli.
    2. 'Remove_DuplicateFeatures'
        - A new parameter, `feature_type` is added to save computation time.
            - default: feature_type=None
            - Unless feature_type='NUM', only CAT features be examined to reduce running time.
        - Printing paired duplicate features is suppressed.

References:    
    - sklearn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
    - Feature Selector: https://github.com/WillKoehrsen/feature-selector
    - In-house Code: 'data_procesing_for_modeling.py' 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin



def Summarize_Features(df_X, df_Summary, ls_features, y=None):
    '''
    Purpose
    ---------------
    A custom utility function
    - (1) to append summary statistics and 
    - (2) to append churn rate if y is provided
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df) with all features
    df_Summary: a df with simple summary statistics
    ls_features: a list of selected features
    y: a pandas series that represent a churn status
    - default: y=None

    Returns
    ---------------
    df_Summary_full: a df with full summary statistics
    '''

    df_desc         = df_X[ls_features].describe().T
    df_Summary_full = df_Summary. \
                      merge(df_desc, left_on='feature', right_index=True, how='inner').\
                      reset_index(drop=True)

    if not (y is None):    # When y is given
        temp_df    = pd.concat([y, df_X[ls_features]], axis=1)
        churn_rate = {}

        for fe in ls_features:
            y_mean = np.round(temp_df[(temp_df[fe].notnull())].iloc[:, 0].mean(), 4)
            churn_rate[fe] = y_mean

        churn_df   = pd.DataFrame.from_dict(churn_rate, orient='index')
        churn_df.columns = ['churn_rate']

        df_Summary_full = df_Summary. \
                          merge(churn_df, left_on='feature', right_index=True, how='inner').\
                          merge(df_desc, left_on='feature', right_index=True, how='inner').\
                          reset_index(drop=True)                

    return df_Summary_full



class Remove_MissingFeatures(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to identify missing percentages of features and 
    - (2) to remove features with missing % >= threshold missing %
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    missing_threshold: missing percentage ~ [0, 1]
       - default: missing_threshold=0.99    
    
    Returns
    ---------------
    dataframe
        - X: a df that consists of selected features after dropping via '.tranform(X)'
        - summary_dropped_: a df that includes dropped features with simple summary statistics
        - summary_dropped_NUM_: a df that includes dropped NUM features with detailed summary statistics
        - summary_dropped_CAT_: a df that includes dropped CAT features with detailed summary statistics
    list
        - features_dropped_: features that drop due to missing
        - features_kept_: features that are kept
    Plot: a histogrm that shows the frequency of missing pct bins
        
    References
    ---------------        
    Feature Selector: https://github.com/WillKoehrsen/feature-selector
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, missing_threshold=0.99):
        self.missing_threshold = missing_threshold
        if (self.missing_threshold<0) | (self.missing_threshold>1):
            raise ValueError('Missing threshold should be in the range of 0 to 1.')

        print('*'*50 + '\nPre-Processing: Remove_MissingFeatures\n' + '*'*50 + \
              '\n- It will remove features with a high missing pct.\n')


    def fit(self, X, y=None):
        # 0. Remove Irrelevant Features
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM      = X[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT      = X[features_relevant].select_dtypes(include=[object]).columns.tolist()
        
        # 1. Compute Missing % of All Features
        self.summary_full_ = pd.DataFrame(X[features_relevant].isnull().sum() / X.shape[0]).reset_index().\
                             rename(columns = {'index': 'feature', 0: 'missing_pct'}).\
                             sort_values('missing_pct', ascending = False).\
                             reset_index(drop=True)
        
        # 2. Compute Missing % of Dropped Features
        self.summary_dropped_ = self.summary_full_[self.summary_full_['missing_pct'] > self.missing_threshold].\
                                reset_index(drop=True)
        self.summary_kept_    = self.summary_full_[self.summary_full_['missing_pct'] <= self.missing_threshold].\
                                reset_index(drop=True)
        
        # 3. Make a List of Dropped Features
        self.features_kept_    = self.summary_kept_.sort_values('feature')['feature'].tolist()        
        self.features_dropped_ = self.summary_dropped_.sort_values('feature')['feature'].tolist()
        features_dropped_NUM   = [fe for fe in self.features_dropped_ if fe in features_NUM]
        features_dropped_CAT   = [fe for fe in self.features_dropped_ if fe in features_CAT]        

        # 4. Make a df of dropped features with summary statistics
        # Note: A custom utility function, 'Summarize_Features()' is used.
        
        if (len(self.features_dropped_)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.copy()
            self.summary_dropped_CAT_ = self.summary_dropped_.copy()
        elif (len(self.features_dropped_)>0) & (len(features_dropped_NUM)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)
        elif (len(self.features_dropped_)>0) & (len(features_dropped_CAT)==0):
            self.summary_dropped_CAT_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
        else:
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)
        
        print('{} features with greater than {}% missing values'.format(len(self.features_dropped_), self.missing_threshold * 100))
        
        return self


    def transform(self, X, y=None):
        return X[self.features_kept_]

    
    def plot(self):
        plt.style.use('seaborn-white')
        plt.figure(figsize = (7, 5))
        plt.hist(self.summary_full_['missing_pct'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'blue', linewidth = 1.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Pct', size = 14); plt.ylabel('Count of Features', size = 14)
        plt.title("Histogram: Missing Values", size = 16)



class Remove_ConstantFeatures(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to identify features with a single unique value and 
    - (2) to remove those constant features
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    unique_threshold: number of unique values >= 1 in integer
       - default: unique_threshold=1
    missing_threshold: missing percentage ~ [0, 1]
       - default: missing_threshold=0.00     
           - missing_threshold=0.00 --> Focus on features with full non-missing values.
           - missing_threshold=1.00 --> Focus on all features regardless of missing pct.
    
    Returns
    ---------------
    datafrme
        - X: a df that consists of selected features after dropping via '.tranform(X)'        
        - summary_dropped_: a df that includes dropped features with simple summary statistics
        - summary_dropped_NUM_: a df that includes dropped NUM features with full summary statistics
        - summary_dropped_CAT_: a df that includes dropped CAT features with full summary statistics
    list
        - features_dropped_: features that drop due to non-unique values
        - features_kept_: features that are kept
    
    References
    ---------------        
    Feature Selector: https://github.com/WillKoehrsen/feature-selector
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, unique_threshold=1, missing_threshold=0.00):
        self.unique_threshold  = unique_threshold
        self.missing_threshold = missing_threshold
        
        print('*'*50 + '\nPre-Processing: Remove_ConstantFeatures\n' + '*'*50 + \
              '\n- It will remove features with {} unique value(s).\n'.format(self.unique_threshold))
        

    def fit(self, X, y=None):
        # 0. Remove Irrelevant Features
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM = X[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT = X[features_relevant].select_dtypes(include=[object]).columns.tolist()

        # 1. Compute Unique Values of All Features   
        # Unique values should be considered in conjuection with missing pct
        missing_full_      = pd.DataFrame(X[features_relevant].isnull().sum() / X.shape[0]).reset_index().\
                             rename(columns = {'index': 'feature', 0: 'missing_pct'})
                             
        self.summary_full_ = pd.DataFrame(X[features_relevant].nunique()).reset_index().\
                             rename(columns = {'index': 'feature', 0: 'nunique'}).\
                             merge(missing_full_, left_on='feature', right_on='feature', how='inner').\
                             sort_values('nunique', ascending = False).\
                             reset_index(drop=True)
        
        # 2. Identify Features with a single unique value
        flag_dropped          = (self.summary_full_['nunique'] <= self.unique_threshold) & \
                                (self.summary_full_['missing_pct'] <= self.missing_threshold)
                                
        self.summary_dropped_ = self.summary_full_[flag_dropped].reset_index(drop=True)
        self.summary_kept_    = self.summary_full_[~flag_dropped].reset_index(drop=True)
        
        # 3. Make a List of Dropped Features
        self.features_kept_    = self.summary_kept_.sort_values('feature')['feature'].tolist()        
        self.features_dropped_ = self.summary_dropped_.sort_values('feature')['feature'].tolist()
        features_dropped_NUM   = [fe for fe in self.features_dropped_ if fe in features_NUM]
        features_dropped_CAT   = [fe for fe in self.features_dropped_ if fe in features_CAT]        

        # 4. Make a df of dropped features with summary statistics
        # Note: A custom utility function, 'Summarize_Features()' is used.
        
        if (len(self.features_dropped_)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.copy()
            self.summary_dropped_CAT_ = self.summary_dropped_.copy()
        elif (len(self.features_dropped_)>0) & (len(features_dropped_NUM)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)
        elif (len(self.features_dropped_)>0) & (len(features_dropped_CAT)==0):
            self.summary_dropped_CAT_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
        else:
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)

        print('{} features with {} or fewer unique value(s)'.format(len(self.features_dropped_), self.unique_threshold))
        
        return self


    def transform(self, X, y=None):
        return X[self.features_kept_]

    
    def plot(self):
        plt.style.use('seaborn-white')
        plt.figure(figsize = (7, 5))
        plt.hist(self.summary_full_['nunique'], edgecolor = 'k', color = 'blue', linewidth = 1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Unique Values', size = 14); plt.ylabel('Count of Features', size = 14); 
        plt.title("Histogram: Unique Values", size = 16);



class Remove_CorrelatedFeatures(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to compute pairwise correlation between features and 
    - (2) to remove features with abs(correlation) >= threshold correlation
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    correlation_threshold: abs(correlation) ~ [0, 1]
       - default: correlation_threshold=0.90    
    
    Returns
    ---------------
    datafrme
        - X: a df that consists of selected features after dropping via '.tranform(X)'        
        - summary_dropped_: a df with pairwise correlation between dropped/correlated features
    list
        - features_dropped_: features that drop due to multicollinearity
        - features_kept_: features that are kept
        
    References
    ---------------        
    Feature Selector: https://github.com/WillKoehrsen/feature-selector
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, correlation_threshold=0.90):
        self.correlation_threshold = correlation_threshold
        if (self.correlation_threshold<0) | (self.correlation_threshold>1):
            raise ValueError('Correlation threshold should be in the range of 0 to 1.')
            
        print('*'*50 + '\nPre-Processing: Remove_CorrelatedFeatures\n' + '*'*50 + \
              '\n- It will work on Numerical Features Only, doing nothing on Categorical Features.' + \
              '\n- It may take 10+ minutes. Be patient!\n')


    def fit(self, X, y=None):
        # 0. Select Numerical Features Only (after Removing Irrelevant Features)
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM      = X[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT      = X[features_relevant].select_dtypes(include=[object]).columns.tolist()

        # Sort Features in abs(correlation) with Target. Default: Features in Alphabetic Order        
        if not (y is None):       
            corr_all    = pd.concat([y, X[features_NUM]], axis=1).corr()
            corr_target = corr_all.iloc[1:, 0].to_frame()
            corr_target.columns = ['corr_target']
            features_NUM  = corr_all.iloc[1:, 0].abs().sort_values(ascending=False).index.tolist()
        
        # 1. Compute Pairwise Correlation between Features
        corr_full  = X[features_NUM].corr()
        corr_upper = corr_full.where(np.triu(np.ones(corr_full.shape), k = 1).astype(np.bool))
        
        # 2. Identify Features with abs(correlation) > correlation threshold
        features_dropped_      = [fe for fe in corr_upper.columns \
                                 if any(corr_upper[fe].abs() > self.correlation_threshold)]
        features_dropped_.sort()                         
        self.features_dropped_ = features_dropped_
        # Combine not dropping features with CAT features
        self.features_kept_    = list(set(features_NUM) - set(self.features_dropped_)) + features_CAT
        
        # 3. Create Pairwise Correlation Dataframe
        summary_dropped_ = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'pairwise_corr'])
        
        for fe in self.features_dropped_:
            corr_features = list(corr_upper.index[corr_upper[fe].abs() > self.correlation_threshold])
            corr_values   = list(corr_upper[fe][corr_upper[fe].abs() > self.correlation_threshold])
            drop_features = [fe for _ in range(len(corr_features))]    
            temp_df       = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                                    'corr_feature': corr_features,
                                                    'pairwise_corr': corr_values})
            summary_dropped_ = summary_dropped_.append(temp_df, ignore_index = True)
         
        self.summary_dropped_ = summary_dropped_
        
        if not (y is None):       # Append Correlation with Target
            summary_dropped_ = summary_dropped_.\
                               merge(corr_target, left_on='drop_feature', right_index=True, how='left').\
                               merge(corr_target, left_on='corr_feature', right_index=True, how='left').\
                               sort_values(['drop_feature', 'corr_feature']).\
                               reset_index(drop=True)
            summary_dropped_.columns = ['drop_feature', 'corr_feature', 'pairwise_corr', 'corr_DF_target', 'corr_CF_target']
            self.summary_dropped_ = summary_dropped_
        
        print('{} features with abs(correlation ) > {} with other features'.format(len(self.features_dropped_), self.correlation_threshold))
        
        return self


    def transform(self, X, y=None):
        return X[self.features_kept_]



class Remove_DuplicateFeatures(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to identify features with duplicate columns and 
    - (2) to remove features with duplicate columns
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    feature_type = a string to indicate data types of features
       - default: feature_type=None
       - Unless feature_type='NUM', only CAT features be examined to save computation time.

    Returns
    ---------------
    datafrme
        - X: a df that consists of selected features after dropping via '.tranform(X)'        
        - summary_dropped_: a df that includes dropped/duplicate features
        - summary_dropped_NUM_: a df that includes dropped/duplicate NUM features with full summary statistics
        - summary_dropped_CAT_: a df that includes dropped/duplicate CAT features with full summary statistics
    list
        - features_dropped_: features that drop due to duplicate columns
        - features_kept_: features that are kept
        
    References
    ---------------        

    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, feature_type = None):
        self.feature_type = feature_type
        print('*'*50 + '\nPre-Processing: Remove_DuplicateFeatures\n' + '*'*50 + \
              '\n- It may take 10+ minutes. Be patient!\n')


    def fit(self, X, y=None):
        # 0. Remove Irrelevant Features
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM      = X[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT      = X[features_relevant].select_dtypes(include=[object]).columns.tolist()
        
        # 1. Identify Features with Duplicate Columns
        X_CAT_tmp = X[features_CAT].copy()        
        X_NUM_tmp = X[features_NUM].copy()

        features_dropped_  = []
        summary_dropped_    = pd.DataFrame(columns = ['drop_feature', 'duplicate_feature'])
        
        for i in range(0, len(features_CAT)):
            fe_1 = X_CAT_tmp.columns[i]
            for fe_2 in X_CAT_tmp.columns[i + 1:]:
                if X_CAT_tmp[fe_1].equals(X_CAT_tmp[fe_2]):
                    # print("'" + fe_1 + "' is the same as '" + fe_2 + "'" + '\n')
                    features_dropped_.append(fe_2)
                    temp_df = pd.DataFrame.from_dict({'drop_feature': [fe_2], 'duplicate_feature': [fe_1]})
                    summary_dropped_ = summary_dropped_.append(temp_df, ignore_index = True)

        if (self.feature_type=='NUM'):
            for i in range(0, len(features_NUM)):
                fe_1 = X_NUM_tmp.columns[i]
                for fe_2 in X_NUM_tmp.columns[i + 1:]:
                    if X_NUM_tmp[fe_1].equals(X_NUM_tmp[fe_2]):
                        # print("'" + fe_1 + "' is the same as '" + fe_2 + "'" + '\n')
                        features_dropped_.append(fe_2)
                        temp_df = pd.DataFrame.from_dict({'drop_feature': [fe_2], 'duplicate_feature': [fe_1]})
                        summary_dropped_ = summary_dropped_.append(temp_df, ignore_index = True)

        features_dropped_.sort()
        self.features_dropped_ = list(set(features_dropped_))
        self.features_kept_    = list(set(features_relevant) - set(self.features_dropped_))
        self.summary_dropped_  = summary_dropped_

        # 2. Create data frames that summarize drop/duplicate features
        if (len(self.features_dropped_)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.copy()
            self.summary_dropped_CAT_ = self.summary_dropped_.copy()
        else:
            desc_CAT                  = X_CAT_tmp.describe().T     # Categorical Features
            summary_dropped_CAT_      = summary_dropped_.\
                                        merge(desc_CAT, left_on='drop_feature', right_index=True, how='inner').\
                                        merge(desc_CAT, left_on='duplicate_feature', right_index=True,
                                              how='inner', suffixes=('_drop', '_duplicate'))
            self.summary_dropped_CAT_ = summary_dropped_CAT_

            desc_NUM                  = X_NUM_tmp.describe().T     # Numerical Features
            summary_dropped_NUM_      = summary_dropped_.\
                                        merge(desc_NUM, left_on='drop_feature', right_index=True, how='inner').\
                                        merge(desc_NUM, left_on='duplicate_feature', right_index=True,
                                              how='inner', suffixes=('_drop', '_duplicate'))
            self.summary_dropped_NUM_ = summary_dropped_NUM_

        print('{} features with duplicate columns'.format(len(self.features_dropped_)))    

        return self


    def transform(self, X, y=None):
        return X[self.features_kept_]



class Use_DefaultDataType(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to identify features having data types inconsistent with default data types and 
    - (2) to convert the data types into the default data types if inconsistent
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    default_dtypes: a data dictionary that provides default data types of features
    
    Returns
    ---------------
    datafrme
        - X: a df that consists of all features with default data types via '.tranform(X)'
        - summary_inconsistent_dtypes_: a df that includes features with inconsistent data types.
        - summary_inconsistent_NUM_: a df that includes inconsistent NUM features with full summary statistics
        - summary_inconsistent_CAT_: a df that includes inconsistent CAT features with full summary statistics
        
    list
        - features_inconsistent_dtypes_: all features that have inconsistent data types
        - features_inconsistent_NUM_: NUM features that have inconsistent data types
        - features_inconsistent_CAT_: CAT features that have inconsistent data types
        
    References
    ---------------        
    'data_procesing_for_modeling.py'    
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, default_dtypes):
        self.default_dtypes = default_dtypes
        print('*'*50 + '\nPre-Processing: Use_DefaultDataType\n' + '*'*50 + \
              '\n- It will convert data types into default ones.\n')


    def fit(self, X, y=None):
        # Note:
        # 'fit()' is not needed for data type conversion. 
        # 'fit()', however, is used to generate a list/df of inconsistent features.
        
        # 0. Remove Irrelevant Features
        X_tmp             = X.copy()
        features_relevant = list(set(X_tmp.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM = X_tmp[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT = X_tmp[features_relevant].select_dtypes(include=[object]).columns.tolist()
        
        # 1. Group features by default data type
        default_NUM = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'NUM')]        
        default_CAT = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'CAT')]
        
        # 2. Identify features with inconsistent data types
        features_inconsistent_NUM_ = [fe for fe in features_CAT if fe in default_NUM]
        features_inconsistent_CAT_ = [fe for fe in features_NUM if fe in default_CAT]

        # 3. Create a list/df of inconsistent features
        # list
        self.features_inconsistent_NUM_    = features_inconsistent_NUM_
        self.features_inconsistent_CAT_    = features_inconsistent_CAT_
        features_inconsistent_dtypes_      = features_inconsistent_NUM_ + features_inconsistent_CAT_
        self.features_inconsistent_dtypes_ = features_inconsistent_dtypes_
        
        # df
        if (len(features_inconsistent_NUM_)==0):
            temp_df1                       = pd.DataFrame(columns = ['feature', 'default_dtype', 'python_dtype'])
            self.summary_inconsistent_NUM_ = temp_df1
        else:
            temp_df1 = pd.DataFrame.\
                       from_dict({'feature': features_inconsistent_NUM_,
                                  'default_dtype': 'NUM',
                                  'python_dtype': X_tmp[features_inconsistent_NUM_].dtypes})
            desc_NUM = X_tmp[features_inconsistent_NUM_].describe().T
            self.summary_inconsistent_NUM_ = temp_df1.\
                                             merge(desc_NUM, left_on='feature', right_index=True, how='inner').\
                                             reset_index(drop=True)      

        if (len(features_inconsistent_CAT_)==0):
            temp_df2                       = pd.DataFrame(columns = ['feature', 'default_dtype', 'python_dtype'])
            self.summary_inconsistent_CAT_ = temp_df2
        else:
            temp_df2 = pd.DataFrame.\
                       from_dict({'feature': features_inconsistent_CAT_,
                                  'default_dtype': 'CAT',
                                  'python_dtype': X_tmp[features_inconsistent_CAT_].dtypes})
            desc_CAT = X_tmp[features_inconsistent_CAT_].describe().T
            self.summary_inconsistent_CAT_ = temp_df1.\
                                             merge(desc_CAT, left_on='feature', right_index=True, how='inner').\
                                             reset_index(drop=True)      
                                             
        self.summary_inconsistent_dtypes_ = pd.concat([temp_df1, temp_df2], axis=0).reset_index(drop=True)          

        return self


    def transform(self, X, y=None):
        # Note:
        # Data type conversion be based on a data passed to 'transform()', not 'fit()'.
        
        # 0. Remove Irrelevant Features in Data Type Conversion
        X_tmp             = X.copy().replace(r'^\s*$', np.nan, regex=True)  # Impute Whitespaces with Nans
        features_relevant = list(set(X_tmp.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM      = X_tmp[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT      = X_tmp[features_relevant].select_dtypes(include=[object]).columns.tolist()
        
        # 1. Group features by default data type
        default_NUM = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'NUM')]        
        default_CAT = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'CAT')]
        
        # 2. Identify features with inconsistent data types
        features_inconsistent_NUM_ = [fe for fe in features_CAT if fe in default_NUM]
        features_inconsistent_CAT_ = [fe for fe in features_NUM if fe in default_CAT]

        # 3. Convert data types of inconsistent features
        if (len(features_inconsistent_NUM_)>0):
            X_tmp[features_inconsistent_NUM_] = X_tmp[features_inconsistent_NUM_].astype(float, coerce=True)

        if (len(features_inconsistent_CAT_)>0):
            X_tmp[features_inconsistent_CAT_] = X_tmp[features_inconsistent_CAT_].astype(object, coerce=True)
        
        return(X_tmp)



class Use_DefaultImputer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to append the default imputation values to missing values and 
    - (2) to examine whether/how the default imputation values change data
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    default_imputers: a data dictionary that provides default imputation values of features
    default_dtypes: a data dictionary that provides default data types of features
    
    Returns
    ---------------
    datafrme
        - X: a df that consists of all features with default data types and imputation values via '.tranform(X)'
        - summary_imputation_: a df that includes the all features with simple summary statistics
        - summary_imputation_NUM_: a df that includes the full summary statistics of NUM features before/after imputation
        - summary_imputation_CAT_: a df that includes the full summary statistics of CAT features before/after imputation
    list
        - NA
        
    References
    ---------------        
    'data_procesing_for_modeling.py'    
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'competitive_area', 'status']
    
    
    def __init__(self, default_imputers, default_dtypes):
        self.default_imputers = default_imputers
        self.default_dtypes   = default_dtypes
        print('*'*50 + '\nPre-Processing: Use_DefaultImputere\n' + '*'*50 + \
              '\n- It will append default imputation values to missings.\n')


    def fit(self, X, y=None):
        # 0. Remove Irrelevant Features
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        
        # 1. Compute Summary Statistics before Imputation
        # Note: dtypes automatically change after imputation.
        X_tmp = X[features_relevant].copy().replace(r'^\s*$', np.nan, regex=True)  # Impute Whitespaces with Nans 
        
        missing_before  = pd.DataFrame(X_tmp.isnull().sum() / X_tmp.shape[0]).reset_index().\
                          rename(columns = {'index': 'feature', 0: 'missing_pct_before_imputation'})
                          
        # Python dtypes
        features_NUM = X_tmp[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT = X_tmp[features_relevant].select_dtypes(include=[object]).columns.tolist()
        # Default dtypes
        default_NUM = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'NUM')]        
        default_CAT = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'CAT')]
        # Features with inconsistent dtypes
        features_inconsistent_NUM_ = [fe for fe in features_CAT if fe in default_NUM]
        features_inconsistent_CAT_ = [fe for fe in features_NUM if fe in default_CAT]

        if (len(features_inconsistent_NUM_)>0):
            X_tmp[features_inconsistent_NUM_] = X_tmp[features_inconsistent_NUM_].astype(float, coerce=True)
        if (len(features_inconsistent_CAT_)>0):
            X_tmp[features_inconsistent_CAT_] = X_tmp[features_inconsistent_CAT_].astype(object, coerce=True)
                          
        desc_NUM_before = X_tmp[default_NUM].describe().T        # Numerical Features with Default dtypes
        desc_CAT_before = X_tmp[default_CAT].describe().T        # Categorical Features with Default dtypes
        
        # 2. Conduct Imputation, and Compute Summary Statistics after Imputation
        # Note: dtypes automatically change after imputation.
        for fe in features_relevant:
            X_tmp[fe].fillna(self.default_imputers[fe], inplace=True)
            
        missing_after  = pd.DataFrame(X_tmp.isnull().sum() / X_tmp.shape[0]).reset_index().\
                         rename(columns = {'index': 'feature', 0: 'missing_pct_after_imputation'})
                          
        # Python dtypes
        features_NUM = X_tmp[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT = X_tmp[features_relevant].select_dtypes(include=[object]).columns.tolist()
        # Default dtypes
        default_NUM = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'NUM')]        
        default_CAT = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'CAT')]
        # Features with inconsistent dtypes
        features_inconsistent_NUM_ = [fe for fe in features_CAT if fe in default_NUM]
        features_inconsistent_CAT_ = [fe for fe in features_NUM if fe in default_CAT]

        if (len(features_inconsistent_NUM_)>0):
            X_tmp[features_inconsistent_NUM_] = X_tmp[features_inconsistent_NUM_].astype(float, coerce=True)
        if (len(features_inconsistent_CAT_)>0):
            X_tmp[features_inconsistent_CAT_] = X_tmp[features_inconsistent_CAT_].astype(object, coerce=True)

        # Note: dtypes automatically change after imputation.
        desc_NUM_after = X_tmp[default_NUM].describe().T        # Numerical Features with Default dtypes
        desc_CAT_after = X_tmp[default_CAT].describe().T        # Categorical Features with Default dtypes

        # 3. Create data frames that summarize data change before/after the imputation
        df_impute                = pd.Series(self.default_imputers).to_frame().reset_index().\
                                   rename(columns = {'index': 'feature', 0: 'default_imputation_value'})
        summary_imputation_      = df_impute.\
                                   merge(missing_before, left_on='feature', right_on='feature', how='inner').\
                                   merge(missing_after, left_on='feature', right_on='feature', how='inner')
        self.summary_imputation_ = summary_imputation_
        
        desc_NUM                     = desc_NUM_before.merge(desc_NUM_after, left_index=True, right_index=True,\
                                                             how='inner', suffixes=('_before', '_after'))
        summary_imputation_NUM_      = summary_imputation_.\
                                       merge(desc_NUM, left_on='feature', right_index=True, how='inner').\
                                       reset_index(drop=True)
        self.summary_imputation_NUM_ = summary_imputation_NUM_                         
        
        desc_CAT                     = desc_CAT_before.merge(desc_CAT_after, left_index=True, right_index=True,\
                                                             how='inner', suffixes=('_before', '_after'))
        summary_imputation_CAT_      = summary_imputation_.\
                                       merge(desc_CAT, left_on='feature', right_index=True, how='inner').\
                                       reset_index(drop=True)
        self.summary_imputation_CAT_ = summary_imputation_CAT_

        return self


    def transform(self, X, y=None):
        # 0. Remove Irrelevant Features in Imputation
        X_tmp             = X.copy()
        features_relevant = list(set(X_tmp.columns) - set(self.features_irrelevant))
        
        # 1. Group features by default data type
        X_tmp[features_relevant] = X_tmp[features_relevant].replace(r'^\s*$', np.nan, regex=True)  # Impute Whitespaces with Nans 
        # Python dtypes
        features_NUM = X_tmp[features_relevant].select_dtypes(exclude=[object]).columns.tolist()
        features_CAT = X_tmp[features_relevant].select_dtypes(include=[object]).columns.tolist()
        # Default dtypes        
        default_NUM = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'NUM')]        
        default_CAT = [fe for fe in features_relevant if (self.default_dtypes[fe] == 'CAT')]

        # 2. Conduct Imputation
        # Note: dtypes automatically change after imputation.
        for fe in features_relevant:
            X_tmp[fe].fillna(self.default_imputers[fe], inplace=True)
        
        # 3. Convert data types of inconsistent features
        features_inconsistent_NUM_ = [fe for fe in features_CAT if fe in default_NUM]
        features_inconsistent_CAT_ = [fe for fe in features_NUM if fe in default_CAT]

        if (len(features_inconsistent_NUM_)>0):
            X_tmp[features_inconsistent_NUM_] = X_tmp[features_inconsistent_NUM_].astype(float, coerce=True)
        if (len(features_inconsistent_CAT_)>0):
            X_tmp[features_inconsistent_CAT_] = X_tmp[features_inconsistent_CAT_].astype(object, coerce=True)
            
        return(X_tmp)                
