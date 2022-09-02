"""
Author: Taesun Kim
Date:   5/13/2019

Title: Create Custom Transformers to Primarily Conduct CLUSTER Feature Engineering

Purpose:
    1. Make Custom Transformers to Conduct Feature Engineering 
       Which Are Compatible with General Model Building Process and sklearn API


Custom Transformers
1. Pre-Processing:
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

2. Feature Engineering for Numerical Features
    1. sklearn-Based Transformers: 
        - Note: Data Will Have 'pandas dataframe' Format after Transformation.
        - `StandardScaler_DF`
        - `Normalizer_DF`
        - `MinMaxScaler_DF`
        - `MaxAbsScaler_DF`
        - `RobustScaler_DF`
        - `Binarizer_DF`
        - `KBinsDiscretizer_DF`
        - `QuantileTransformer_DF`
        - `PowerTransformer_DF`
        - `PolynomialFeatures_DF`
        - `OneHotEncoder_DF`
        - `OrdinalEncoder_DF`
        -
    2. numpy-Based Transformers:
        - Note: Data Will Have 'pandas dataframe' Format after Transformation.
        - `Log1pTransformer`
        - `SqrtTransformer`
        - `ReciprocalTransformer`
        - 
    3. Utility Transformers/Functions:
        - Note: Utility Transformers/Functions Are Used with Other Transformers.
        - `UniversalTransformer`  
        - `PassTransformer`      
        - `FeatureSelector`
        - `FeatureSelector_NUM`
        - `FeatureSelector_CAT`
        - `get_feature_name`

3. Feature Engineering for Categorical Features
    1. `FeatureMaker`
        - Create new features that can be used as grouping variables.
    2. `FeatureAggregator`
        - Aggregate both Numerical and Categorical features by a grouping variable.
        - Use aggregated features as new features.
    3. `RareCategoryEncoder`
        - Re-group rare categories into either 'all_other' or most common category.
        - Create more representative/manageable number of categories.
    4. `FeatureInteractionTransformer`
        - Account for interaction between any pair of given features.
        - Use newly created interaction features in further analyses
    5. `UniversalCategoryEncoder`
        - Encode CATEGORICAL features with selected encoding methods.
        - Eocoding methods:
            - `ohe`: Generate 0/1 binary variable for every label of CATEGORICAL features.
            - `pct`: Replace category with its corresponding %.
            - `count`: Replace category with its corresponding count.
            - `ordinal`: Replace category with its order of average value of target y.
            - `y_mean`: Replace category with its corresponding average value of target y.
            - `y_log_ratio`: Replace category with its corresponding log(p(Churner)/p(Non-Churner)).
            - `y_ratio`: Replace category with its corresponding (p(Churner)/p(Non-Churner)).


References:    
    - sklearn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
    - sklear Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Feature Selector: https://github.com/WillKoehrsen/feature-selector    
    - Featuretools: https://docs.featuretools.com/#
    - Feature Engine: https://pypi.org/project/feature-engine/
    - Category Encoders: http://contrib.scikit-learn.org/categorical-encoding/
    - pandas-pipelines-custom-transformers: https://github.com/jem1031/pandas-pipelines-custom-transformers
    - In-house Code: 'data_procesing_for_modeling.py' 

"""





#####################################################################
#   Required Packages                                               #
#####################################################################

### Import Base Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# To Build Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler,\
                                   Binarizer, KBinsDiscretizer, QuantileTransformer, PowerTransformer,\
                                   PolynomialFeatures, OneHotEncoder, OrdinalEncoder)

# To Build Pipelines
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer





#####################################################################
#   Pre-Processing                                                  #
#####################################################################

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
                           'chc_id', 'house', 'status']
    
    
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
                           'chc_id', 'house', 'status']
    
    
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
                           'chc_id', 'house', 'status']
    
    
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
                           'chc_id', 'house', 'status']
    
    
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
                           'chc_id', 'house', 'status']
    
    
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
                           'chc_id', 'house', 'status']
    
    
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





#####################################################################
#   Feature Engineering: Numerical Features                         #
#####################################################################

#####################################################################
#   Custom Utility Functions                                        #
#####################################################################

def get_feature_name(df_X, y=None, prefix=None, suffix=None):
    '''
    Purpose
    ---------------
    A custom utility function to provide appropriate feature names to transformed data
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df)
    y: a pandas series that represent a churn status
        - default: y=None
    prefix: a string that is appended as prefix to the feature names of df_X
        - default: prefix=None
    suffix: a string that is appended as suffix to the feature names of df_X
        - default: suffix=None

    Returns
    ---------------
    feature_Name: a list of transformed feature names
    '''

    feature_Name     = df_X.columns.tolist()
    
    if (prefix is not None) & (suffix is not None):
        feature_Name = [prefix + '_' + fe + '_' + suffix for fe in df_X.columns]
    elif (prefix is not None):   
        feature_Name = [prefix + '_' + fe for fe in df_X.columns]
    elif (suffix is not None):   
        feature_Name = [fe + '_' + suffix for fe in df_X.columns]

    return feature_Name





#####################################################################
#   Custom Utility Transformers                                     #
#####################################################################

class FeatureUnion_DF(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to concatenate all returns of Custom Transformers in dataframe
    
    Parameters
    ---------------
    transformer_list: a list of Custom Transformers that return dataframe
        - Following sklearn `FeatureUnion` convention, a list of two-item tuple, i.e., (name, transformer), be provided
        - Example: transformer_list = [('Standard', StandardScaler_DF), ('Robust', RobustScaler_DF)]
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.pipeline.FeatureUnion:
        - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    Scikit-learn pipelines and pandas:
        - https://www.kaggle.com/jankoch/scikit-learn-pipelines-and-pandas
    '''
    
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None, **fitparams):
        transformer_fitted = []
        for _, tran in self.transformer_list:
#             fitted_tran = tran.fit(X, y=None, **fitparams)
            fitted_tran = tran.fit(X, y, **fitparams)            
            transformer_fitted.append(fitted_tran)
        self.transformer_fitted = transformer_fitted
        return self

    def transform(self, X, **transformparamn):
        X_transformed = pd.concat([tran.transform(X) for tran in self.transformer_fitted], axis=1)
        return X_transformed



class UniversalTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom UNIVERSAL transformer to transform a given df with a general function a user provides
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
    Note: 'na' is filled with 0.

    References
    ---------------        
    numpy Mathematical Functions:
        - https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.math.html
    '''
    
    def __init__(self, function, y=None, prefix=None, suffix=None):
        self.function = function
        self.prefix   = prefix
        self.suffix   = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = X.apply(self.function).fillna(0).values
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class PassTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to pass a given dataframe to next without any transformation
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None):
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        feature_Name  = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed = pd.DataFrame(X.values, index=X.index, columns=feature_Name)
        return X_transformed



class FeatureSelector(TransformerMixin):
    # A Custom Transformer to Select Given Features Only
    # 'features', a list of features, be provided
    # X_transformed will include 'features' from X.
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X[self.features]
        return X_transformed



class FeatureSelector_NUM(TransformerMixin):
    # A Custom Transformer to Select Numeric Features Only
    # X_transformed will include Numeric Features from X.
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_selected = X.select_dtypes(exclude=[object, 'category']).columns.tolist()
        X_transformed     = X[features_selected]
        return X_transformed



class FeatureSelector_CAT(TransformerMixin):
    # A Custom Transformer to Select Categorical Features Only
    # X_transformed will include Categorical Features from X.
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_selected = X.select_dtypes(include=[object, 'category']).columns.tolist()
        X_transformed     = X[features_selected]
        return X_transformed





#####################################################################
#   numpy-Based Transformers                                        #
#####################################################################

class Log1pTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to transform a given df with numpy.log1p
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
    Note: 'na' is filled with 0.

    References
    ---------------        
    numpy Mathematical Functions:
        - https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.math.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None):
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = X.apply(np.log1p).fillna(0).values
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class SqrtTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to transform a given df with numpy.sqrt
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
    Note: 'na' is filled with 0.

    References
    ---------------        
    numpy Mathematical Functions:
        - https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.math.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None):
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = X.apply(np.sqrt).fillna(0).values
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed

        

class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to transform a given df with numpy.reciprocal
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
    Note: 'na', 'inf', & '-inf' are filled with 0.

    References
    ---------------        
    numpy Mathematical Functions:
        - https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.math.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None):
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = X.apply(np.reciprocal).replace([np.inf, np.NINF], 0).fillna(0).values
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed





#####################################################################
#   sklearn-Based Transformers                                      #
#####################################################################

class StandardScaler_DF(StandardScaler):
    '''
    Purpose
    ---------------
    A custom transformer to make 'StandardScaler' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'StandardScaler' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.StandardScaler:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class RobustScaler_DF(RobustScaler):
    '''
    Purpose
    ---------------
    A custom transformer to make 'RobustScaler' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'RobustScaler' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.RobustScaler:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True):
        super().__init__(with_centering=with_centering, with_scaling=with_scaling, 
                         quantile_range=quantile_range, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class QuantileTransformer_DF(QuantileTransformer):
    '''
    Purpose
    ---------------
    A custom transformer to make 'QuantileTransformer' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'QuantileTransformer' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.QuantileTransformer:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, 
                 subsample=int(1e5), random_state=None, copy=True):
        super().__init__(n_quantiles=n_quantiles, output_distribution=output_distribution, 
                         ignore_implicit_zeros=ignore_implicit_zeros, subsample=subsample, 
                         random_state=random_state, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class PowerTransformer_DF(PowerTransformer):
    '''
    Purpose
    ---------------
    A custom transformer to make 'PowerTransformer' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
    method : str, (default='yeo-johnson')
       - 'yeo-johnson' [1]_, works with positive and negative values
       - 'box-cox' [2]_, only works with strictly positive values
       
    Returns
    ---------------
    Common:
         - All the attributes that 'PowerTransformer' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.PowerTransformer:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 method='yeo-johnson', standardize=True, copy=True):
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class Binarizer_DF(Binarizer):
    '''
    Purpose
    ---------------
    A custom transformer to make 'Binarizer' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'Binarizer' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.Binarizer:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 threshold=0.0, copy=True):
        super().__init__(threshold=threshold, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class MinMaxScaler_DF(MinMaxScaler):
    '''
    Purpose
    ---------------
    A custom transformer to make 'MinMaxScaler' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'MinMaxScaler' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.MinMaxScaler:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 feature_range=(0, 1), copy=True):
        super().__init__(feature_range=feature_range, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class MaxAbsScaler_DF(MaxAbsScaler):
    '''
    Purpose
    ---------------
    A custom transformer to make 'MaxAbsScaler' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'MaxAbsScaler' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.MaxAbsScaler:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, copy=True):
        super().__init__(copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class Normalizer_DF(Normalizer):
    '''
    Purpose
    ---------------
    A custom transformer to make 'Normalizer' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'Normalizer' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.Normalizer:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 norm='l2', copy=True):
        super().__init__(norm=norm, copy=copy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed



class KBinsDiscretizer_DF(KBinsDiscretizer):
    '''
    Purpose
    ---------------
    A custom transformer to make 'KBinsDiscretizer' return the outcome in dataframe
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    prefix: a string to add a prefix to all column names
       - default: prefix=None
    suffix: a string to add a suffix to all column names
       - default: suffix=None
       
    Returns
    ---------------
    Common:
         - All the attributes that 'KBinsDiscretizer' returns
    Specific:
        - X_transformed: a transformed outcome in dataframe with proper column/index names
        
    References
    ---------------        
    sklearn.preprocessing.KBinsDiscretizer:
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
    '''
    
    def __init__(self, y=None, prefix=None, suffix=None, 
                 n_bins=5, encode='onehot', strategy='quantile'):
        super().__init__(n_bins=n_bins, encode=encode, strategy=strategy)
        self.prefix = prefix
        self.suffix = suffix
        
    def transform(self, X):
        # Note: A custom utility function, 'get_feature_name' is used.
        tmp_transformed = super().transform(X.values)
        feature_Name    = get_feature_name(X, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed   = pd.DataFrame(tmp_transformed, index=X.index, columns=feature_Name)
        return X_transformed





#####################################################################
#   Feature Engineering: Categorical Features                       #
#####################################################################

#####################################################################
#   Custom Utility Functions                                        #
#####################################################################

def aggregate_features(df_X, group_feature, max_cateogry_count=20):
    '''
    Purpose
    ---------------
    A custom utility function to aggregate features by a given grouping variable
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df) with all features
        - df_X is expected to be processed via custom pre-processing transformers.
    group_feature: a grouping variable on which features are aggregated by data type

    Returns
    ---------------
    df_aggregated: a df with newly aggregated features
    '''

    # NUMERICAL Features
    features_NUM      = df_X.select_dtypes(exclude=[object, 'category']).columns.tolist()
    features_NUM      = list(set(features_NUM) - set([group_feature]))  
    features_relevant = [group_feature] + features_NUM
    df_NUM            = df_X[features_relevant].copy()

    key_stats         = ['count', 'mean', 'median', 'max', 'min', 'std', 'sum']
    tmp_NUM           = df_NUM.groupby(group_feature).agg(key_stats)

    feature_names     = []
    for fe in tmp_NUM.columns.levels[0]:
        for key_stat in tmp_NUM.columns.levels[1]:
            feature_names.append(f'{group_feature.upper()}__*__{fe}__*__{key_stat}')

    tmp_NUM.columns   = feature_names
    tmp_NUM.reset_index(inplace=True)


    # CATEGORICAL Features
    features_CAT          = df_X.select_dtypes(include=[object, 'category']).columns.tolist()
    list_nunique          = df_X[features_CAT].nunique()
    # Exclude CAT Features if nunique >= 20.    
    features_CAT          = list_nunique[list_nunique <= max_cateogry_count].index.tolist()
    features_CAT          = list(set(features_CAT) - set([group_feature]))
    features_CAT.sort()
    features_relevant     = [group_feature] + features_CAT
    df_CAT                = df_X[features_relevant].copy()

    tmp_df_list           = []
    for fe in features_CAT:
        tmp_df            = df_CAT.groupby(group_feature)[fe].value_counts(normalize=True).to_frame().\
                            rename(columns={fe: f'{fe}_Pct'}).reset_index()
        tmp_pivot         = pd.pivot_table(tmp_df, index=group_feature, columns=fe, values=f'{fe}_Pct', 
                                        aggfunc='mean', fill_value=0)
        tmp_pivot.columns = [f'{group_feature.upper()}__*__{fe}__*__{var}' for var in tmp_pivot.columns]
        tmp_df_list.append(tmp_pivot)

    tmp_CAT               = pd.concat(tmp_df_list, axis=1)

    # Concatenate aggregated NUM and CAT features.
    df_aggregated         = pd.merge(tmp_NUM, tmp_CAT, on=group_feature, how='inner')

    return df_aggregated





#####################################################################
#   Custom Category Encoders                                        #
#####################################################################

class FeatureMaker(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer to create new features that can be used as grouping variables
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    features_new_: 
        - a list of newly created features

    References
    ---------------        
    '''
    
    def __init__(self):
        pass


    def fit(self, X, y=None):
        # Note: a list of new features will be constantly increased.
        # May need prod/srv change grouping
        features_new_      = ['grp_tenure_3m', 'grp_tenure_1m', 'grp_tenure_6m', \
                              'grp_payment_method', \
                              'grp_payment_25dollar', 'grp_payment_10dollar', \
                              'grp_payment_change_5dollar', 'grp_payment_change_10dollar',\
                              'grp_payment_change_2pct', 'grp_payment_change_5pct',\
                              'ratio_payment_income', 'grp_payment_income',\
                              'grp_call_csc', 'grp_call_bill', \
                              'grp_call_csr', 'grp_call_tsr']
        self.features_new_ = features_new_

        return self


    def transform(self, X):
        ### 'grp_tenure'
        ##  'grp_tenure_3m': grouping tenure by 3 months
        tmp_fe      = ['kom_tenure']
        tmp_df      = X[tmp_fe]

        tmp_bins    = [-float('inf')] + np.arange(12, 97, 3).tolist() + [float('inf')]
        tmp_labels  = ['Under 12 Months'] + \
                    [f'{cat+1} ~ {cat+3} Months' for cat in np.arange(12, 97, 3)]
        tmp_df['grp_tenure_3m'] = pd.cut(tmp_df.kom_tenure, bins=tmp_bins, labels=tmp_labels)

        ##  'grp_tenure_1m': grouping tenure by 1 month
        tmp_bins    = [-float('inf')] + np.arange(12, 97, 1).tolist() + [float('inf')]
        tmp_labels  = ['Under 12 Months'] + \
                    [f'{cat+1} Months' for cat in np.arange(12, 97, 1)]
        tmp_df['grp_tenure_1m'] = pd.cut(tmp_df.kom_tenure, bins=tmp_bins, labels=tmp_labels)

        ##  'grp_tenure_6m': grouping tenure by 6 months
        tmp_bins    = [-float('inf')] + np.arange(12, 97, 6).tolist() + [float('inf')]
        tmp_labels  = ['Under 12 Months'] + \
                    [f'{cat+1} ~ {cat+6} Months' for cat in np.arange(12, 97, 6)]
        tmp_df['grp_tenure_6m'] = pd.cut(tmp_df.kom_tenure, bins=tmp_bins, labels=tmp_labels)
        tmp_transformed         = tmp_df[['grp_tenure_3m', 'grp_tenure_1m', 'grp_tenure_6m']]

        ### 'grp_payment_method'
        # Note: Most of 'AUTPAY' is made by 'CREDIT'. 
        # So 'AUTPAY' with 'CREDIT' is classified as 'AUTPAY'.
        tmp_fe      = ['autpay_payments_6m', 'credit_payments_6m', \
                       'debit_payments_6m', 'cash_payments_6m', \
                       'center_payments_6m', 'check_payments_6m']
        tmp_df      = X[tmp_fe]
        tmp_df['grp_payment_method'] = tmp_df.idxmax(axis=1).str.replace('_payments_6m', '').\
                                       str.upper()
        tmp_transformed = tmp_transformed.merge(tmp_df[['grp_payment_method']], how='inner', \
                                                left_index=True, right_index=True)                               

        ### 'grp_payment_dollar'
        tmp_fe      = ['revenue_m1', 'revenue_m2', 'revenue_m3', \
                       'revenue_m4', 'revenue_m5', 'revenue_m6']
        tmp_df      = X[tmp_fe]
        tmp_df['revenue_mean'] = tmp_df.mean(axis=1)

        ##  'grp_payment_25dollar'
        tmp_bins    = [-float('inf')] + np.arange(50, 250, 25).tolist() + [float('inf')]
        tmp_labels  = ['Under 50 Dollars'] + \
                      [f'{cat+1} ~ {cat+25} Dollars' for cat in np.arange(50, 250, 25)] 
        tmp_df['grp_payment_25dollar'] = pd.cut(tmp_df.revenue_mean, bins=tmp_bins, labels=tmp_labels)

        ##  'grp_payment_10dollar'
        tmp_bins    = [-float('inf')] + np.arange(50, 250, 10).tolist() + [float('inf')]
        tmp_labels  = ['Under 50 Dollars'] + \
                      [f'{cat+1} ~ {cat+10} Dollars' for cat in np.arange(50, 250, 10)] 
        tmp_df['grp_payment_10dollar'] = pd.cut(tmp_df.revenue_mean, bins=tmp_bins, labels=tmp_labels)
        tmp_transformed = tmp_transformed.merge(tmp_df[['grp_payment_25dollar', 'grp_payment_10dollar']], \
                                                how='inner', left_index=True, right_index=True)

        ### 'grp_payment_change'
        ##  'grp_payment_change_5dollar'
        tmp_fe     = ['ttl_rev_abs_change_6m', 'ttl_rev_perc_change_6m']
        tmp_df     = X[tmp_fe]
        tmp_bins   = [-float('inf')] + np.arange(-50, -4, 5).tolist() + \
                     [-0.001, 0] + np.arange(5, 55, 5).tolist() + [float('inf')]
        tmp_labels = ['Under -50 Dollars'] + [f'{cat} ~ {cat+4} Dollars' for cat in np.arange(-50, -4, 5)] + \
                     ['0 Dollar'] + \
                     [f'{cat+1} ~ {cat+5} Dollars' for cat in np.arange(0, 50, 5)] + \
                     ['Over 50 Dollars']
        tmp_df['grp_payment_change_5dollar'] = pd.cut(tmp_df.ttl_rev_abs_change_6m, \
                                                      bins=tmp_bins, labels=tmp_labels)

        ##  'grp_payment_change_10dollar'
        tmp_bins   = [-float('inf')] + np.arange(-50, -4, 10).tolist() + \
                     [-0.001, 0] + np.arange(5, 55, 10).tolist() + [float('inf')]
        tmp_labels = ['Under -50 Dollars'] + [f'{cat} ~ {cat+9} Dollars' for cat in np.arange(-50, -4, 10)] + \
                     ['0 Dollar'] + \
                     [f'{cat+1} ~ {cat+10} Dollars' for cat in np.arange(0, 50, 10)] + \
                     ['Over 50 Dollars']
        tmp_df['grp_payment_change_10dollar'] = pd.cut(tmp_df.ttl_rev_abs_change_6m, \
                                                       bins=tmp_bins, labels=tmp_labels)

        ##  'grp_payment_change_2pct'
        tmp_bins   = [-float('inf')] + np.arange(-20, 0, 2).tolist() + \
                     [-0.001, 0] + np.arange(2, 22, 2).tolist() + [float('inf')]
        tmp_labels = ['Under -20 Pct'] + [f'{cat} ~ {cat+1} Pct' for cat in np.arange(-20, -1, 2)] + \
                     ['0 Pct'] + \
                     [f'{cat+1} ~ {cat+2} Pct' for cat in np.arange(0, 20, 2)] + ['Over 20 Pct']
        tmp_df['grp_payment_change_2pct'] = pd.cut(tmp_df.ttl_rev_perc_change_6m, \
                                                   bins=tmp_bins, labels=tmp_labels)

        ##  'grp_payment_change_5pct'
        tmp_bins   = [-float('inf')] + np.arange(-30, 0, 5).tolist() + \
                     [-0.001, 0] + np.arange(5, 35, 5).tolist() + [float('inf')]
        tmp_labels = ['Under -30 Pct'] + [f'{cat} ~ {cat+4} Pct' for cat in np.arange(-30, 0, 5)] + \
                     ['0 Pct'] + \
                     [f'{cat+1} ~ {cat+5} Pct' for cat in np.arange(0, 30, 5)] + ['Over 30 Pct']
        tmp_df['grp_payment_change_5pct'] = pd.cut(tmp_df.ttl_rev_perc_change_6m, 
                                                   bins=tmp_bins, labels=tmp_labels)        
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['grp_payment_change_5dollar', 'grp_payment_change_10dollar', \
                                'grp_payment_change_2pct', 'grp_payment_change_5pct']], \
                                how='inner', left_index=True, right_index=True)

        ### 'grp_payment_income'
        tmp_fe                  = ['income_demos', 'revenue_m1', 'revenue_m2', 'revenue_m3',\
                                   'revenue_m4', 'revenue_m5', 'revenue_m6']
        tmp_df                  = X[tmp_fe]
        tmp_df['revenue_mean']  = tmp_df.mean(axis=1)
        tmp_map                 = {'a.$1-$14,999': 8000, 'b.$15k-$24': 20000, 'c.$25k-$34': 30000, \
                                   'd.$35k-$49': 43000, 'e.$50k-$74': 63000, 'f.$75k-$99': 88000, \
                                   'g.$100k-$124':113000, 'h.$125k-$149': 138000, 'i.$150k-$174': 163000,\
                                   'j.$175k-$199': 188000, 'k.$200k-$249': 225000, 'l.$250k+': 300000,\
                                   'UNKNOWN': 63000} # 'e.$50k-$74' is the most common category.
        tmp_df['income_dollars'] = tmp_df.income_demos.map(tmp_map)

        ##  'ratio_payment_income' = annualized payments / income
        tmp_df['ratio_payment_income'] = 100 * 12 * tmp_df['revenue_mean'] / tmp_df['income_dollars']

        ##  'grp_payment_income'
        tmp_bins                     = [-1, 1, 2, 3, 4, 5, 7, 10, 15, 20, float('inf')]
        tmp_labels                   = ['Under 1 pct', '1 ~ 1.99 pct', '2 ~ 2.99 pct', '3 ~ 3.99 pct',\
                                        '4 ~ 4.99 pct', '5 ~ 6.99 pct', '7 ~ 9.99 pct', '10 ~ 14.99 pct',\
                                        '15 ~ 19.99 pct', 'Over 20 pct']
        tmp_df['grp_payment_income'] = pd.cut(tmp_df.ratio_payment_income, \
                                              bins=tmp_bins, labels=tmp_labels)
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['ratio_payment_income', 'grp_payment_income']], \
                                how='inner', left_index=True, right_index=True)        

        ### 'grp_call'
        ##  'grp_call_csc'
        tmp_fe                   = ['csc_m1', 'csc_m2', 'csc_m3', 'csc_m4', 'csc_m5', 'csc_m6']
        tmp_df                   = X[tmp_fe]
        tmp_df['call_csc_total'] = tmp_df.sum(axis=1)
        tmp_bins                 = np.arange(-2, 11, 2).tolist() + [float('inf')]
        tmp_labels               = ['0 Call'] + \
                                   [f'{cat+1} ~ {cat+2} Calls' for cat in np.arange(0, 9, 2)] + \
                                   ['10+ Calls']
        tmp_df['grp_call_csc']   = pd.cut(tmp_df.call_csc_total, bins=tmp_bins, labels=tmp_labels)
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['grp_call_csc']], \
                                how='inner', left_index=True, right_index=True)        

        ##  'grp_call_bill'
        tmp_fe                    = ['bill_m1', 'bill_m2', 'bill_m3', 'bill_m4', 'bill_m5', 'bill_m6']
        tmp_df                    = X[tmp_fe]
        tmp_df['call_bill_total'] = tmp_df.sum(axis=1)
        tmp_bins                  = np.arange(-1, 5, 1).tolist() + [float('inf')]
        tmp_labels                = ['0 Call'] + [f'{cat+1} Calls' for cat in np.arange(0, 4, 1)] + \
                                    ['4+ Calls']
        tmp_df['grp_call_bill']   = pd.cut(tmp_df.call_bill_total, bins=tmp_bins, labels=tmp_labels)
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['grp_call_bill']], \
                                how='inner', left_index=True, right_index=True)        

        ##  'grp_call_csr'
        tmp_fe                   = ['csr_no_of_calls_m1', 'csr_no_of_calls_m2', \
                                    'csr_no_of_calls_m3', 'csr_no_of_calls_m4', \
                                    'csr_no_of_calls_m5', 'csr_no_of_calls_m6']
        tmp_df                   = X[tmp_fe]
        tmp_df['call_csr_total'] = tmp_df.sum(axis=1)
        tmp_bins                 = np.arange(-2, 11, 2).tolist() + [float('inf')]
        tmp_labels               = ['0 Call'] + \
                                   [f'{cat+1} ~ {cat+2} Calls' for cat in np.arange(0, 9, 2)] + \
                                   ['10+ Calls']
        tmp_df['grp_call_csr']   = pd.cut(tmp_df.call_csr_total, \
                                          bins=tmp_bins, labels=tmp_labels)        
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['grp_call_csr']], \
                                how='inner', left_index=True, right_index=True)        

        ##  'grp_call_tsr'
        tmp_fe                   = ['tsr_no_of_calls_m1', 'tsr_no_of_calls_m2',\
                                    'tsr_no_of_calls_m3', 'tsr_no_of_calls_m4',\
                                    'tsr_no_of_calls_m5', 'tsr_no_of_calls_m6']
        tmp_df                   = X[tmp_fe]
        tmp_df['call_tsr_total'] = tmp_df.sum(axis=1)
        tmp_bins                 = np.arange(-1, 5, 1).tolist() + [float('inf')]
        tmp_labels               = ['0 Call'] + \
                                   [f'{cat+1} Calls' for cat in np.arange(0, 4, 1)] + \
                                   ['4+ Calls']
        tmp_df['grp_call_tsr']   = pd.cut(tmp_df.call_tsr_total, \
                                        bins=tmp_bins, labels=tmp_labels)
        tmp_transformed = tmp_transformed.\
                          merge(tmp_df[['grp_call_tsr']], \
                                how='inner', left_index=True, right_index=True)        

        X_transformed   = tmp_transformed.copy()

        return X_transformed



class FeatureAggregator(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to aggregate both Numerical and Categorical features by a grouping variable
    - (2) to use aggregated features as new features in further analysis

    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    features_grouping: a list of grouping features on which both Numerical and Categorical features are aggregated
        - e.g.: features_grouping = ['census', 'cleansed_city', 'cleansed_zipcode']
    correlation_threshold: abs(correlation) ~ [0, 1]
       - default: correlation_threshold=0.01
       - Note:
            - Most of the features are not informative.
            - Selected are features with abs(correlation) >= correlation_threshold.
    category_max_count: the number of categories/labels after transformation
       - default: category_count_threshold=31
            - The max number of categories after transformation <= 31
            - category_count_threshold=31 --> Top 30 categories and 'All Other'

    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    features_aggregate_:
        - a list of aggregated features with abs(corr) >= 'correlation_threshold'
    features_original_
        - a list of original features resulting in seleted aggregated features

    References
    ---------------  
    Featuretools: 
        - https://docs.featuretools.com/#      
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant   = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                             'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                             'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                             'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                             'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                             'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                             'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', \
                             'chc_id', 'house', 'status']

    # Need to specify grouping variables that will be used in feature aggregation.
    # features_geo_cardinal = a list of high cardinal/geographic features that can be used as grouping variables
    features_geo_cardinal = ['census', 'cleansed_city', 'cleansed_zipcode', 'ecohort_code', \
                             'clust', 'corp', 'fta_desc', 'house', 'hub', 'trunk', 'node', \
                             'geo', 'geo2', 'geo3']
    
    def __init__(self, features_grouping, correlation_threshold=0.01, category_max_count=31):
        if not isinstance(features_grouping, list):
            raise ValueError('Features should be provided in list.')
        print("'FeatureAggregator' requires target y.")

        features_grouping.sort() 
        self.features_grouping     = features_grouping
        self.correlation_threshold = correlation_threshold
        self.category_max_count    = category_max_count        


    def fit(self, X, y):
        # Create a List of  Relevant Features as a Private Variable
        _features_relevant      = list(set(X.columns) - (set(self.features_irrelevant + self.features_geo_cardinal)))
        _features_relevant.sort()
        self._features_relevant = _features_relevant        

        # Note: A custom utility function, 'aggregate_features' is used.
        tmp_df_list           = []

        for fe in self.features_grouping:
            features_relevant = list((set([fe] + _features_relevant)))
            tmp_df            = X[features_relevant]
            tmp_agg           = aggregate_features(tmp_df, fe, max_cateogry_count=self.category_max_count)
            tmp_agg.set_index([fe], inplace=True)
            tmp_df2           = tmp_df[[fe]].sort_values([fe]).reset_index()
            tmp_df2.set_index([fe], inplace=True)
            tmp_df_agg        = tmp_agg.join(tmp_df2)         # DO NOT USE pd.MERGE!!!
            tmp_df_agg.set_index(['chc_id'], inplace=True)
            tmp_df_agg        = tmp_df_agg.sort_index()
            tmp_df_list.append(tmp_df_agg)

        tmp_transformed       = pd.concat(tmp_df_list, axis=1)

        # The number of features are too large, which should be reduced!
        # Let's select features based on 'correlation_threshold'.
        if y is None:
            raise ValueError("Target y should be provided.")
        if not (tmp_transformed.index.equals(y.index)):
            raise ValueError("Target y should be sorted by 'chc_id' index.")

        tmp_corr                 = tmp_transformed.apply(lambda x: x.corr(y))
        tmp_features_aggregate_  = tmp_corr[abs(tmp_corr) >= self.correlation_threshold].index.tolist()
        tmp_features_aggregate_.sort()
        tmp_features_original_   = [fe.split('__*__')[1] for fe in tmp_features_aggregate_]
        tmp_features_original_   = list(set(tmp_features_original_))
        tmp_features_original_.sort()

        self.features_aggregate_ = tmp_features_aggregate_  # Aggregated features with abs(corr) >= 'correlation_threshold'
        self.features_original_  = tmp_features_original_   # Original features resulting in seleted aggregated features

        return self


    def transform(self, X, y=None):
        # Note: A custom utility function, 'aggregate_features' is used.
        tmp_df_list           = []

        # Don't use pd.merge for big data. It takes too long!!!
        # Use a small df first and join a large df sencond.
        for fe in self.features_grouping:
            features_relevant = list((set([fe] + self.features_original_)))
            tmp_df            = X[features_relevant]
            tmp_agg           = aggregate_features(tmp_df, fe, max_cateogry_count=self.category_max_count)
            tmp_agg.set_index([fe], inplace=True)
            tmp_df2           = tmp_df[[fe]].sort_values([fe]).reset_index()
            tmp_df2.set_index([fe], inplace=True)
            tmp_df_agg        = tmp_agg.join(tmp_df2)         # DO NOT USE pd.MERGE!!!
            tmp_df_agg.set_index(['chc_id'], inplace=True)
            tmp_df_agg        = tmp_df_agg.sort_index()
            tmp_df_list.append(tmp_df_agg)
        
        # Concatenate aggregated dfs by a grouping feature.
        tmp_transformed       = pd.concat(tmp_df_list, axis=1)

        # Include 'self.features_aggregate_' only.
        # If any feature is not in 'self.features_aggregate_', create those missing features.
        # TRAIN and TEST may not exactly match each other in categories!
        fe_missing       = list(set(self.features_aggregate_) - set(tmp_transformed.columns))
        if len(fe_missing)>=1:
            for fe in fe_missing:
                tmp_transformed[fe] = 0

        X_transformed         = tmp_transformed[self.features_aggregate_]
        # X_transformed.columns = [fe.replace('__*__', '__') for fe in X_transformed.columns]

        return X_transformed



class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) to re-group rare categories into either 'all_other' or most common category
    - (2) to create more representative/manageable number of categories
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    category_min_pct: category Pct ~ [0, 1]
       - default: category_min_pct=0.01 after transformation
            - The % of any category be >= 1% of total sample.
            - Otherwise, this category is relabeled as either (1) 'all_other' or
              (2) the most common category after transformation.
    category_max_count: the number of categories/labels after transformation
       - default: category_count_threshold=20
            - The max number of categories after transformation <= 20
    encoding_method: how to re-classify rare categories
       - default: encoding_method=None --> assigned into 'all_other' category.
       - encoding_method='common' --> assigned into the most common category.
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    category_mapper_: 
        - a data dictionary of mapping features to corresponding categories after encoding

    References
    ---------------        
    Feature Engine: 
        - https://pypi.org/project/feature-engine/
        - https://github.com/solegalli/feature_engine/blob/master/feature_engine/categorical_encoders.py
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant   = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                             'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                             'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                             'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                             'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                             'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                             'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', \
                             'chc_id', 'house', 'status']

    # Need to specify grouping variables that will be used in feature aggregation.
    # features_geo_cardinal = a list of high cardinal/geographic features that can be used as grouping variables
    features_geo_cardinal = ['census', 'cleansed_city', 'cleansed_zipcode', 'ecohort_code', \
                             'clust', 'corp', 'fta_desc', 'house', 'hub', 'trunk', 'node', \
                             'geo', 'geo2', 'geo3']

    def __init__(self, category_min_pct=0.01, category_max_count=20, encoding_method=None, prefix=None, suffix=None):
        self.category_min_pct   = category_min_pct
        self.category_max_count = category_max_count
        self.encoding_method    = encoding_method
        self.prefix             = prefix
        self.suffix             = suffix


    def fit(self, X, y=None):
        # 0. Select Relevant Features for Rare Category Transformation
        features_relevant        = list(set(X.columns) - (set(self.features_irrelevant + self.features_geo_cardinal)))
        features_relevant.sort()
        # Note: 'category' is not intentionally included in dtypes.
        features_CAT_            = X[features_relevant].select_dtypes(include=[object]).columns.tolist()

        # 1. Create Catetory Mapping Dictionary
        # - Total numer of categories = self.category_max_count
        # - Min pct of selected categories >= self.category_min_pct
        category_mapper_         = {}

        for fe in features_CAT_:
            mapping              =  X[fe].value_counts(normalize=True).iloc[:self.category_max_count]
            category_mapper_[fe] = mapping[mapping >= self.category_min_pct].index

        self.features_CAT_       = features_CAT_
        self.category_mapper_    = category_mapper_ 

        return self


    def transform(self, X, y=None):
        # encoding_method: 
        #   - default: 'all_other' for rare categories
        #   - if method = 'common', then the most common category for rare categories
        tmp_df             = X.copy()

        for fe in self.features_CAT_:
            if self.encoding_method is None:
                tmp_df[fe] =  np.where(tmp_df[fe].isin(self.category_mapper_[fe]), tmp_df[fe], 'all_other')
            elif self.encoding_method == 'common':
                tmp_df[fe] =  np.where(tmp_df[fe].isin(self.category_mapper_[fe]), \
                                       tmp_df[fe], self.category_mapper_[fe][0])

        # Note: A custom utility function, 'get_feature_name' is used.                                       
        tmp_df.columns     = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed      = tmp_df

        return X_transformed            



class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to account for interaction between any pair of given features    
    - (2) to use newly created interaction features in further analyses
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    features_1st: a list of primary features to be examined
        - e.g.: features_1st = ['income_demos', 'ethnic']
    features_2nd: a list of secondary features to be examined
        - e.g.: features_2nd = ['income_demos', 'ethnic', 'age_demos', 'cleansed_city', 'cleansed_zipcode']
        - Note: 'features_1st' and 'features_2nd' can have common features.
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    features_new_: 
        - a list of newly created features

    References
    ---------------        
    '''
    
    def __init__(self, features_1st, features_2nd):
        if not isinstance(features_1st, list):
            raise ValueError('Features should be provided in list.')
        if not isinstance(features_2nd, list):
            raise ValueError('Features should be provided in list.')

        features_1st.sort() 
        features_2nd.sort()   
        self.features_1st = features_1st 
        self.features_2nd = features_2nd


    def fit(self, X, y=None):
        features_new_      = []

        for fe1 in self.features_1st:
            for fe2 in self.features_2nd:
                if (fe1 != fe2) & (f'{fe2.upper()}-{fe1}' not in features_new_):
                    features_new_.append(f'{fe1.upper()}-{fe2}')

        self.features_new_ = features_new_

        return self


    def transform(self, X, y=None):
        tmp_df        = X.copy()

        for fe1 in self.features_1st:
            for fe2 in self.features_2nd:
                if (fe1 != fe2) & (f'{fe2.upper()}-{fe1}' not in tmp_df.columns):
                    tmp_df[f'{fe1.upper()}-{fe2}'] = tmp_df[fe1].astype('str') + '___' + tmp_df[fe2].astype('str')

        X_transformed = tmp_df

        return X_transformed



class UniversalCategoryEncoder(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom UNIVERSAL encoder to encode CATEGORICAL features with a selected encoding method
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
        - default: y=None
    encoding_method: ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']
        - One must choose one of the listed encoding methods.
        - 'ohe': OneHotEonding
            - Generate 0/1 binary variable for every category of CATEGORICAL features.
        - 'pct': 
            - Replace category with its corresponding %.
        - 'count':
            - Replace category with its corresponding count.
        - 'ordinal':
            - y should be given.
            - Replace category with its order of average value of target y.
        - 'y_mean':
            - y should be given.
            - Replace category with its corresponding average value of target y.            
        - 'y_log_ratio':
            - y should be given.
            - Replace category with its corresponding log(p(Churner)/p(Non-Churner)).
            - p(Churner) = 0 be set to p(Churner) = 0.00001 to avoid -inf.
        - 'y_ratio':        
            - y should be given.    
            - Replace category with its corresponding (p(Churner)/p(Non-Churner)).
    prefix: a string to add a prefix to all column names
        - default: prefix=None
    suffix: a string to add a suffix to all column names
        - default: suffix=None

    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    category_mapper_: 
        - a data dictionary of mapping features to corresponding categories after encoding

    References
    ---------------        
    Feature Engine: 
        - https://pypi.org/project/feature-engine/
        - https://github.com/solegalli/feature_engine/blob/master/feature_engine/categorical_encoders.py
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant   = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                             'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                             'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                             'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                             'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                             'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                             'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', \
                             'chc_id', 'house', 'status']

    # Need to specify grouping variables that will be used in feature aggregation.
    # features_geo_cardinal = a list of high cardinal/geographic features that can be used as grouping variables
    features_geo_cardinal = ['census', 'cleansed_city', 'cleansed_zipcode', 'ecohort_code', \
                             'clust', 'corp', 'fta_desc', 'house', 'hub', 'trunk', 'node', \
                             'geo', 'geo2', 'geo3']

    def __init__(self, encoding_method, prefix=None, suffix=None):
        if encoding_method not in ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']:
            raise ValueError("ONE encoding method should be choosen from ['ohe', 'pct', 'count', 'ordinal', 'y_mean', 'y_log_ratio', 'y_ratio'].")

        if encoding_method in ['ordinal', 'y_mean', 'y_log_ratio', 'y_ratio']:
            print(f"'{encoding_method}' encoding requires target y.")

        self.encoding_method = encoding_method
        self.prefix          = prefix
        self.suffix          = suffix
        

    def fit(self, X, y=None):
        # 0. Select Relevant Features
        features_relevant = list(set(X.columns) - (set(self.features_irrelevant + self.features_geo_cardinal)))
        features_relevant.sort()
        features_CAT_     = X[features_relevant].select_dtypes(include=[object, 'category']).columns.tolist()

        # Create Catetory Mapping Dictionary
        category_mapper_             = {}

        if self.encoding_method      == 'ohe':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts().index

        elif self.encoding_method    == 'pct':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts(normalize=True).to_dict()

        elif self.encoding_method    == 'count':
            for fe in features_CAT_:
                category_mapper_[fe] = X[fe].value_counts().to_dict()

        elif self.encoding_method    == 'ordinal':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                mapping              = tmp_df.groupby(fe)['y'].mean().sort_values(ascending=True).index
                category_mapper_[fe] = {j:i for i, j in enumerate(mapping, 0)}

        elif self.encoding_method    == 'y_mean':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                category_mapper_[fe] = tmp_df.groupby(fe)['y'].mean().sort_values(ascending=False).to_dict()

        elif self.encoding_method    == 'y_log_ratio':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                tmp_stat             = tmp_df.groupby(fe)['y'].mean()
                tmp_stat             = pd.concat([tmp_stat, 1-tmp_stat], axis=1)
                tmp_stat.columns     = ['p1', 'p0']
                # Assign a small value, .00001 if p = 0
                tmp_stat['p1']       = np.where(tmp_stat['p1'] <= 0, .00001, tmp_stat['p1'])
                tmp_stat['p0']       = np.where(tmp_stat['p0'] <= 0, .00001, tmp_stat['p0'])
                category_mapper_[fe] = (np.log(tmp_stat.p1/tmp_stat.p0)).to_dict()

        elif self.encoding_method    == 'y_ratio':
            tmp_df                   = pd.concat([y, X], axis=1)
            tmp_df.columns           = ['y'] + list(X)
            for fe in features_CAT_:
                tmp_stat             = tmp_df.groupby(fe)['y'].mean()
                tmp_stat             = pd.concat([tmp_stat, 1-tmp_stat], axis=1)
                tmp_stat.columns     = ['p1', 'p0']
                # Assign a small value, .00001 if p = 0
                tmp_stat['p0']       = np.where(tmp_stat['p0'] <= 0, .00001, tmp_stat['p0'])
                category_mapper_[fe] = (tmp_stat.p1/tmp_stat.p0).to_dict()

        self.features_CAT_           = features_CAT_
        self.category_mapper_        = category_mapper_

        return self


    def transform(self, X, y=None):
        # Note: A custom utility function, 'get_feature_name' is used.        
        tmp_df             = X[self.features_CAT_]

        if self.encoding_method == 'ohe':
            for fe in self.features_CAT_:
                for cat in self.category_mapper_[fe]:
                    tmp_df[f'{fe}_{cat}'] = np.where(tmp_df[fe] == cat, 1, 0)
            # Drop the original features for 'ohe' only.
            tmp_df.drop(self.features_CAT_, axis=1, inplace=True)
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)

        elif self.encoding_method == 'y_log_ratio':
            # Note: log(0) = - inf --> replace Null with a very small prob such as .00001.
            for fe in self.features_CAT_:
                tmp_df[fe] = tmp_df[fe].map(self.category_mapper_[fe]).fillna(np.log(.00001/(1-.00001)))
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)    

        else:
            for fe in self.features_CAT_:
                tmp_df[fe] = tmp_df[fe].map(self.category_mapper_[fe]).fillna(0)
            tmp_df.columns = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)    

        X_transformed = tmp_df

        return X_transformed
