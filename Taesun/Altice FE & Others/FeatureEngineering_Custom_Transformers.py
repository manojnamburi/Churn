"""
Author: Taesun Kim
Date:   4/8/2019

Title: Integrate Built-in and Custom Transformers to Primarily Conduct Feature Engineering

Purpose:
    1. Integrate Built-in and Custom Transformers to Primarily Conduct Feature Engineering, 
       Which Are Compatible with General Model Building Process and sklearn API.
        
Custom Transformers That Comprise:
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
        - 

References:    
    - sklearn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
    - sklear Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
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
