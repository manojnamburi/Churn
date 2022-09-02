

"""
Author: Manoj Namburi
Date:   07/18/2022

Title: Create Custom Transformers to Primarily Conduct ML Model Training and Evaluation

Purpose:
    
"""

#####################################################################
#   Required Packages                                               #
# ####################################################################

### Import Base Modules
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import time

# To Build Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler,\
                                   Binarizer, KBinsDiscretizer, QuantileTransformer, PowerTransformer,\
                                   PolynomialFeatures, OneHotEncoder, OrdinalEncoder)

from sklearn.model_selection import cross_val_score                                   

# To Build Pipelines
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# To Build ML models

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score as acc, recall_score as recall, roc_auc_score as auc, f1_score as f1, roc_curve, confusion_matrix
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score as precision


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



class fixed_churn():
    '''
    Purpose 
    ------------------------------  

    Prepare model input and evaluation model output

    Funtions
    ---------------------------------
    1, downsampling the dataset in order to have a comparable postive and negative classes
        **** Inpute: dataframe that needs downsampling 
                     Target variable name
        **** Output: downsampled X
                     downsampled y

    2, feature selections - get the most important features for future model use
        **** Input: X_train_down, y_train_down
        **** Output: dataframe with feature number and AUC score
        ****         dataframe with features and rank

    3, grid_search - get best hyper parameters
        **** Input: X_train_down[best features], y_train_down
        **** Output: best hyper parameters

    4, model evaluation - evaluate model results
        **** Input: best model,X_train_best_features, y_train_down, X_test_best_features, y_test
        **** Output: confusion_matrix, metrics_summary - accuracy, recall and f1

    5, lift analysis - get lift curve for a df
        **** Input: y_test, y_test_proba
        **** Output: df with lift and also share of churn
    '''

    def __init__(self):
        
        pass

    def downsampling(self, df, feature): 

        ####################
        # 1, downsampling the dataset in order to have a comparable postive and negative classes
        # **** Inpute: dataframe that needs downsampling 
        #             Target variable name
        # **** Output: downsampled X
        #             downsampled y
        
        print('*'*50 + "\nDownsampling to make the balanced dataset for modelling\n" )
        df_input=df.copy()
        df_postive = df_input[df_input[feature]==1]
        df_negative=df_input[df_input[feature]==0]
        df_negative_downsample = resample(df_negative, replace = False, n_samples = df_postive.shape[0], random_state = 483)
        train_down = pd.concat([df_postive,df_negative_downsample]).sample(frac = 1)
        X_train_down = train_down.drop(columns =feature)
        y_train_down = train_down[feature]
        return (X_train_down, y_train_down)
    
    def featureselect (self,X_train_down, y_train_down ):
        ####################
        # 2, feature selections - get the most important features for future model use
        # **** Input: X_train_down, y_train_down
        # **** Output: dataframe with feature number and AUC score
        # ****         dataframe with features and rank
        print('*'*50 + "\nImportant Feature Selections\n" )
        feature_select_raw = SFS(xgb.XGBClassifier(objective = 'binary:logistic', eval_metric='mlogloss'),
                             k_features=(3,25), forward=False, floating=False, scoring = 'roc_auc', cv=3)
        feature_select_raw.fit(X_train_down, y_train_down)
        feature_select_dict = feature_select_raw.get_metric_dict()

        nb_features = list(feature_select_raw.get_metric_dict().keys())
        nb_features.sort()
        perf_dict = pd.DataFrame(columns = ['feature number','AUC'])
        count = 0
        for key in feature_select_dict.keys():
            auc = feature_select_dict[key]['avg_score']
            perf_dict.loc[count] = [key, auc]
            count+=1
    
        perf_dict[perf_dict['AUC'] == perf_dict['AUC'].max()]
        ## Save model
        plt.figure(figsize = (15,6))
        sns.lineplot(data = perf_dict, x = 'feature number', y = 'AUC')
        plt.show()
        return (perf_dict, feature_select_dict)

    def grid_search (self, X_train_best_features, y_train_down, cv = 3):
        ####################
        # 3, grid_search - get best hyper parameters
        #    **** Input: X_train_down[best features], y_train_down
        #    **** Output: best hyper parameters
        print('*'*50 + "\nget best hyper parameters\n" )
        clf = xgb.XGBClassifier(objective = 'binary:logistic', n_jobs = -1)
        parameters = {
             "eta"              : [0.05, 0.10, 0.15, 0.20] ,
             "max_depth"        : [ 3,5,7,9],
             "min_child_weight" : [ 1, 3, 5],
             "gamma"            : [ 0.0, 0.1, 0.2, 0.3],
             "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7]
    
             }

        grid = GridSearchCV(clf, parameters, scoring="roc_auc", cv = cv)
        start = time.time()
        grid.fit(X_train_best_features, y_train_down)
        best_model = grid.best_estimator_
        
        return (best_model)

    def model_evaluation (self, best_model,X_train_best_features, y_train_down, X_train_all_best_features, y_train, X_test_best_features, y_test, X_best_features, y):
        ####################
        # 4, model evaluation - evaluate model results
        # **** Input: best model,X_train_best_features, y_train_down, X_test_best_features, y_test
        # **** Output:  metrics_summary - accuracy, recall and f1
  
        best_model.fit(X_train_best_features, y_train_down)

        # Training down_sample Error
        y_train_pred = best_model.predict(X_train_best_features)
        y_train_proba = best_model.predict_proba(X_train_best_features)[:,1]
        train_acc, train_recall, train_f1 = acc(y_train_down,y_train_pred), recall(y_train_down,y_train_pred), \
                                                        f1(y_train_down,y_train_pred)
        train_fpr, train_tpr, train_thresholds = roc_curve(y_train_down, y_train_proba, pos_label=1)
        
        # Training all Error
        y_train_pred_all = best_model.predict(X_train_all_best_features)
        y_train_proba_all = best_model.predict_proba(X_train_all_best_features)[:,1]
        train_acc_all, train_recall_all, train_f1_all = acc(y_train,y_train_pred_all), recall(y_train,y_train_pred_all), \
                                                        f1(y_train,y_train_pred_all)
        train_fpr_all, train_tpr_all, train_thresholds_all = roc_curve(y_train, y_train_proba_all, pos_label=1)
        
        
        # Testing Error
        y_test_pred = best_model.predict(X_test_best_features)
        y_test_proba = best_model.predict_proba(X_test_best_features)[:,1]
        test_acc, test_recall, test_f1 = acc(y_test,y_test_pred), recall(y_test,y_test_pred), \
                                                    f1(y_test,y_test_pred)
        # all data get the probability
        y_pred = best_model.predict(X_best_features)
        y_proba = best_model.predict_proba(X_best_features)[:,1]
        
        ## Metrics Summary
        metrics_summary = pd.DataFrame()
        metrics_summary['Training down_sample'] = [train_acc,train_recall,train_f1]
        metrics_summary['Training'] = [train_acc_all,train_recall_all,train_f1_all]
        metrics_summary['Testing'] = [test_acc,test_recall,test_f1]
        metrics_summary.index = ['Accuracy','Recall','F1-Score']

        # Roc AUC
        test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
        plt.figure(figsize = (15,10))
        plt.plot(train_fpr, train_tpr, color = 'blue', label = 'Train')
        plt.plot(test_fpr, test_tpr, color = 'red', label = 'Test')
        plt.plot([0,1], [0,1], color = 'black')
        plt.xlabel('False Positive Rate', fontsize = 'x-large')
        plt.ylabel('True Positive Rate', fontsize = 'x-large')
        plt.title(f'ROC Curve for:', fontsize = 'x-large')
        plt.legend(fontsize = 'large')
        plt.show()

        
        return (metrics_summary, y_test_proba, y_train_proba_all,y_proba,y_pred)
    
    def feature_importances(self, X_train_down, X_train, y_train,X_test,y_test,  best_features, best_model,active_lines_base_FE, month):
        ################
        ## calcuate feature importances 
        ###Methods: 
        #### xgboost feature importance
        #### AUC
        #### Correlation
        
        
        feature_importances_df = pd.DataFrame()
        feature_importances_df['feature'] = X_train_down[best_features].columns
        feature_importances_df['Importance'] = best_model.feature_importances_
        feature_importances_df  = feature_importances_df.sort_values('Importance', ascending=False)
        feature_importances_df['ranking_importance'] = np.arange(1, feature_importances_df.shape[0]+1, 1)
        #feature_importances_df.sort_values(by = 'Importance', ascending = False).reset_index().drop(columns = 'index')
        
        ##AUC
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import roc_auc_score

        features_ALL = X_train[best_features].columns.tolist()
        features_ALL.sort()

        roc_auc      = []

        for fe in features_ALL:
            clf      = DecisionTreeClassifier()
            clf.fit(X_train[fe].to_frame(), y_train)
            pred_y   = clf.predict_proba(X_test[fe].to_frame())
            roc_auc.append(roc_auc_score(y_test, pred_y[:, 1]))
        #     print(f'{fe}: {roc_auc_score(test_y, pred_y[:, 1])}')

        result_tree  = pd.DataFrame({'feature': features_ALL, 'roc_auc': roc_auc})
        result_tree  = result_tree.sort_values('roc_auc', ascending=False)
        result_tree['ranking_tree'] = np.arange(1, result_tree.shape[0]+1, 1)
        
        ## Corr
        import sklearn.feature_selection as FS


        ### Correlation for All Features
        corr_train   =  X_train[best_features].apply(lambda x: x.corr(y_train))
        corr_test    = X_test[best_features].apply(lambda x: x.corr(y_test))
        result_corr  = pd.DataFrame({'corr_train': corr_train, 'corr_test': corr_test}).\
                       reset_index().rename(columns={'index': 'feature'})
        result_corr['corr_train_abs'] = result_corr.corr_train.abs()
        result_corr                   = result_corr.sort_values('corr_train_abs', ascending=False)
        result_corr['ranking_corr']   = np.arange(1, result_corr.shape[0]+1, 1)
        result_corr.drop('corr_train_abs', axis=1, inplace=True)
        # Create correlation matrix
        corr_all = active_lines_base_FE.corr()

        df_mutural_corr = corr_all.stack().reset_index()
        df_mutural_corr.columns = ['feature','feature2','correlation']
        df_mutural_corr_final=df_mutural_corr[df_mutural_corr['feature']!=df_mutural_corr['feature2']]
        df_mutural_corr_final['abs_corr']=df_mutural_corr_final['correlation'].abs()
        df_mutural_corr_final = df_mutural_corr_final.sort_values(by=['feature', 'abs_corr'],ascending=False)

        df_mutural_corr_final['rank']=df_mutural_corr_final.groupby(['feature']).cumcount()+1
        df_mutural_corr_final_=df_mutural_corr_final[df_mutural_corr_final['rank']<=5][['feature','feature2','correlation','rank']]
        df_mutural_corr_final_pivot1 =df_mutural_corr_final_.pivot(index='feature', columns='rank', values='feature2')\
                                .rename(columns={1:'feature_1',2:'feature_2',3:'feature_3',4:'feature_4',5:'feature_5'}).reset_index()
        df_mutural_corr_final_pivot2 =df_mutural_corr_final_.pivot(index='feature', columns='rank', values='correlation')\
                                .rename(columns={1:'corr_1',2:'corr_2',3:'corr_3',4:'corr_4',5:'corr_5'}).reset_index()
        df_mutural_corr_final_pivot=df_mutural_corr_final_pivot1.merge(df_mutural_corr_final_pivot2, on='feature',how='inner')
        df_mutural_corr_final_pivot=df_mutural_corr_final_pivot[['feature', 'feature_1','corr_1','feature_2','corr_2',
                                                         'feature_3',  'corr_3','feature_4','corr_4','feature_5','corr_5']]
        # Summary all metrics

        result_all    = feature_importances_df.merge(result_tree, how='left', on='feature').\
                                    merge(result_corr, how='left', on='feature').\
                                    merge(df_mutural_corr_final_pivot, how='left', on='feature')
        
        result_all.to_csv("fixed_churn_important_features_{}.csv".format(month), index=False)
        
        return (result_all)


    def lift_analysis(self,y_test, y_test_proba):
        ####################
        # 5, lift analysis - get lift curve for a df
        #    **** Input: y_test, y_test_proba
        #    **** Output: df with lift and also share of churn

        y_test_arr = np.array(y_test).reshape(-1,1)
        y_test_proba_arr = y_test_proba.reshape(-1,1)
        y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
        y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
        y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
        nb_obs = y_test_concat.shape[0] + 1
        intervals = np.linspace(0,nb_obs,11).astype('int')
        x_axis = [f'{k*10}% - {(k+1)*10}%' for k in range(10)]

        churn_rate_overall = y_test.mean()
        churn_ratio = []
        churners=[]
        #decile = []
        for ind in range(10):
            y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
            ## churn Rate
            churn = y_sub['True Label'].sum()/y_sub.shape[0]
            ratio = churn/churn_rate_overall
            churn_ratio += [ratio]
            churners+=[y_sub['True Label'].sum()]
        lift_df = pd.DataFrame()
        lift_df['Decile'] = x_axis
        lift_df['Lift'] = churn_ratio
        lift_df['share of churn']=churners
        return (lift_df)
    def lift_analysis_5(self,y_test, y_test_proba):
            ####################
            # 5, lift analysis - get lift curve for a df
            #    **** Input: y_test, y_test_proba
            #    **** Output: df with lift and also share of churn

            y_test_arr = np.array(y_test).reshape(-1,1)
            y_test_proba_arr = y_test_proba.reshape(-1,1)
            y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
            y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
            y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
            nb_obs = y_test_concat.shape[0] + 1
            intervals = np.linspace(0,nb_obs,21).astype('int')
            x_axis = [f'{k*5}% - {(k+1)*5}%' for k in range(20)]

            churn_rate_overall = y_test.mean()
            churn_ratio = []
            churners=[]
            #decile = []
            for ind in range(20):
                y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
                ## churn Rate
                churn = y_sub['True Label'].sum()/y_sub.shape[0]
                ratio = churn/churn_rate_overall
                churn_ratio += [ratio]
                churners+=[y_sub['True Label'].sum()]
            lift_df = pd.DataFrame()
            lift_df['Decile'] = x_axis
            lift_df['Lift'] = churn_ratio
            lift_df['share of churn']=churners
            return (lift_df)
    def lift_analysis_1(self,y_test, y_test_proba):
            ####################
            # 1, lift analysis - get lift curve for a df
            #    **** Input: y_test, y_test_proba
            #    **** Output: df with lift and also share of churn

            y_test_arr = np.array(y_test).reshape(-1,1)
            y_test_proba_arr = y_test_proba.reshape(-1,1)
            y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
            y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
            y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
            nb_obs = y_test_concat.shape[0] + 1
            intervals = np.linspace(0,nb_obs,101).astype('int')
            x_axis = [f'{k*1}% - {(k+1)*1}%' for k in range(100)]

            churn_rate_overall = y_test.mean()
            churn_ratio = []
            churners=[]
            #decile = []
            for ind in range(100):
                y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
                ## churn Rate
                churn = y_sub['True Label'].sum()/y_sub.shape[0]
                ratio = churn/churn_rate_overall
                churn_ratio += [ratio]
                churners+=[y_sub['True Label'].sum()]
            lift_df = pd.DataFrame()
            lift_df['Decile'] = x_axis
            lift_df['Lift'] = churn_ratio
            lift_df['share of churn']=churners
            return (lift_df)        


class profile_analyzing():
    
    '''
    Purpose 
    ------------------------------  

    Prepare model input and evaluation model output

    Funtions
    ---------------------------------
    1, profiling analysis
    
    '''
    def __init__(self):
        
        pass
    
  
    
    def profiling_analysis(self, y, y_pred,y_proba, active_lines_FE,df_predict_actual1, best_features,all_columns_CAT,all_columns_NUM, X_train,y_train):
        
        def discritizing_varialbe(X_train,profile_NUM, X, profile_CAT ):
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import roc_auc_score

            features_ALL = X_train[profile_NUM].columns.tolist()
            features_ALL.sort()
            profile_CAT=profile_CAT
            
            ### descritize the numerical data
            roc_auc      = []
            name=[]
            for fe in features_ALL:
                clf      = DecisionTreeClassifier(max_depth=2, min_samples_leaf =int(X_train.shape[0]*0.02) )
                clf.fit(X_train[fe].to_frame(), y_train)
                X[str(fe)+'_tree']=clf.predict_proba(X[fe].to_frame())[:,1] 
                name.append(str(fe)+'_tree')
            #print (name)
            X_=X[best_features+name].drop(profile_CAT, axis=1)
            X_['srcSubscriberId']=X['srcSubscriberId']
            return (X_, name)

        def descritized_df (X_,i):
            df=pd.concat( [X_.groupby([i+'_tree'])[i].min().reset_index().rename(columns={i:str(i)+'_min'}),
                    X_.groupby([i+'_tree'])[i].max().reset_index().rename(columns={i:str(i)+'_max'})], axis=1)
            df = df.loc[:,~df.columns.duplicated()]
            df[i+'_range']=df[str(i)+'_min'].astype('str')+'_'+df[str(i)+'_max'].astype('str')
            return df
        df_predict_actual=pd.DataFrame()
        #if (len(y)>0):
        if (y is not None):
            df_predict_actual['actual']=y
        df_predict_actual['predict']=y_pred
        df_predict_actual['predict_prob']=y_proba
        df_predict_actual['srcSubscriberId']=active_lines_FE['srcSubscriberId']
        df_mobile = df_predict_actual1.sort_values(by = 'predict_prob', ascending = False).reset_index()
        sample_size = df_predict_actual1.shape[0]
        decile_size_1 = sample_size/100

        df_mobile['Decile1'] = ((df_predict_actual1.index//decile_size_1)*1+1).astype('int64')

        decile_size_2 = sample_size/10

        df_mobile['Decile2'] = ((df_predict_actual1.index//decile_size_2)*10+10).astype('int64')

        decile_size_3 = sample_size/20

        df_mobile['Decile3'] = ((df_predict_actual1.index//decile_size_3)*5+5).astype('int64')
        df_profiling_ = pd.merge(active_lines_FE[best_features+['srcSubscriberId']], df_mobile, on ='srcSubscriberId', how='inner')
        all_columns_CAT=all_columns_CAT
        all_columns_NUM=all_columns_NUM
        profile_CAT=list(set(all_columns_CAT).intersection(set(best_features)))
        profile_NUM=list(set(all_columns_NUM).intersection(set(best_features)))
        [X_,name] = discritizing_varialbe(X_train,profile_NUM,active_lines_FE[best_features+['srcSubscriberId']] , profile_CAT )


        #NUM_final=pd.DataFrame()
        for i in profile_NUM:
            df= descritized_df(X_,i).reset_index()
            NUM_df = X_.merge(df, on=i+'_tree' , how='left')
            X_=NUM_df
            NUM_final=X_


  

        df_profiling_num = pd.merge(NUM_final, df_mobile, on ='srcSubscriberId', how='inner')
        range_=[]
        for i in profile_NUM:
            range_.append(i+"_range")
        df_profiling= df_profiling_.merge(df_profiling_num[['srcSubscriberId']+range_], on ='srcSubscriberId', how='inner')
        df_all_avg=pd.DataFrame()
        for i in profile_CAT+range_:
            df=pd.DataFrame(df_profiling[i].value_counts(normalize=True).reset_index().values, columns=['levels','proportion'])
            df['attributes']=i
            df_all_avg=df_all_avg.append(df[['attributes','levels','proportion']].rename(columns={'proportion':'avg_proportion'}))
        df_all_avg['key'] = df_all_avg['attributes']+df_all_avg['levels'].astype('str')
        #df.to_csv('profiling_cat_aug.csv',index=False,mode='a',header=False )
        #if  (len(y)>0):
        if  (y is not None):
            df_all_actual=pd.DataFrame()
            for i in profile_CAT+range_:
                df=pd.DataFrame(df_profiling[df_profiling['actual']==1][i].value_counts(normalize=True).reset_index().values, columns=['levels','proportion'])
                df['attributes']=i
                df_all_actual=df_all_actual.append(df[['attributes','levels','proportion']].rename(columns={'proportion':'actual_proportion'}))
            df_all_actual['key'] = df_all_actual['attributes']+df_all_actual['levels'].astype('str')
                #df.to_csv('profiling_cat_aug.csv',index=False,mode='a',header=False )
        df_all_percentile_1_20=pd.DataFrame()
        for j in range(0,21,1):    
            for i in profile_CAT+range_:
                df=pd.DataFrame(df_profiling[df_profiling['Decile1']==j][i].value_counts(normalize=True).reset_index().values, columns=['levels','proportion'])
                df['Decile1']=j
                df['attributes']=i
                df_all_percentile_1_20=df_all_percentile_1_20.append(df[['Decile1','attributes','levels','proportion']].rename(columns={'proportion':'decile_proportion'}))
        df_all_percentile_1_20['key'] = df_all_percentile_1_20['attributes']+df_all_percentile_1_20['levels'].astype('str')
        df_all_percentile_1_20_=df_all_percentile_1_20.pivot(index='key', columns='Decile1', values='decile_proportion').reset_index()
        df_all_decile_5_20=pd.DataFrame()
        for j in range(0,21,5):
            for i in profile_CAT+range_:
                df=pd.DataFrame(df_profiling[df_profiling['Decile3']==j][i].value_counts(normalize=True).reset_index().values, columns=['levels','proportion'])
                df['Decile3']=j
                df['attributes']=i
                df_all_decile_5_20=df_all_decile_5_20.append(df[['Decile3','attributes','levels','proportion']].rename(columns={'proportion':'decile_proportion'}))
        df_all_decile_5_20['key'] = df_all_decile_5_20['attributes']+df_all_decile_5_20['levels'].astype('str')
        df_all_decile_5_20_=df_all_decile_5_20.pivot(index='key', columns='Decile3', values='decile_proportion').reset_index()
        df_all_decile_30_100=pd.DataFrame()
        for j in range(30,101,10):
            for i in profile_CAT+range_:
                df=pd.DataFrame(df_profiling[df_profiling['Decile2']==j][i].value_counts(normalize=True).reset_index().values, columns=['levels','proportion'])
                df['Decile2']=j
                df['attributes']=i
                df_all_decile_30_100=df_all_decile_30_100.append(df[['Decile2','attributes','levels','proportion']].rename(columns={'proportion':'decile_proportion'}))
        df_all_decile_30_100['key'] = df_all_decile_30_100['attributes']+df_all_decile_30_100['levels'].astype('str')
        df_all_decile_30_100=df_all_decile_30_100.pivot(index='key', columns='Decile2', values='decile_proportion').reset_index()
        #if  (len(y)>0):
        if  (y is not None):
            df_all = df_all_avg.merge(df_all_actual[['key','actual_proportion']], on='key', how='left')\
                 .merge(df_all_percentile_1_20_, on='key', how='left')\
                 .merge(df_all_decile_5_20_, on='key', how='left')\
                 .merge(df_all_decile_30_100, on='key', how='left')
        else:
            df_all = df_all_avg.merge(df_all_percentile_1_20_, on='key', how='left')\
                 .merge(df_all_decile_5_20_, on='key', how='left')\
                 .merge(df_all_decile_30_100, on='key', how='left')
        return (df_all, NUM_final,df_profiling)
