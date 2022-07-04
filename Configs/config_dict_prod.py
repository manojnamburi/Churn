# -*- coding: utf-8 -*-
"""

Created on Fri Mar 29 17:08:10 2019



@author: SYADALAM

"""
##Configuration dictionary file for db connections
config_dict = {}


#GBQ params
config_dict['BUCKET'] = 'atus-am-prod-am'
config_dict['PROJECT'] = 'atus-am-prod'
config_dict['credentials_file'] = '/home/saitarun_yadalam_alticeusa_com/auth_files/atus-am-prod-8af5ff57841f.json'
config_dict['dataset'] = 'am_churn'
config_dict['churn_table'] = 'churn'
config_dict['customer_info_table'] = 'v_d_customer_info_history'
config_dict['custmaster_table'] = 'v_d_churn_custmaster'
##Adding parameters for Voluntary
config_dict['vol_churn_view'] = 'v_vol_churn_with_new_tier_map'
##Location of the voluntary artifacts (Specified relative to the bucket)
config_dict['voluntary_obj_folder'] = 'voluntary_production_artifacts'
##Adding the training data table for record keeping
config_dict['train_source'] = 'poc.voluntary_train_data_202007'