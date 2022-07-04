from Class_GeneralUtilitiesNonTF import GeneralUtilitiesNonTF
from voluntary_prediction_segment_dictionary import voluntary_seg_dict
import voluntary_prediction_helpers as vph
import sys
import pandas as pd

def create_voluntary_prediction_data(bucket,project_id,dataset,auth_file,segment,view,sql_dt,run_type = 'Infer',atts=[]):
    
    ##Instantiate a default dataframe for df_data
    df_data = pd.DataFrame()
    
    #Instantiate the util obj that will be used to interact with BQ
    util_obj = GeneralUtilitiesNonTF(project_id = project_id,dataset = dataset,bucket_name = bucket,json_path = auth_file)
    
    ##Prepare the train data for each segment in the segment list
    ##Get the 6 months string for the  data
    list_6m = vph.get_6m_list(sql_dt)
    ##Subset of the list to get the next 3 months
    list_3m = list_6m[0:3]
    
    if (atts != '*') & (atts != []):
        query_vars = str('chc_id,cust,') + str(",".join(atts))
    elif atts == '*':
        query_vars = '*'
    elif atts == []:
        query_vars = str('chc_id,cust')
    
    if run_type == 'Train':
    ##Prepare the sql to read in the train data
    ##Adding IF NULL condition to pending drform on 08/27/2019 due to change in logic of pending drform
        input_sql = """
    SELECT
        1 as status,
       {atts}    
    FROM    
        {dataset}.{view} 

    WHERE 
        dt = '{traindt}'
        AND drform IN ('ACTIVE')
        AND products IN ('2: Video/OOL','3: Video/OOL/OV')  
        and IFNULL(pending_drform,'XX') not in ('VOLUNTARY', 'MOVE AND TRANSFER')-- To get rid of customers who are already in pending status except pending NPD
        AND chc_id IN  
            (SELECT DISTINCT chc_id FROM
                (
                    SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (vol_pend_disco = 'Y' or pending_mover = 'FALSE') AND dt = '{month1}'
                        UNION DISTINCT
                    SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (vol_pend_disco = 'Y' or pending_mover = 'FALSE') AND dt = '{month2}'
                        UNION DISTINCT
                    SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (vol_pend_disco = 'Y' or pending_mover = 'FALSE') AND dt = '{month3}'
                )
            )  
            AND --Segment Definition 
            {seg_def}
            
    UNION DISTINCT

    SELECT 
        0 as status,
        {atts}    
    FROM    
        {dataset}.{view}
        
    WHERE 
        dt = '{traindt}'
        AND drform IN ('ACTIVE')
        AND products IN ('2: Video/OOL','3: Video/OOL/OV')                           
        AND IFNULL(pending_drform,'XX') not in ('VOLUNTARY', 'MOVE AND TRANSFER')
        AND chc_id IN (SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (ifnull(vol_pend_disco,'XX') <> 'Y' and ifnull(pending_mover,'XX') <> 'FALSE') AND drform IN ('ACTIVE') AND dt = '{month1}')
        AND chc_id IN (SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (ifnull(vol_pend_disco,'XX') <> 'Y' and ifnull(pending_mover,'XX') <> 'FALSE') AND drform IN ('ACTIVE') AND dt = '{month2}')
        AND chc_id IN (SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (ifnull(vol_pend_disco,'XX') <> 'Y' and ifnull(pending_mover,'XX') <> 'FALSE') AND drform IN ('ACTIVE') AND dt = '{month3}')
        AND --Segment_Definition 
        {seg_def}
    """.format(dataset = dataset,view = view,seg_def=voluntary_seg_dict[segment],traindt=str(sql_dt),month1=list_3m[0],month2=list_3m[1],month3=list_3m[2], atts = query_vars)
        
        
    
    
    ##Added IF NULL condition to pending drform on 08/27/2019 due to change in the logic of pending drform
    ##Get the prediction data ready based on the type of run 
    elif run_type == 'Infer':
        input_sql = """
SELECT
    {atts}

FROM    
    {dataset}.{view}

WHERE 
    dt = '{rundt}'
    AND products IN ('2: Video/OOL','3: Video/OOL/OV')  
    --Added additional filters similar to train
    AND drform IN ('ACTIVE')
    AND ifnull(pending_drform,'XX') not in ('VOLUNTARY', 'MOVE AND TRANSFER')
    --Segment definition
    AND {seg_def}
""".format(dataset = dataset, view = view, rundt = str(sql_dt), seg_def = voluntary_seg_dict[segment], atts = query_vars)
    
    elif run_type == 'Test':
        
        
        input_sql = """
SELECT 
    1 as status,
    {atts}  

FROM    
    {dataset}.{view}

WHERE 
    dt = '{testdt}'
    AND drform IN ('ACTIVE') -- Add to infer
    AND products IN ('2: Video/OOL','3: Video/OOL/OV')                           
    AND IFNULL(pending_drform,'XX') not in ('VOLUNTARY', 'MOVE AND TRANSFER')  -- Add to infer
    AND chc_id IN  
        (SELECT DISTINCT chc_id FROM
            (
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month1}'
                    UNION DISTINCT
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month2}'
                    UNION DISTINCT
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month3}'
            )
        )
    --Segment definition
    AND {seg_def}

UNION DISTINCT

SELECT 
    0 as status, 
   {atts} 

FROM    
    {dataset}.{view}
    
WHERE 
    dt = '{testdt}'
    AND drform IN ('ACTIVE')
    AND products IN ('2: Video/OOL','3: Video/OOL/OV')                           
    AND IFNULL(pending_drform,'XX') not in ('VOLUNTARY', 'MOVE AND TRANSFER')
    AND chc_id NOT IN  
        (SELECT DISTINCT chc_id FROM
            (
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month1}'
                    UNION DISTINCT
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month2}'
                    UNION DISTINCT
                SELECT DISTINCT chc_id FROM {dataset}.{view} WHERE (pending_drform = 'VOLUNTARY' or pending_mover = 'FALSE') AND dt = '{month3}'
            )
        )
    AND {seg_def}
""".format(dataset = dataset, view = view, testdt = str(sql_dt), month1 = list_3m[0] , month2 = list_3m[1], month3 = list_3m[2], seg_def = voluntary_seg_dict[segment], atts = query_vars)
    else:
        vph.log('*'*10+"\nImproper run type specified")
        sys.exit(4)
    
    try:
        df_data = util_obj.read_gbq(input_sql)
    except Exception as e:
        vph.log('*'*10+"\nAn exception has occurred in reading in the {atts} data: ".format(atts = run_type)+ str(e))
        
        
    return df_data
        
    
        
    
    