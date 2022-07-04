import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import os
import sys
from datetime import datetime
import collections
import pandas_gbq as gbq
#import Class_GenericUtilities

class GeneralUtilitiesNonTF:
    def __init__(self,project_id,dataset,bucket_name,json_path=None,auth_link=None):
        ##Dont pass any params other than project_id to use the default credentials
        ##Pass the json path when using in windows local machine
        self.json_path = json_path
        self.project_id = project_id
        self.bucket = bucket_name
        self.dataset = dataset
        if json_path != None:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_path
        self.bq_client = bigquery.Client(project = self.project_id)
        self.gcs_client = storage.Client(project= self.project_id)
   





    ##Use the bq client to shard the data to storage and get the information schema details after sharding.
    ##Specify the destination folder relative to the gs://bucket path.
    ##Possible file formats:
    ##csv,json
    ##compression is used to compress. Default compression type is N. If yes, GZIP will be used
    ##Specify wildcard in the filename for multiple files
    ##Reference to the extract job params: 
    #https://googleapis.github.io/google-cloud-python/latest/bigquery/generated/google.cloud.bigquery.job.ExtractJobConfig.html
    def bq_to_gcs(self,sql_query,file_name,destination_folder,file_format ='CSV' \
                  ,compression = False):
        ##Initiate the bq client
        bq_client = self.bq_client
        
        ######Create a temp table
        query_job_config = bigquery.QueryJobConfig()
        
        
        dataset_ref = bq_client.dataset(self.dataset,project=self.project_id)
        
        #Generating temp table name 
        ts = datetime.strftime(datetime.now(),'%m_%d_%y_%H_%M_%S')
        table_id = 'temp_gcs_{}'.format(ts)
        print('Temp table to be created: '+table_id)
        table_ref = dataset_ref.table(table_id)
        
        ##Write the query to the temp table to get the schema of the query results
        query_job_config.destination = table_ref
        
        
        
        query_job = bq_client.query(sql_query,job_config = query_job_config)
        query_job.result() ##Wait for the query to complete
        
        print('*'*10+'\nQuery executed: ')
        print(sql_query)
        
        print('Temp table ' + table_id + ' created')
        
        ##Get the schema details from the information.schema. This will help in creating a schema for read csv
        info_query = str("""SELECT table_name,column_name,data_type,is_nullable FROM {}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='{}'""".format(self.dataset,table_id))
        
        print('*'*10+"\nInfo query: "+info_query)
        
        #Execute the information query
        info_df = bq_client.query(info_query).to_dataframe()
       
        #print("Printing info df columns") 
        #print(info_df)
        
        ###Sharding starts now.
        extract_job_config = bigquery.ExtractJobConfig()
        compression_format = 'gzip'
        file_format = str.upper(file_format)
        ##Check if valid file format is specified. Currently accepts only csv and json. If wrong, defaults to csv
        if ('CSV' not in file_format) and ('JSON' not in file_format):
            print(file_format)
            print('Wrong type specified. Defaulting to csv')
            file_format = 'CSV'
        if ('JSON' in file_format):
            file_format = 'NEWLINE_DELIMITED_JSON'
            
        extract_job_config.destination_format = file_format
        

        ##Compress the files in case compression is specified to be true. Currently does gzip compression
        if compression:
            file_extension = str.lower(file_format)+'.'+compression_format
            extract_job_config.compression = str.upper(compression_format)
        elif ('JSON' in file_format):
            file_extension = 'json'
        else:
            file_extension = 'csv'
        file_name = file_name + '.' + file_extension
        
        destination_uri = "gs://{}/{}".format(self.bucket + '/' + destination_folder, file_name)
        
        
        ##Execute the extract job
        extract_job = bq_client.extract_table(\
        source = table_ref,\
        destination_uris= destination_uri,\
        job_config = extract_job_config\
        )
        extract_job.result() #Execute and wait for the job to finish
        print('Extraction to the '+destination_folder+' complete')
        
        bq_client.delete_table(table_ref)
        return info_df
    
    ##Method to read in the files from the storage into pandas df
    ##Acceptable file formats = csv and json
    ##Specify the info df for using the dtype attribute of read_csv
    def gcs_to_df(self,file_path,file_name,file_format='csv',delimiter = '/',\
                  compression = False,info_df=None):
        #Instantiate the mapping to create a dict for reading in the columns
        sql_to_dtype = collections.OrderedDict()
       
        sql_to_dtype['INT64']                         = 'float64'
        sql_to_dtype['FLOAT64']                       = 'float64'
        sql_to_dtype['BOOL']                          = 'bool'
        sql_to_dtype['STRING']                         = 'str'
        sql_to_dtype['DATE']                            = 'datetime64[ns]'
        sql_to_dtype['TIMESTAMP']                       = 'datetime64[ns]'
        sql_to_dtype['NUMERIC']                        = 'float64'
        sql_to_dtype['DATETIME']                       = 'datetime64[ns]'

        #print(info_df.dtypes)

        if info_df is None:
            col_dtype_mapping = None
        ##Create the dtype mapping using the ordered dict
        else:
            col_dtype_mapping = dict(zip(info_df['column_name'],[sql_to_dtype[dcol] for dcol in info_df['data_type']])) 
        ##Get list of datetime columns
        dt_cols = [k for (k,v) in col_dtype_mapping.items() if v == 'datetime64[ns]']
        if dt_cols == []:
            dt_cols = False
        else:
            for col in dt_cols:
                del col_dtype_mapping[col]
        
        ##Initiate the storage client
        gcs_client = self.gcs_client
        
        ##Create the prefix by combining the name of the file and path to the file
        prefix = file_path + '/'+ file_name
        print('Prefix: ' + prefix)
        #Get the bucket object
        gcs_bucket = gcs_client.get_bucket(self.bucket)
        
        #Get the blobs that match the pattern
        blobs = gcs_bucket.list_blobs(prefix = prefix,delimiter = delimiter)
        ##Initialize the final dataframe that would be returned from this method
        final_df = pd.DataFrame()
        
        if compression:
                file_extension = file_format + '.' + 'gzip'
        
        files_to_be_read = [blob.name for blob in blobs if file_format in blob.name]
        
        for file in files_to_be_read:
            print('Pulling file: '+ file)
            tempblob = gcs_bucket.blob(file)
            temp_file_name = 'temp_'+ datetime.strftime(datetime.now(),'%m_%d_%y_%H_%M_%S')+'.'+file_extension
            tempblob.download_to_filename(temp_file_name)
            ##Not perfect as pandas doesnt support nulls in the integer and float type columns and in columns like homeown which has 'null' as a value, they will be read in as nan. TBD if other format can be used.
            if file_format == 'csv' and compression == False:
                temp_df = pd.read_csv(temp_file_name,dtype = col_dtype_mapping, keep_default_na = True, parse_dates = dt_cols, date_parser = pd.to_datetime)
            
            elif file_format == 'csv' and compression == True:
                temp_df = pd.read_csv(temp_file_name,compression = 'gzip', dtype = col_dtype_mapping, keep_default_na = True,parse_dates = dt_cols, date_parser = pd.to_datetime)
            elif file_format == 'json' and compression == False:
                temp_df = pd.read_json(temp_file_name)
            elif file_format == 'json' and compression == True:
                temp_df = pd.read_json(temp_file_name, compression = 'gzip')
            final_df = final_df.append(temp_df)
            os.remove(temp_file_name)
          
        return final_df
    
    def read_gbq(self,sql_query):
        ##Shard the data into a temp folder, read from it and delete the temp folder
        ts = datetime.strftime(datetime.now(),'%m_%d_%y_%H_%M_%S')
        temp_file_name = 'tmp_file_'+ts
        temp_folder = 'tmp_folder_'+ts
        info_df = self.bq_to_gcs(sql_query,temp_file_name+'-*',temp_folder,'CSV',True)
        ##Read the sharded data to a df
        final_df = self.gcs_to_df(temp_folder,temp_file_name,'csv','/',True,info_df)
        ##Initiate the storage client
        gcs_client = self.gcs_client
        gcs_bucket = gcs_client.get_bucket(self.bucket)
        blobs = gcs_bucket.list_blobs(prefix = temp_folder+"/", delimiter = "/")
        for blob in blobs:
            blob.delete()
        print('Deleted the temp folder '+ temp_folder)
        
        return final_df
   
    ##For ad-hoc queries use the read_gbq lite which leverages pandas_gbq read_gbq.
    def read_gbq_lite(self,sql_query):
        rs = None
        try:
            rs = gbq.read_gbq(sql_query,project_id=self.project_id,dialect='standard')
            print("*"*10+'\nSuccessfully read in the data using the pandas_gbq module')
        except Exception as e:
            print("*"*10+'\nException occurred reading the query: '+str(e))
        return rs




    
    
      ##Module to write a df to BQ and storage

    def df_to_gcp(self, source_df, key, destination_table, drop = False):
        
        final_table = self.dataset + "." + destination_table
        
        ##############################
        #Create a map of Pandas data types to BQ data types 
        ##############################
        dtype_to_sql = collections.OrderedDict()
        dtype_to_sql['int64']                           = 'INT64'
        dtype_to_sql['int32']                           = 'INT64'
        dtype_to_sql['int16']                           = 'INT64'
        dtype_to_sql['int8']                            = 'INT64'
        dtype_to_sql['uint8']                           = 'INT64'
        dtype_to_sql['uint16']                          = 'INT64'
        dtype_to_sql['uint32']                          = 'INT64'
        dtype_to_sql['uint64']                          = 'INT64'
        dtype_to_sql['float32']                         = 'FLOAT64'
        dtype_to_sql['float64']                         = 'FLOAT64'
        dtype_to_sql['bool']                            = 'BOOL'
        dtype_to_sql['object']                          = 'STRING'
        dtype_to_sql['datetime64[ns]']                  = 'TIMESTAMP'
        
        
        time_cols = list(source_df.columns[source_df.dtypes == 'datetime64[ns]'])
        for col in time_cols:
            source_df[col] = pd.to_datetime(source_df[col],format = '%Y-%m-%d %H:%M:%S')
        
        ts = datetime.strftime(datetime.now(),'%m_%d_%y_%H_%M_%S')
        temp_file_name = 'temp_file_'+ ts + '.csv' 
        source_df.to_csv(temp_file_name,index=False)
        
        gcs_bucket = self.gcs_client.get_bucket(self.bucket)
        blob = gcs_bucket.blob(key)
        blob.upload_from_filename(temp_file_name)
        
        print('*'*10+"\nSuccessfully uploaded data to: "+key+" in bucket "+self.bucket)
        
        try:
            os.remove(temp_file_name)
        except Exception as e:
            print('Couldnt delete {} temp file due to an exception {}'.format(temp_file_name,str(e)))
        
        dataset_ref = self.bq_client.dataset(self.dataset)
        job_config = bigquery.LoadJobConfig()
        if drop:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        else:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        
        #Auto interpret schema. 
        #Reference: https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv#limitations
        #job_config.autodetect = True
        
        ##Generate the bigquery schema field to map the pandas dtype to BigQuery. Pattern of the field: bigquery.SchemaField(column, dtype)
        ##List of bigquery.SchemaField method using the dtype_to_sql map to convert the dtype of col to BigQuery data type
        job_config.schema = [bigquery.SchemaField(str(col),dtype_to_sql[source_df[[col]].dtypes.astype(str)[0]])for col in source_df.columns]
        
        #Specify the source type to be csv and skip the header row
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1
        
        destination_uri = "gs://{}/{}".format(self.bucket,key)
        load_job = self.bq_client.load_table_from_uri(destination_uri,dataset_ref.table(destination_table),job_config = job_config)
        try:
            load_job.result() #Waits for the job to finish
            print('*'*10+"\nSuccessfully uploaded data to: "+destination_table+" in dataset "+self.dataset)
        except:
            print('Exception occurred. Errors: '+ load_job.errors)
        return

    