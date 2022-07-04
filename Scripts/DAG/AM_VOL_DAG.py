from datetime import datetime, timedelta

from airflow.models import DAG, Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.hooks.base_hook import BaseHook
import tempfile

from libs.alerts.functions import _get_alert_params_internal
from libs.alerts.snmp import send_alert

def failure_notification(context):
    """This is a function that will run within the DAG execution"""
    # Originating System Name
    params = _get_alert_params_internal(context)
    print(params)
    alerting_component = 'DAG ID: {}'.format(params['dag_id'])
    alert_params = [params['originating_system'],
                    params['source_ip'],
                    params['dest_ip'],
                    params['run_id'],
                    params['dag_id'],
                    params['task_id'],
                    params['start_date'],
                    params['execution_date'],
                    params['try_number'],
                    params['state']]
    if int(Variable.get('AM_VOL_SEND_ALERT'))==1:
        send_alert(alerting_component, alert_params)
        print("Alert sent to the OpenNMS system.")
    else:
        print('*'*10+"\nAlert trigger disabled.")


default_args = {
    'owner': 'ATUS-AM',
    'start_date': datetime(2019, 7, 8, 0, 0, 0),
    'email_on_failure': False,
    'email_on_retry': False,
    'depends_on_past': True,
    'wait_for_downstream': True,
    'queue': 'default',
    'on_failure_callback': failure_notification
}

# Dag definition
dag = DAG('AM_VOL_PREDICTION',
          description='Run Voluntary Prediction model',
          default_args=default_args,
          schedule_interval="00 08 11 * *",
          catchup=False,
          max_active_runs=1,
          dagrun_timeout=timedelta(hours=12),
          on_failure_callback=failure_notification
          )

instance_name = Variable.get('AM_COMPUTE_INSTANCE')
project = Variable.get('AM_PROJECT_NAME')
connection_id = 'am_gcp_credentials'

##Use the basehook to get the credentials from the ConnectionsDB
connection = BaseHook.get_connection(connection_id)
hook = connection.get_hook()
credentials = hook.extras['extra__google_cloud_platform__keyfile_dict']
#print('*'*10+"\nCredentials fetched: "+credentials)
#create temporary credentials file for authorization
f = tempfile.NamedTemporaryFile(prefix='atus_am_', suffix='.json')
f.write(credentials)
f.flush()
tmp_file = f.name 

##Get the run mode argument from the composer for automated or manual type run
##1 - Automated: Date generated based on system info
##2 - Manual: Set the variable 'AM_VOL_PREDICTION_RUN_DATE' param for manually running the code

run_mode = Variable.get('AM_VOL_PREDICTION_RUN_MODE')


if int(run_mode) == 1:
    ##Generate the run date on the fly to be taken in as the argument to the bash script
    if datetime.now().month==1:
        run_date = int(datetime.strftime(datetime.now(),format='%Y%m')) - 100 + 11 ##Only at the start of the year, get the previous year and last month as run_month
    else:
        run_date = int(datetime.strftime(datetime.now(),format='%Y%m')) - 1
else:
    run_date = Variable.get('AM_VOL_PREDICTION_RUN_DT')

##Get the train date for the model
train_date = Variable.get('AM_VOL_PREDICTION_TRAIN_DT')

##Get the segments being modelled. Currently hard coded to use OPTION3 - combination of 6 segments
#seg_list = Variable.get('AM_VOL_PREDICTION_SEGMENTS')

print('*'*100)
print('*'*10+"\nExecuting VOL run for: {} using {} as the train month".format(str(run_date),str(train_date)))
print('*'*100)




bash_command = """
gcloud auth activate-service-account --key-file={jsonkey};
gcloud beta compute --project {project} ssh {instance} --zone us-east1-b --internal-ip --command "sh /home/saitarun_yadalam_alticeusa_com/vol_model_run.sh {traindt} {rundt};"
""".format(project = project,jsonkey = tmp_file,instance = instance_name, traindt = train_date, rundt = run_date)


final_command = bash_command

'''
Order of execution of the steps in the bash script:
    1. Create a temperory run directory
    2. Get the latest code from repo
    3. Change the directory to the run directory
    4. Execute the code
    5. Clean up the directory and move the log files to the storage
'''

dummy_start = DummyOperator(task_id='START',
                            dag=dag)

task1 = BashOperator(task_id='run_vol_prediction_code',
                       bash_command=final_command,
                       dag=dag
                    )

dummy_end = DummyOperator(task_id='END',
                          dag=dag)

dummy_start >> task1 >> dummy_end