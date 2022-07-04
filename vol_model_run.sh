#!/bin/sh
#Create the run parameters for the model
DAY=$(date +%d)
MONTH=$(date +%m)
YEAR=$(date +%Y)
#RUNMONTH=$(($(date +%Y%m)-1))
RUNMONTH=$2
TRAINMONTH=$1
SEGMENT=$3
TS=$(date +%Y%m%d_%H%M%S)
RUNDIR="vol_rundir_$TS"
echo "Infer run on data from: $RUNMONTH"



#mkdir $RUNDIR

#Get the latest artifact from the repository
gsutil cp gs://alticeusa-am/code_repo/Voluntary/dev_code.zip /home/jupyter/$RUNDIR/dev_code.zip

#Unzip the data
unzip /home/jupyter/$RUNDIR/dev_code.zip -d /home/jupyter/$RUNDIR

#Change directory to the run dir
cd /home/jupyter/$RUNDIR/dev_code

#Copy the log of the run to the bucket
cleanup () {
gsutil cp *.log gs://alticeusa-am/model_run_logs/

cd /home/jupyter

#Delete the temp directory
rm -rf /home/jupyter/$RUNDIR

echo "status code: $1"

}

#Run the code in the dev_code

trap 'cleanup $?' EXIT


if python3 voluntary_prediction_run.py $TRAINMONTH $RUNMONTH; then
	exit 0
else
	exit $?
fi