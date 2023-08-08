#!/bin/bash

#
# Submit chainsaw job array to SGE cluster
#
# usage:
#   qsub -t 1-16933 chainsaw_submit.ucl_cs.sh
#
# Each job will expect to find an equivalent file containing a list of zipfiles,
# created by:
#
#   split --suffix-length=8 --numeric-suffixes=1 --lines=1000 ../all_proteome_zipfiles.txt zipfiles.
#


if [[ "$SGE_TASK_ID" == "undefined" ]]; then
	echo "Need to specify the job array details in the qsub command"
	echo "example: qsub -t 1-16933:1 <script>"
	echo
	exit 1
fi

CATH_VERSION=v4_3_0
HOME_DIR=/home/$USER
SHARED_REPO="${SHARED_REPO:-${SGE_O_WORKDIR}/chainsaw}"
SCRATCH_DIR=/scratch0/$USER
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
PYTHON_EXE=${SGE_O_WORKDIR}/venv/bin/python3
APPEND_FLAG="--append"

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
ZIPFILES_LIST_FILE=${SGE_O_WORKDIR}/data/zipfiles.`printf "%08d" $SGE_TASK_ID`
RESULTS_FILE=${SGE_O_WORKDIR}/results/$(basename $ZIPFILES_LIST_FILE).results.csv

if [ ! -e "${PYTHON_EXE}" ]
then
    echo "ERROR: python executable does not exist - have you built a venv? (PYTHON_EXE=$PYTHON_EXE)"
    exit 1
fi

if [ ! -e "${ZIPFILES_LIST_FILE}" ]
then
    echo "ERROR: input list file '${ZIPFILES_LIST_FILE}' does not exist"
    exit 1
fi

if [ -e "${RESULTS_FILE}" && "${APPEND_FLAG}" != "" ]
then
    echo "ERROR: output file '${RESULTS_FILE}' already exists (will not overwrite)"
    exit 1
fi

echo "ZIPFILES_LIST_FILE : $ZIPFILES_LIST_FILE"
echo "RESULTS_FILE       : $RESULTS_FILE"
echo "DATE_STARTED       : "`date`
echo
echo "HOSTNAME            : $HOSTNAME"
echo "SGE_TASK_ID         : $SGE_TASK_ID"
echo "GIT_REMOTE          : $SHARED_REPO"
echo "LOCAL_TASK_DIR      : $LOCAL_TASK_DIR"

# die on errors
set -e

echo "Creating output directories ..."
mkdir -p $LOCAL_TASK_DIR
mkdir -p $(dirname $RESULTS_FILE)
echo "...DONE"

echo "Loading python ..."
source /share/apps/source_files/python/python-3.8.5.source
echo "...DONE"

echo "Extracting PDBs ... "
echo $(date)
cd $LOCAL_TASK_DIR
LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
mkdir -p $LOCAL_PDB_DIR
cat $ZIPFILES_LIST_FILE | xargs -I XXX unzip -d $LOCAL_PDB_DIR XXX
echo $(date)
echo "...DONE "

echo "Running chainsaw ..."
echo $(date)
$PYTHON_EXE $SHARED_REPO/get_predictions.py --structure_directory $LOCAL_PDB_DIR $APPEND_FLAG -o $RESULTS_FILE && rc=$? || rc=$?
echo $(date)
echo "EXIT_CODE: $rc"
echo "...DONE"

echo "Removing local temp dir ..."
rm -rf $LOCAL_TASK_DIR
echo "...DONE"

echo "DATE_FINISHED   : " $(date)
if [ "$rc" == "0" ]
then
    echo "JOB_COMPLETE"
elif [ "$rc" == "1" ]
then
    echo "JOB_COMPLETE (ERROR)"
fi
