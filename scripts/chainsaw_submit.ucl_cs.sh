#!/bin/bash

#
# Submit chainsaw job array to SGE cluster
# 
# usage:
#   qsub -t 1-16933 chainsaw_submit.ucl_cs.sh
#
# Assumes index file has been split into chunks e.g. "zipindex.00000001":
#   split -a 8 -d 1 -l 1000 data/zipindex zipindex.
#


if [[ "$SGE_TASK_ID" == "undefined" ]]; then
	echo "Need to specify the job array details in the qsub command"
	echo "example: qsub -t 1-16933:1 <script>"
	echo
	exit 1
fi

CATH_VERSION=v4_3_0
ZIP_DIR=/SAN/bioinf/afdb_domain/zipfiles
HOME_DIR=/home/$USER
SHARED_REPO="${SHARED_REPO:-${SGE_O_WORKDIR}/chainsaw}"
ZIP_EXTRACT=$SHARED_REPO/scripts/zip_extract.py
SCRATCH_DIR=/scratch0/$USER
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
PYTHON_EXE=${SGE_O_WORKDIR}/venv/bin/python3

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
ZIP_INDEX_FILE=${SGE_O_WORKDIR}/data/zip_index.`printf "%08d" $SGE_TASK_ID`
RESULTS_FILE=${SGE_O_WORKDIR}/results/$(basename $ZIP_INDEX_FILE).results.csv

if [ ! -e "${PYTHON_EXE}" ]
then
    echo "ERROR: python executable does not exist - have you built a venv? (PYTHON_EXE=$PYTHON_EXE)"
    exit 1
fi

if [ ! -e "${ZIP_INDEX_FILE}" ]
then
    echo "ERROR: input index file '${ZIP_INDEX_FILE}' does not exist"
    exit 1
fi

if [ -e "${RESULTS_FILE}" ]
then
    echo "ERROR: output file '${RESULTS_FILE}' already exists (will not overwrite)"
    exit 1
fi

echo "INDEX_FILE         : $ZIP_INDEX_FILE"
echo "RESULTS_FILE       : $RESULTS_FILE"
echo "DATE_STARTED       : "`date`
echo
echo "HOSTNAME            : $HOSTNAME"
echo "SGE_TASK_ID         : $SGE_TASK_ID"
echo "GIT_REMOTE          : $SHARED_REPO"
echo "LOCAL_TASK_DIR      : $LOCAL_TASK_DIR"

# die on error
set -o errexit

echo "Creating output directories ..."
mkdir -p $LOCAL_TASK_DIR
mkdir -p $(dirname $RESULTS_FILE)
echo "...DONE"

echo "Loading python ..."
module load python/3.8.5
echo "...DONE"

echo "Extracting PDBs ..."
cd $LOCAL_TASK_DIR
LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
mkdir -p $LOCAL_PDB_DIR
$ZIP_EXTRACT -i $ZIP_INDEX_FILE -z $ZIP_DIR -o $LOCAL_PDB_DIR 
echo "...DONE"

echo "Running chainsaw ..."
$PYTHON_EXE $SHARED_REPO/get_predictions.py --structure_directory $LOCAL_PDB_DIR > $RESULTS_FILE
echo

echo "DATE_FINISHED   : "`date`
echo "JOB_COMPLETE"

