#!/bin/bash

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y

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
SHARED_REPO=/SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold/chainsaw
SHARED_REPO="${SHARED_REPO:-${SGE_O_WORKDIR}/chainsaw}"
ZIP_EXTRACT=$SHARED_REPO/scripts/zip_extract.py
SCRATCH_DIR=/scratch0/$USER
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
PYTHON_EXE=${SGE_O_WORKDIR}/venv/bin/python3

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
ZIP_INDEX_FILE=/SAN/bioinf/afdb_domain/datasets/index/all_models_unique_partitions/zip_index.$(printf "%08d" $SGE_TASK_ID)
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
source /share/apps/source_files/python/python-3.8.5.source
# source ./venv/bin/activate
echo "...DONE"

echo "Extracting PDBs ..."
cd $LOCAL_TASK_DIR
LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
mkdir -p $LOCAL_PDB_DIR
/usr/bin/time $ZIP_EXTRACT -i $ZIP_INDEX_FILE -z $ZIP_DIR -o $LOCAL_PDB_DIR 
echo "...DONE"

echo "Running chainsaw ..."
/usr/bin/time $PYTHON_EXE $SHARED_REPO/get_predictions.py --structure_directory $LOCAL_PDB_DIR -o $RESULTS_FILE \
  --ss_mod --model_dir $SHARED_REPO/saved_models/ss_c_base_no_excl/version_2/epoch_11
echo

echo "Removing local temp dir ..."
rm -rf $LOCAL_TASK_DIR
echo

echo "DATE_FINISHED   : "`date`
echo "JOB_COMPLETE"

