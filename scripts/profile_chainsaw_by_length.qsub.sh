#!/bin/bash

#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=128:0:0
#$ -o /home/ahawkins/chainsaw/logs

#$ -S /bin/bash
#$ -j y

ZIP_DIR=/SAN/bioinf/afdb_domain/zipfiles
REPO_DIR=${HOME_DIR}/chainsaw
REPRESENTATIVES_CSV=$REPO_DIR/proteome-tax_id-9606-14_v4_representatives.csv
SHARED_REPO=/SAN/bioinf/domdet/chainsaw_cluster/chainsaw
N_COPIES=100


SCRATCH_DIR=/scratch0/$USER
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
PYTHON_EXE=${SHARED_REPO}/venv/bin/python3

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
echo $SGE_TASK_ID
echo $(printf "%08d" $SGE_TASK_ID)
# ZIP_INDEX_FILE=/SAN/bioinf/afdb_domain/datasets/index/all_models_unique_partitions/zip_index.$(printf "%08d" $SGE_TASK_ID)
PROTEOME_FILE=$ZIP_DIR/proteome-tax_id-9606-14_v4.zip
echo $PROTEOME_FILE

RESULTS_DIR=${REPO_DIR}/speed_test

if [ ! -e "${PYTHON_EXE}" ]
then
    echo "ERROR: python executable does not exist - have you built a venv? (PYTHON_EXE=$PYTHON_EXE)"
    exit 1
fi

if [ ! -e "${PROTEOME_FILE}" ]
then
    echo "ERROR: input proteome file '${PROTEOME_FILE}' does not exist"
    exit 1
fi

if [ -e "${RESULTS_FILE}" ]
then
    echo "ERROR: output file '${RESULTS_FILE}' already exists (will not overwrite)"
    exit 1
fi

echo "PROTEOME_FILE         : $PROTEOME_FILE"
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
source /share/apps/source_files/python/python-3.8.5.source  # TODO: might need to change
source /SAN/bioinf/domdet/venv/bin/activate  # I copied this so that I have write permisssions
# https://askubuntu.com/questions/320996/how-to-make-python-program-command-execute-python-3
alias python=python3  # probably required for pyinstrument?
# source ${SHARED_REPO}/venv/bin/activate  # is this necessary?
echo "...DONE"


echo "Extracting PDBs ..."
cd $LOCAL_TASK_DIR
LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
mkdir -p $LOCAL_PDB_DIR
/usr/bin/time unzip $PROTEOME_FILE -d $LOCAL_PDB_DIR
ls $LOCAL_PDB_DIR | wc -l
# /usr/bin/time $ZIP_EXTRACT -i $ZIP_INDEX_FILE -z $ZIP_DIR -o $LOCAL_PDB_DIR 
echo "...DONE"

representatives=$(cut -d "," -f 1 proteome-tax_id-9606-14_v4_representatives.csv | tail -n +2)
for rep in $representatives
    do
        pdb_path=$LOCAL_PDB_DIR/$rep.pdb
        results_file=$RESULTS_DIR/$rep.csv
        json_file=$RESULTS_DIR/$rep.json
        echo "Running chainsaw 100 times on file "$pdb_path
        # TODO: to make this work we need a way of passing a list and of making python call python3
        # we also need a flag which tells get_predictions to rerun duplicates
        pyinstrument -r json -o $json_file $REPO_DIR/get_predictions.py --structure_file $(printf "${pdb_path}%.0s " {1..100}) \
        -o $results_file --ss_mod --model_dir ${SHARED_REPO}/saved_models/ss_c_base_no_excl/version_2/epoch_11 --force_rerun

done