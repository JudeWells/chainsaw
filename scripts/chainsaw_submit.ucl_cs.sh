#!/bin/bash -l

echo "!!!! WARNING: THIS HAS NOT BEEN TESTED YET!!!!"

#
# Submit chainsaw job array to SGE cluster
#
# usage:
#   qsub -t 1-16933 chainsaw_submit.ucl_cs.sh
#

#$ -l tmem=23G
#$ -l h_vmem=23G
#$ -l h_rt=4:55:00
#$ -S /bin/bash
#$ -N chainsaw-bfg
#$ -wd /SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold
#$ -t 3
#$ -o /SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold/qsub/logs
#$ -j y


if [[ "$SGE_TASK_ID" == "undefined" ]]; then
	echo "Need to specify the job array details in the qsub command"
	echo "example: qsub -t 1-16933:1 <script>"
    	echo
	exit 1
fi

CATH_VERSION=v4_3_0
ZIP_DIR=/SAN/bioinf/afdb_domain/zipmaker/zipfiles
HOME_DIR=/home/$USER
SHARED_REPO=$HOME_DIR/github/chainsaw
ZIP_EXTRACT=$SHARED_REPO/scripts/zip_extract.py
GIT_TAG=main

ZIP_INDEX_ID=zipfile.`expr $SGE_TASK_ID - 1`
SCRATCH_DIR=/scratch0/$USER
LOCAL_CODE_DIR=$SCRATCH_DIR/$JOB_NAME
LOCAL_CHAINSAW_HOME=$LOCAL_CODE_DIR/chainsaw
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
ZIP_INDEX_FILE=${SGE_O_WORKDIR}/data/zipindex.`printf "%08d" $ZIP_INDEX_ID`
RESULTS_FILE=${SGE_O_WORKDIR}/results/${ZIP_INDEX_FILE}.results.csv

if [ -e "${ZIP_INDEX_FILE}" ]
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
echo "GIT_TAG             : $GIT_TAG"
echo "LOCAL_CODE_DIR      : $LOCAL_CODE_DIR"
echo "LOCAL_CHAINSAW_HOME : $LOCAL_CHAINSAW_HOME"
echo "LOCAL_TASK_DIR      : $LOCAL_TASK_DIR"

# die on error
set -o errexit

echo "Creating output directories ..."
mkdir -p `dirname $RESULTS_FILE`
mkdir $LOCAL_CODE_DIR
mkdir $LOCAL_TASK_DIR
echo "...DONE"

echo "Loading python ..."
module load python/3.8.5
echo "...DONE"

# TODO: move to singularity?
echo "Installing code ..."
if [ -d "${LOCAL_CHAINSAW_HOME}" ]
then
    echo "WARNING: local chainsaw repo '${LOCAL_CHAINSAW_HOME}' already exists, attempting to pull latest changes"
    cd $LOCAL_CHAINSAW_HOME
    git pull
else
    echo "INFO: local chainsaw repo '${LOCAL_CHAINSAW_HOME}' does not exist, attempting to create"
    mkdir -p $LOCAL_CHAINSAW_HOME
    cd $LOCAL_CHAINSAW_HOME
    git clone ${SHARED_REPO}.git .
fi
echo "Updating chainsaw repo to git tag $GIT_TAG ... "
cd $LOCAL_CHAINSAW_HOME
git checkout $GIT_TAG
echo "Installing python dependencies ..."
LOCAL_PYTHON_EXE=$LOCAL_CHAINSAW_HOME/venv/bin/python
if [ -e "${LOCAL_PYTHON_EXE}" ]
then
    echo "WARNING: local python virtualenv '${LOCAL_PYTHON_EXE}' already exists, attempting to use"
else
    echo "INFO: local python virtualenv '${LOCAL_PYTHON_EXE}' does not exist, attempting to create"
    cd $LOCAL_CHAINSAW_HOME
    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
fi
echo "...DONE"

echo "Extracting PDBs ..."
cd $LOCAL_TASK_DIR
LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
mkdir -p $LOCAL_PDB_DIR
$ZIP_EXTRACT -i $ZIP_INDEX_FILE -z $ZIP_DIR -o $LOCAL_PDB_DIR > $LOCAL_TASK_DIR/zip_extract.log
echo "...DONE"

echo "Running chainsaw ..."
LOCAL_PYTHON_EXE $LOCAL_CHAINSAW_HOME/get_predictions.py --structure_directory $LOCAL_PDB_DIR > $RESULTS_FILE
echo

echo "DATE_FINISHED   : "`date`
echo "JOB_COMPLETE"

