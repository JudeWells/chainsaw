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

log () {
    echo "[$(date)] ${1:-}"
}

if [[ "$SGE_TASK_ID" == "undefined" ]]; then
	log "Need to specify the job array details in the qsub command"
	log "example: qsub -t 1-16933:1 <script>"
	log
	exit 1
fi

CATH_VERSION=v4_3_0
HOME_DIR=/home/$USER
SHARED_REPO="${SHARED_REPO:-${SGE_O_WORKDIR}/chainsaw}"
SCRATCH_DIR=/scratch0/$USER
LOCAL_TASK_DIR=$SCRATCH_DIR/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
PYTHON_EXE=${SGE_O_WORKDIR}/venv/bin/python3
APPEND_FLAG="--append"
RESULT_VERSION=1

# assumes the zip index file has been split into chunks (named 'zipindex.00000001')
ZIPFILES_LIST_FILE=${SGE_O_WORKDIR}/data/zipfiles.`printf "%08d" $SGE_TASK_ID`
RESULTS_FILE=${SGE_O_WORKDIR}/results/$(basename $ZIPFILES_LIST_FILE).results_v${RESULT_VERSION}.csv

if [ ! -e "${PYTHON_EXE}" ]
then
    log "ERROR: python executable does not exist - have you built a venv? (PYTHON_EXE=$PYTHON_EXE)"
    exit 1
fi

if [ ! -e "${ZIPFILES_LIST_FILE}" ]
then
    log "ERROR: input list file '${ZIPFILES_LIST_FILE}' does not exist"
    exit 1
fi

if [ -e "${RESULTS_FILE}" && "${APPEND_FLAG}" != "" ]
then
    log "ERROR: output file '${RESULTS_FILE}' already exists (will not overwrite)"
    exit 1
fi

log "ZIPFILES_LIST_FILE : $ZIPFILES_LIST_FILE"
log "RESULTS_FILE       : $RESULTS_FILE"
log "DATE_STARTED       : "`date`
log
log "HOSTNAME            : $HOSTNAME"
log "SGE_TASK_ID         : $SGE_TASK_ID"
log "GIT_REMOTE          : $SHARED_REPO"
log "LOCAL_TASK_DIR      : $LOCAL_TASK_DIR"

# die on errors
set -e

log "Creating output directories ..."
mkdir -p $(dirname $RESULTS_FILE)
log "   ...DONE"

log "Loading python ..."
source /share/apps/source_files/python/python-3.8.5.source
log "   ...DONE"


process_zipfile () {

    zipfile="$1"

    mkdir -p $LOCAL_TASK_DIR
    cd $LOCAL_TASK_DIR
    LOCAL_PDB_DIR=$LOCAL_TASK_DIR/pdb
    mkdir -p $LOCAL_PDB_DIR

    log "Unzipping $zipfile ..."
    unzip -d $LOCAL_PDB_DIR $zipfile
    log "   ...DONE"

    log "Running chainsaw on $zipfile ..."
    $PYTHON_EXE $SHARED_REPO/get_predictions.py --structure_directory $LOCAL_PDB_DIR $APPEND_FLAG -o $RESULTS_FILE && rc=$? || rc=$?
    log "EXIT_CODE: $rc"
    log "...DONE"

    log "Removing local temp task dir ..."
    rm -rf $LOCAL_TASK_DIR
    log "...DONE"

    log "DATE_FINISHED   : " $(date)
    if [ "$rc" == "0" ]
    then
        log "JOB_COMPLETE OK $zipfile"
    elif [ "$rc" == "1" ]
    then
        log "JOB_COMPLETE ERROR $zipfile"
    fi

}

for zipfile in $(cat $ZIPFILES_LIST_FILE)
do
    process_zipfile $zipfile
done
