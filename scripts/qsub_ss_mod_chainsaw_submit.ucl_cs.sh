#!/bin/bash
#$ -l tmem=7G
#$ -l h_vmem=7G
#$ -l h_rt=40:00:00
#$ -l scratch0free=10G
#$ -L tscratch=10G
#$ -P cath
#$ -S /bin/bash
#$ -N chsaw-ss_mod
#$ -e ./logs
#$ -o ./logs
#$ -j y
#$ -t 1-18892:1000
cat "$0"
cd /SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold/chainsaw
bash scripts/ss_mod_chainsaw_submit.ucl_cs.sh