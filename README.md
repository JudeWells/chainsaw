# Chainsaw

Chainsaw is a deep learning method for predicting protein domain boundaries for a given
protein structure.

If you find Chainsaw useful in your research, please cite:

**Chainsaw: protein domain segmentation with fully convolutional neural networks**

Jude Wells, Alex Hawkins-Hooker, Nicola Bordin, Brooks Paige and Christine Orengo

[Bioinformatics](https://doi.org/10.1093/bioinformatics/btae296)

## Installation
Ensure that python 3.8, 3.9, 3.10, or 3.11 is installed, then install dependencies with 
the command `bash setup.sh`

Optional:
To visualise the domain assignments, ensure that you have pymol installed and update the
`PYMOL_EXE` variable in `src/constants.py` to point to the pymol executable.

## Usage
`python get_predictions.py --structure_file /path/to/file.pdb`
or
`python get_predictions.py --structure_directory /path/to/pdb_or_mmcif_directory`

Note that the output predicted boundaries are based on residue consecutive indexing
starting from 1 (not based on pdb auth numbers).

## Installation Troubleshooting
If setup.sh fails follow the instructions below:

1) install stride: source code and instructions are packaged in this repository in the
    `stride` directory. See stride/README_stride for instructions. 

2) install the python dependencies: `pip install -r requirements.txt`

3) test it's working by running `python get_predictions.py --structure_file example_files/AF-A0A1W2PQ64-F1-model_v4.pdb --output results/test.tsv`
    by default the output will be saved in the `results` directory.

