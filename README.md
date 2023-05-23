# CHAINSAW

Chainsaw is a deep learning method for predicting protein domain boundaries for a given
protein structure.

## Installation

1) install stride: scource code and instructions are packaged in this repository in the
    `stride` directory.  You will need to compile stride and put the executable in your
    path. Update the stride_path variable in get_predictions.py to point to the stride
    executable.

2) install the python dependencies: `pip install -r requirements.txt`

3) test it's working by running `python get_predictions.py --structure_file example_files/AF-Q5T5X7-F1-model_v4.pdb`
    by default the output will be saved in the `results` directory.

## Usage
`python get_predictions.py --structure_file /path/to/file.pdb`
or
`python get_predictions.py --structure_directory /path/to/pdb_or_mmcif_directory`