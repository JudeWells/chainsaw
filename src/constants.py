import os


BASEDIR = os.path.dirname(os.path.dirname(__file__))
IS_HPC = os.environ.get("CHAINSAW_ENV", None) == "hpc"
if IS_HPC:
    DATA_DIR = "/SAN/bioinf/domdet"
else:
    DATA_DIR = BASEDIR


aa_letters = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

aa_letters_wgap = ['-'] + aa_letters