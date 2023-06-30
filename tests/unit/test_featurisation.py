from pathlib import Path
import Bio.PDB
import numpy as np
from get_predictions import get_model_structure, get_distance

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "example_files"


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues."""
    try:
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
        dist = np.sqrt(np.sum(diff_vector * diff_vector))
    except:
        dist = 20.0
    return dist


def v0_calc_dist_matrix(chain):
    """Returns a matrix of C-alpha distances between two chains"""
    distances = np.zeros((len(chain), len(chain)), 'float')
    for row, residue_one in enumerate(chain):
        for col, residue_two in enumerate(chain):
            distances[row, col] = calc_residue_dist(residue_one, residue_two)
    return distances


def v0_get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    # n.b. removed if chain is not None and replaced with assert for now
    if chain is not None:
        residues = [c for c in structure_model[chain].child_list]
    dist_matrix = v0_calc_dist_matrix(structure_model[chain].child_list)  # recycling dimensions are added later
    x = np.expand_dims(dist_matrix, axis=0)
    # replace zero values and then invert.
    x[0][x[0] == 0] = x[0][x[0] > 0].min()  # replace zero values in pae / distance
    x[0] = x[0] ** (-1)
    return x


def test_vectorised_distances():
    af_id = "AF-A0A1W2PQ64-F1-model_v4"
    example_structure_path = str(DATA_DIR / f"{af_id}.pdb")
    model_structure = get_model_structure(example_structure_path)
    expected_distances = v0_get_distance(model_structure)
    distances = get_distance(model_structure)

    # don't get exact matches for some reason
    assert np.isclose(distances, expected_distances).all()
