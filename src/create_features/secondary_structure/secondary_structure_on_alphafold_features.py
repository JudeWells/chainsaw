"""
Created by Jude Wells 2023-05-22
create features to run Chainsaw on
alphafold structures for the Facebook
test set, so that we can run benchmarking
of the latest chainsaw models to compare
performance when using alphafold structures
instead of PDB structures.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PDBIO

from src.create_features.make_2d_features import get_up_distance_and_plddt, reshape_distance_matrix, create_mapping,\
    MISSING_UP_IX, fetch_missing_pdb_res_ix, fetch_alphafold_pdb, make_pair_labels, make_domain_mapping_dict

from src.create_features.secondary_structure.secondary_structure_features import calculate_ss, make_ss_matrix


def get_test_ids():
    with open("splits.json", "r") as f:
        splits = json.load(f)
    test_ids = splits["test"]
    return test_ids


def remove_residues_by_index(pdb_filename, output_filename, indexes_to_remove):
    parser = PDBParser()
    structure = parser.get_structure("my_structure", pdb_filename)
    io = PDBIO()

    for model in structure:
        for chain in model:
            for residue in list(chain):
                # PDB residue numbers are accessed with residue.get_id()[1]
                if residue.get_id()[1] in indexes_to_remove:
                    chain.detach_child(residue.get_id())

    io.set_structure(structure)
    io.save(output_filename)

def trim_alphafold_pdb_to_match_pdb(row, mapping_dict, pdb_dir, alphafold_structure_dir):
    """
    Cut all ATOM lines from the AlphaFold PDB file
    that do not align with the PDB file.
    removes residues that are either not aligned in SIFTS
    and when some residues are missing in the PDB file
    """
    pdbid = row.chain_id[:-1]
    chain = row.chain_id[-1]
    pdb_missing = list(fetch_missing_pdb_res_ix(pdbid, chain, pdb_dir=pdb_dir))
    residues_to_remove = [k for k, v in mapping_dict.items() if v == MISSING_UP_IX]
    residues_to_remove += pdb_missing
    alphafold_filepath = os.path.join(alphafold_structure_dir, f"{row.uniprot_id}.pdb")
    trimmed_filepath = alphafold_filepath.replace(".pdb", "_trimmed.pdb")
    if not os.path.exists(alphafold_filepath):
        fetch_status = fetch_alphafold_pdb(up_id, alphafold_structure_dir)
        if fetch_status != "success":
            print(f"Failed to fetch {up_id}")
            return
    remove_residues_by_index(alphafold_filepath, trimmed_filepath, residues_to_remove)
    return trimmed_filepath


if __name__=="__main__":
    alphafold_structure_dir = "../data_for_domdet/data/af_pdbs"
    pdb_dir = "../data_for_domdet/features/cath_pdbs_for_unidoc"
    stride_path = '/Users/judewells/bin/stride'
    temporary_ss_path = "/tmp/ss.txt"
    df = pd.read_csv("training_v4.csv")
    output_dir = "../data_for_domdet/features/alphafold_facebook_test"
    new_features_dir = os.path.join(output_dir, "2d_features")
    new_label_dir = os.path.join(output_dir, "pairwise")
    os.makedirs(new_features_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    test_ids = get_test_ids()
    df = df[df.chain_id.isin(test_ids)]
    min_coverage_threshold = 0.8
    new_rows = []
    for i, row in df.iterrows():
        if row.uniprot_id == "Q96BD8":
            bp=1
        try:
            up_id = row["uniprot_id"]
            pdbid = row.chain_id[:-1]
            chain = row.chain_id[-1]
            alphafold_filepath = os.path.join(alphafold_structure_dir, f"{up_id}.pdb")
            mapping_dict = create_mapping(row)
            pdb_missing = list(fetch_missing_pdb_res_ix(pdbid, chain, pdb_dir=pdb_dir))
            for miss_res in pdb_missing:
                mapping_dict[miss_res] = MISSING_UP_IX
            non_aligned = [k for k, v in mapping_dict.items() if v == MISSING_UP_IX]
            if len(non_aligned) / len(mapping_dict) > 1 - min_coverage_threshold:
                print(f"Skipping {up_id} due to low coverage")
                continue
            dist_matrix, plddt = get_up_distance_and_plddt(row, id_col='uniprot_id', pdb_dir=alphafold_structure_dir, return_plddt=True)
            nres_uniprot = dist_matrix.shape[0]
            calculate_ss(alphafold_filepath, "A", stride_path, ssfile=temporary_ss_path)
            helix, strand = make_ss_matrix(temporary_ss_path, nres_uniprot)
            os.remove(temporary_ss_path)
            stacked_features = []
            for matrix_2d in [dist_matrix, helix, strand]:
                stacked_features.append(reshape_distance_matrix(matrix_2d, mapping_dict, trim_non_aligned=True))
            domain_dict = make_domain_mapping_dict(row)
            labels = make_pair_labels(len(mapping_dict), domain_dict, id_string=None, save_dir=None, non_aligned_residues=non_aligned)
            assert len(labels) == len(stacked_features[0])
            new_row = row.to_dict()
            new_row["plddt"] = plddt.mean()
            new_row["pdb_missing"] = "|".join([str(x) for x in pdb_missing])
            new_rows.append(new_row)
            np.savez_compressed(os.path.join(new_features_dir, f"{up_id}.npz"), np.stack(stacked_features))
            np.savez_compressed(os.path.join(new_label_dir, f"{up_id}.npz"), labels)
        except Exception as e:
            print(f"Failed to process {up_id}")
            print(e)
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(os.path.join(output_dir, "alphafold_features_for_facebook_test.csv"), index=False)



        

    bp=1
