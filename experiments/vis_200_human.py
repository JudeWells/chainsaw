import os
import argparse
from get_predictions import predict, load_model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='saved_models/secondary_structure_epoch17/version_2',
                        help='path to model directory must contain model.pt and config.json')
    parser.add_argument('--uniprot_id', type=str, default=None, help='single uniprot id')
    parser.add_argument('--uniprot_id_list_file', type=str, default=None,
                        help='path to file containing uniprot ids')
    parser.add_argument('--structure_directory', type=str, default=None,
                        help='path to directory containing PDB or MMCIF files')
    parser.add_argument('--structure_file', type=str, default=None,
                        help='path to PDB or MMCIF files')
    parser.add_argument('--pdb_id', type=str, default=None, help='single pdb id')
    parser.add_argument('--pdb_id_list_file', type=str, default=None, help='path to file containing uniprot ids')
    parser.add_argument('--save_dir', type=str, default='results', help='path where results and images will be saved')
    parser.add_argument('--remove_disordered_domain_threshold', type=float, default=0.35,
                        help='if the domain is less than this fraction secondary structure, it will be removed')
    structure_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/UP000005640_9606_HUMAN_v4"
    outer_save_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/human200"
    args = parser.parse_args()
    model = load_model(args)
    with open("experiments/sampled_structures.txt", 'r') as f:
        lines = f.readlines()
    for af_id in lines:
        af_id, n_res = af_id.split(",")
        af_id = af_id.strip()
        print(af_id, n_res)
        pdb_path = os.path.join(structure_dir, f"{af_id}.pdb")
        predict(model, pdb_path, outer_save_dir, pymol_visual=True)