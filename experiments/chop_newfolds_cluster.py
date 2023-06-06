import sys
import os
import argparse
from get_predictions import predict, load_model, parse_args, write_pymol_script
import time

parser = argparse.ArgumentParser()
parser.add_argument('--task_index', type=int, default=0)
parser.add_argument('--pymol_exe', type=str, default="/Applications/PyMOL.app/Contents/MacOS/PyMOL")
parser.add_argument('--working_dir', type=str, default="/SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold")
parser.add_argument('--structure_dir', type=str, default="/SAN/cath/cath_v4_3_0/alphafold/chainsaw_on_alphafold/newfolds_from_tcluster/pdb")
parser.add_argument('--outer_save_dir', type=str, default="chainsaw_results")
parser.add_argument('--pymol_visual', dest='pymol_visual', action='store_true', help='whether to generate pymol images')

args = parser.parse_args()

global PYMOL_EXE
PYMOL_EXE = args.pymol_exe


def write_index_file(structure_index_filepath, structure_dir):
    with open(structure_index_filepath, 'w') as f:
        for fname in sorted(os.listdir(structure_dir)):
            f.write(fname + "\n")

def write_results(prediction_results, result_dir):
    with open(os.path.join(result_dir, prediction_results[0].chain_id + '.txt'), 'w') as f:
        f.write(f'{prediction_results[0].chain_id}, uncertainty:{prediction_results[0].uncertainty:.3g}\n')
        for res in prediction_results:
            f.write(f'{res.domain_id}, {res.chopping}\n')

if __name__=="__main__":
    start = time.time()
    working_directory = args.working_dir
    os.chdir(working_directory)
    error_filepath = 'errors.txt'
    if not os.path.exists(error_filepath):
        with open(error_filepath, 'w') as f:
            f.write("")
    batch_size = 36
    task_index = int(args.task_index) -1
    structure_dir = args.structure_dir
    outer_save_dir = args.outer_save_dir
    structure_index_filepath = 'file_list.txt'
    if not os.path.exists(structure_index_filepath):
        write_index_file(structure_index_filepath, structure_dir)

    with open(structure_index_filepath, 'r') as f:
        structure_lines = f.readlines()[task_index*batch_size:(task_index+1)*batch_size]
    sys.argv = sys.argv[:1] # stupid hack to use argparse with two sets of arguments
    args = parse_args()
    visualize_domain_preds = args.pymol_visual
    image_save_dir = os.path.join(outer_save_dir, "images")
    os.makedirs(outer_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    model = load_model(model_dir=args.model_dir, remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
                       min_ss_components=args.min_ss_components, min_domain_length=args.min_domain_length)

    for i, line in enumerate(structure_lines):
        structure_path = os.path.join(structure_dir, line.strip())
        try:
            prediction_results = predict(model, structure_path, renumber_pdbs=True)
            write_results(prediction_results, outer_save_dir)
            if i in [0, 5] and visualize_domain_preds:
                write_pymol_script(results=prediction_results, save_dir=image_save_dir)
        except Exception as e:
            with open(error_filepath, 'a') as f:
                f.write(f"{line.strip()}, {e}, {task_index} \n")
            continue
    total_time = time.time() - start
    print(f"Time taken for {batch_size}: {total_time}")

