import os
import time

import pandas as pd

from get_predictions import predict, load_model, parse_args
import matplotlib.pyplot as plt


if __name__=="__main__":
    args = parse_args()
    # structure_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/human_200_c2m"
    outer_save_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/human200_ss17_model_w_post_proc"
    structure_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/UP000005640_9606_HUMAN_v4"
    # outer_save_dir = "/Users/judewells/Documents/dataScienceProgramming/data_for_domdet/human200_mse_ss_excl_f32_MSE_sym"
    os.makedirs(outer_save_dir, exist_ok=True)
    model = load_model(model_dir=args.model_dir, remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
                       min_ss_components=args.min_ss_components, min_domain_length=args.min_domain_length)
    with open("experiments/sampled_structures.txt", 'r') as f:
        lines = f.readlines()
    res_list = []
    time_list = []
    af_id_list = []
    for i, af_id in enumerate(lines):
        try:
            af_id, n_res = af_id.split(",")
            af_id = af_id.strip()
            print(af_id, n_res)
            pdb_path = os.path.join(structure_dir, f"{af_id}.pdb")
            start = time.time()
            prediction_results = predict(model, pdb_path, renumber_pdbs=False)
            end = time.time()
            res_list.append(int(n_res))
            time_list.append(end-start)
            af_id_list.append(af_id)
        except:
            pass
    plt.scatter(res_list, time_list)
    plt.xlabel("Number of residues")
    plt.ylabel("Time (s)")
    plt.savefig("chainsaw_macbook_speed_test.png")
    plt.show()
    df = pd.DataFrame({"af_id": af_id_list, "n_res": res_list, "time": time_list})
    df.to_csv("chainsaw_macbook_speed_test.csv", index=False)

