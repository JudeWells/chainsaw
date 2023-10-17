import os
import pandas as pd
from src.benchmark.benchmark import get_prediction_metrics_from_df

"""
UniDoc needs to be run on linux
This script assumes that you have already generated
the output predictions from unidoc and stored the preds in a 
directory called unidoc_dir

See the following script for an example of how to run unidoc:
/SAN/bioinf/domdet/domdet/src/configuration/submit_scripts/unidoc.qsub.sh

Or to run on the facebook test set:
/SAN/bioinf/domdet/domdet/src/benchmark/unidoc/non_cath_run_unidoc_fb_test.qsub.sh

For some reason the program does not work on the CATH pdbs.
"""


def make_unidoc_pred_df(unidoc_dir="src/benchmark/unidoc/unidoc_preds"):
    new_rows = []
    for fname in os.listdir(unidoc_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(unidoc_dir, fname), "r") as f:
                lines = f.readlines()
        assert len(lines) == 1
        pred_str = lines[0].strip()
        dnames = []
        dom_bounds = []
        for i, dbounds in enumerate(pred_str.split("/")):
            for dseg in dbounds.split(","):
                dnames.append(f"d{i+1}")
                dom_bounds.append(dseg.replace("~", "-"))
        bounds_str = "|".join(dom_bounds)
        name_str = "|".join(dnames)
        chain_id = fname.split(".")[0]
        new_rows.append(
            {
                "chain_id": chain_id,
                "dom_bounds_pdb_ix": bounds_str,
                "dom_names": name_str,
                "n_domains": len(set(dnames)),
            }
        )
    return pd.DataFrame(new_rows)


def adjust_zero_indexing(df):
    """
    adds 1 to all the domain indexes to turn zero-indexing into one-indexing
    """
    for i, row in df.iterrows():
        new_bounds = []
        for dom_bounds in row.dom_bounds_pdb_ix.split("|"):
            start, end = dom_bounds.split("-")
            new_bounds.append(f"{int(start)+1}-{int(end)+1}")
        df.loc[i, "dom_bounds_pdb_ix"] = "|".join(new_bounds)
    return df


if __name__ == "__main__":
    # test_path = 'casp6_test_primary_assignment_only.csv'
    # pred_path = 'src/benchmark/unidoc/casp_unidoc_preds'
    # test_path = 'facebook_test.csv'
    test_path = "cath_nopae_facebook_test.csv"
    pred_path = "src/benchmark/unidoc/facebook_test_preds"
    savepath = f"src/benchmark/unidoc/unidoc_results_{test_path}"
    # test_path = 'casp14_test.csv'
    preds = make_unidoc_pred_df(unidoc_dir=pred_path)
    df = pd.read_csv(test_path)
    df["n_domains"] = df.dom_names.apply(lambda x: len(set(x.split("|"))))
    df = adjust_zero_indexing(df)
    preds = get_prediction_metrics_from_df(df, preds)
    preds.to_csv(savepath, index=False)
    bp = 1
