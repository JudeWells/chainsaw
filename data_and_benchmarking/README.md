Train test splits can be found in `chainsaw_cath1363_train_test_splits.json`

Ground truth domains, Chainsaw predictions and performance metrics on the CATH1363 test set can be found in `chainsaw_model_v3_on_cath1363_test.csv` 

Chainsaw predictions on the CATH1363 test set can be reproduced with the following command:

`python get_predictions.py --structure_dir path/to/pdbs --model_dir saved_models/model_v3 --renumber_pdbs -o output.tsv --use_first_chain --no_post_processing`

Note that the predictions in `chainsaw_model_v3_on_cath1363_test.csv` assume zero-indexed pdbs with consecutive indices for all residues in the pdb file.

`|` separates different domains 

`_` separates different segments in the same domain
