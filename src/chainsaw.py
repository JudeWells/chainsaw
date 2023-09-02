"""Domain predictor classes.
"""
import hashlib
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from src import featurisers
from src.domain_assignment.util import make_pair_labels, make_domain_mapping_dict
from src.post_processors import SSPostProcessor
from src.models.results import PredictionResult


import logging
LOG = logging.getLogger(__name__)


class Chainsaw(nn.Module):

    """Wrapper for a pairwise domain co-membership predictor, adding in domain prediction post-processing."""

    def __init__(
        self,
        model,
        domain_caller,
        device,
        loss="bce",
        x_has_padding_mask=True,
        checkpoint_dir=None,
        max_recycles=0,
        post_process_domains=True,
        min_ss_components=2,
        min_domain_length=30,
        remove_disordered_domain_threshold=0,
        trim_each_domain=True,
        renumber_pdbs=True,
        ss_mod=True,
        **kwargs,  # note that there may be some additional train kwargs
    ):
        nn.Module.__init__(self)
        self.model = model
        self.domain_caller = domain_caller
        self.device = device
        self.x_has_padding_mask = x_has_padding_mask
        self.max_recycles = max_recycles
        self.post_process_domains = post_process_domains
        self.checkpoint_dir = checkpoint_dir
        self.renumber_pdbs = renumber_pdbs
        self.ss_mod = ss_mod
        self.post_processor = SSPostProcessor(
            min_ss_components=min_ss_components,
            min_domain_length=min_domain_length,
            remove_disordered_domain_threshold=remove_disordered_domain_threshold,
            trim_each_domain=trim_each_domain,
        )

    def load_checkpoints(self):
        weights_file = os.path.join(self.checkpoint_dir, "weights.pt")
        LOG.info(f"Loading weights from: {weights_file}")
        state_dict = torch.load(weights_file, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def predict_pairwise(self, x):
        x = x.to(self.device)
        if np.isnan(x.cpu().numpy()).any():
            raise Exception('NAN values in data')
        y_pred = self.model(x).squeeze(1)  # b, L, L
        assert y_pred.ndim == 3
        return y_pred

    def domains_from_pairwise(self, y_pred):
        assert y_pred.ndim == 3
        domain_preds = []
        uncertainty_list = []
        for pred_single in y_pred.cpu().numpy():
            single_domains, uncertainty = self.domain_caller(pred_single)
            domain_preds.append(single_domains)
            uncertainty_list.append(uncertainty)
        return domain_preds, uncertainty_list

    @torch.no_grad()
    def _predict(self, x, return_pairwise=True):
        x = x.to(self.device)
        for i in range(self.max_recycles):
            x = self.recycle_predict(x)
        y_pred = self.predict_pairwise(x)
        domain_dicts, uncertainty = self.domains_from_pairwise(y_pred)
        if return_pairwise:
            return y_pred, domain_dicts, uncertainty
        else:
            return domain_dicts, uncertainty

    def featurise(self, pdb_path, chain_id):
        # TODO refactor to remove repeated pdb_path arg, so that we
        # only need model_structure as input to featurise.
        x = featurisers.inference_time_create_features(
            pdb_path,
            chain=chain_id, 
            secondary_structure=True, 
            renumber_pdbs=self.renumber_pdbs, 
            model_structure=model_structure,
            ss_mod=self.ss_mod,
            add_recycling=self.max_recycles > 0,
        )
        return x

    def predict(self, pdb_path, chain_id):
        model_structure = featurisers.get_model_structure(pdb_path)
        model_structure_seq = featurisers.get_model_structure_sequence(model_structure, chain=chain_id)
        model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()
        feats = self.featurise(model_structure, chain_id)
        domain_dicts, uncertainty_array = self._predict(x, return_pairwise=False)
        # remove batch dimension
        domain_dict = domain_dicts[0]
        if self.post_process_domains:
            x = x[0].cpu().numpy()
            helix = x[1]
            sheet = x[2]
            domain_dict = self.post_processor.post_process(domain_dict, helix, sheet) # todo move this to domains from pairwise function

        chopping_str = self.domain_dict_to_chopping_str(domain_dicts[0])
        num_domains = 0 if chopping_str is None else len(chopping_str.split(','))
        result = PredictionResult(
            pdb_path=pdb_path,
            sequence_md5=model_structure_md5,
            nres=len(model_structure_seq),
            ndom=num_domains,
            chopping=chopping_str,
            uncertainty=uncertainty_array[0],
        )
        return result

    @torch.no_grad()
    def recycle_predict(self, x):
        x = x.to(self.device)
        y_pred = self.predict_pairwise(x)
        domain_dicts, uncertainty = self.domains_from_pairwise(y_pred)
        y_pred_from_domains = np.array(
            [make_pair_labels(n_res=x.shape[-1], domain_dict=d_dict) for d_dict in domain_dicts])
        y_pred_from_domains = torch.tensor(y_pred_from_domains).to(self.device)
        if self.x_has_padding_mask:
            x[:, -2, :, :] = y_pred # assumes that last dimension is padding mask
            x[:, -3, :, :] = y_pred_from_domains
        else:
            x[:, -1, :, :] = y_pred
            x[:, -2, :, :] = y_pred_from_domains
        return x
