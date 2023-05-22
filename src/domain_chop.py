"""Domain predictor classes.
"""
import copy
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.create_features.make_2d_features import make_pair_labels
from src.create_features.make_2d_features import make_domain_mapping_dict




def get_checkpoint_epoch(checkpoint_file):
    return int(os.path.splitext(checkpoint_file)[0].split(".")[-1])


class PairwiseDomainPredictor(nn.Module):

    """Wrapper for a pairwise domain co-membership predictor, adding in domain prediction post-processing."""

    def __init__(
        self,
        model,
        domain_caller,
        device,
        loss="bce",
        x_has_padding_mask=True,
        mask_padding=True,
        n_checkpoints_to_average=1,
        checkpoint_dir=None,
        load_checkpoint_if_exists=False,
        save_val_best=True,
        max_recycles=0,
        trim_disordered=False,
        remove_disordered_domain_threshold=0,
    ):
        super().__init__()
        self._train_model = model  # we want to keep this hanging around so that optimizer references dont break
        self.model = self._train_model
        self.domain_caller = domain_caller
        self.device = device
        self.x_has_padding_mask = x_has_padding_mask
        self.mask_padding = mask_padding  # if True use padding mask to mask loss
        self.n_checkpoints_to_average = n_checkpoints_to_average
        self.checkpoint_dir = checkpoint_dir
        self._epoch = 0
        self.save_val_best = save_val_best
        self.best_val_metrics = {}
        self.max_recycles = max_recycles
        self.trim_disordered = trim_disordered
        self.remove_disordered_domain_threshold = remove_disordered_domain_threshold
        if load_checkpoint_if_exists:
            checkpoint_files = sorted(
                glob.glob(os.path.join(self.checkpoint_dir, "weights*")),
                key=get_checkpoint_epoch,
                reverse=True,
            )
            if len(checkpoint_files) > 0:
                self._epoch = get_checkpoint_epoch(checkpoint_files[0])
                print(f"Loading saved checkpoint(s) ending at epoch {self._epoch}")
                self.load_checkpoints(average=True)
                self.load_checkpoints()
            else:
                print("No checkpoints found to load")

        if loss == "bce":
            self.loss_function = nn.BCELoss(reduction="none")
        elif loss == "mse":
            self.loss_function = nn.MSELoss(reduction="none")

    def load_checkpoints(self, average=False, old_style=False):
        start_idx = max(self._epoch - self.n_checkpoints_to_average, 1)
        end_idx = self._epoch
        if self.n_checkpoints_to_average == 1:
            weights_file = os.path.join(self.checkpoint_dir, "weights.pt")
            print(f"loading model weights from {weights_file}")
        else:
            # for e.g. resetting training weights for next training epoch after testing with avg
            print(f"Loading last checkpoint (epoch {end_idx})", flush=True)
            weights_file = os.path.join(self.checkpoint_dir, f"weights.{end_idx}.pt")
        print("Loading weights from", weights_file, flush=True)
        state_dict = torch.load(weights_file, map_location=self.device)
        if old_style:
            self.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

    def predict_pairwise(self, x):
        x = x.to(self.device)
        if np.isnan(x.cpu().numpy()).any():
            raise Exception('NAN values in data')
        y_pred = self.model(x).squeeze(1)  # b, L, L
        assert y_pred.ndim == 3
        return y_pred

    def get_mask(self, x):
        """Binary mask 1 for observed, 0 for padding."""
        x = x.to(self.device)
        if self.x_has_padding_mask:
            mask = 1 - x[:, -1]  # b, L, L
        else:
            mask = None
        return mask

    def epoch_start(self):
        self.model = self._train_model
        self.model.train()
        self._epoch += 1

    def test_begin(self):
        if self.n_checkpoints_to_average > 1:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'weights.{self._epoch}.pt'))
            start_idx = self._epoch - self.n_checkpoints_to_average
            if start_idx >= 2:
                os.remove(os.path.join(self.checkpoint_dir, f"weights.{start_idx-1}.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'weights.pt'))

        if self.n_checkpoints_to_average > 1:
            # self.model.to("cpu")  # free up gpu memory for average model
            self.load_checkpoints(average=True)

        self.model.eval()

    def forward(self, x, y, batch_average=True):
        """A training step."""
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.predict_pairwise(x)
        mask = self.get_mask(x)
        return self.compute_loss(y_pred, y, mask=mask)

    def compute_loss(self, y_pred, y, mask=None, batch_average=True):
        y_pred, y = y_pred.to(self.device), y.to(self.device)
        if mask is None or not self.mask_padding:
            mask = torch.ones_like(y)
        # mask is b, L, L. To normalise correctly, we need to divide by number of observations
        loss = (self.loss_function(y_pred, y)*mask).sum((-1,-2)) / mask.sum((-1,-2))

        # metrics characterising inputs: how many residues, how many with domain assignments.
        labelled_residues = ((y*mask).sum(-1) > 0).sum(-1)  # b
        non_padding_residues = (mask.sum(-1) > 0).sum(-1)  # b
        labelled_frac = labelled_residues / non_padding_residues
        metrics = {
            "labelled_residues": labelled_residues.detach().cpu().numpy(),
            "residues": non_padding_residues.detach().cpu().numpy(),
            "labelled_frac": labelled_frac.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy(),
        }
        if batch_average:
            loss = loss.mean(0)
            metrics = {k: np.mean(v) for k, v in metrics.items()}

        return loss, metrics

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
    def predict(self, x, return_pairwise=True):
        x = x.to(self.device)
        for i in range(self.max_recycles):
            x = self.recycle_predict(x)
        y_pred = self.predict_pairwise(x)
        domain_dicts, uncertainty = self.domains_from_pairwise(y_pred)
        if self.trim_disordered:
            domain_dicts = self.remove_disordered_from_assignment(domain_dicts, x) # todo move this to domains from pairwise function
        if return_pairwise:
            return y_pred, domain_dicts, uncertainty
        else:
            return domain_dicts, uncertainty

    @torch.no_grad()
    def recycle_predict(self, x):
        x = x.to(self.device)
        y_pred = self.predict_pairwise(x)
        domain_dicts, uncertainty = self.domains_from_pairwise(y_pred)
        y_pred_from_domains = np.array([make_pair_labels(n_res=x.shape[-1], domain_dict=d_dict) for d_dict in domain_dicts])
        y_pred_from_domains = torch.tensor(y_pred_from_domains).to(self.device)
        if self.x_has_padding_mask:
            x[:, -2, :, :] = y_pred # assumes that last dimension is padding mask
            x[:, -3, :, :] = y_pred_from_domains
        else:
            x[:, -1, :, :] = y_pred
            x[:, -2, :, :] = y_pred_from_domains
        return x

    def remove_disordered_from_assignment(self, domain_dicts, x): # todo should then check if minimum domain size is met
        """Residues that aren't part of secondary structure at the start and end
        of the chain are removed from domain assignments and assigned to linker regions.
        if self.remove_disordered_domain_threshold >0 then domains which are less than this threshold secondary structure are removed
        """
        new_domain_dicts = []
        for single_assign, single_x in zip(domain_dicts, x):
            helix, sheet = single_x[1].cpu().numpy(), single_x[2].cpu().numpy()
            trace_helix = np.diagonal(helix)
            trace_sheet = np.diagonal(sheet)
            ss_residues = list(np.where(trace_helix==1)[0]) + list(np.where(trace_sheet==1)[0])
            if len(ss_residues) < 8:
                new_domain_dicts.append({"linker": [i for i in range(x.shape[-1])]})
                continue
            start = min(ss_residues)
            end = max(ss_residues)
            single_assign = {k: [r for r in v if r >= start and r <= end] for k, v in single_assign.items()}
            single_assign["linker"] += [r for r in range(start)] + [r for r in range(end+1, x.shape[-1])]
            if self.remove_disordered_domain_threshold > 0:
                new_single_assign = {}
                for dname, res in single_assign.items():
                    if dname == "linker":
                        continue
                    if len(res) == 0:
                        continue
                    if len(set(res).intersection(set(ss_residues))) / len(res) < self.remove_disordered_domain_threshold:
                        single_assign["linker"] += res
                    else:
                        new_single_assign[dname] = res
                new_single_assign["linker"] = single_assign["linker"]
                single_assign = new_single_assign
            new_domain_dicts.append(single_assign)
        return new_domain_dicts


class CSVDomainPredictor:
    def __init__(self, csv_predictions):
        self.csv_filepath = csv_predictions
        self.predictions = pd.read_csv(csv_predictions)

    def predict(self, x):
        # x should be a chain_id
        one_pred = self.predictions[self.predictions.chain_id == x[0][:5]]
        if len(one_pred) == 0:
            return None
        domain_dicts = [make_domain_mapping_dict(one_pred.iloc[0])]
        return None, domain_dicts