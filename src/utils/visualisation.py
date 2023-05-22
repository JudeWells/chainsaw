import itertools
import os
import numpy as np
import torch
import wandb
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

from src.data import TrainDataset
from src.evaluation.ScoreNDO import ndo_score


def get_cluster_masks(cluster_inds, ignore_label=None):
    n_clust = max(cluster_inds) + 1
    return [cluster_inds==i for i in range(n_clust) if i != ignore_label]


def get_domdict_masks(domain_dict, L):
    domain_masks = []
    for dom_name, dom_inds in domain_dict.items():
        if dom_name != "linker":
            dom_mask = np.zeros(L)
            dom_mask[dom_inds] = 1
            domain_masks.append(dom_mask.astype(bool))
    return domain_masks


def pairwise_plot_with_domain_boxes(pair_vals, domain_masks=None, figsize=(10,10), linewidth=4, ax=None):
    """
    N.B. there is an issue with the visualisation not quite lining up
    which may be because lines / vertices aren't centred on relevant coordinates.

    ref https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    """
    
    if ax is None:
        plt.figure(figsize=figsize)

        # Display the image
        plt.imshow(pair_vals)

        # Get the current reference

        ax = plt.gca()

    else:
        ax.imshow(pair_vals)
    
    # https://matplotlib.org/stable/gallery/color/color_cycle_default.html
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    for dom_ix, domain_mask in enumerate(domain_masks):
        # https://stackoverflow.com/questions/50465162/numpy-find-indeces-of-mask-edges
        dom_slices = np.ma.clump_masked(np.ma.masked_where(domain_mask, domain_mask))
        # If a domain only has a single continuous segment, then the bounding
        # box is entirely on-diagonal. If not, there is an on diagonal and off-diagonal
        # part
        
        # On-diagonal part (continuous 'segment' boxes)
        for dom_slice in dom_slices:
            L = dom_slice.stop - dom_slice.start
            # print(dom_slice.start, dom_slice.stop)
            # print(dom_slice.stop, dom_slice.start)
            rect = Rectangle(
                (dom_slice.start,dom_slice.start),L,L,
                linewidth=linewidth,
                edgecolor=colours[dom_ix % len(colours)],
                facecolor="none"
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
        
        
        # Off-diagonal part
        if len(dom_slices) > 1:
            for segment_pair in itertools.combinations(dom_slices, 2):
                slice_A = segment_pair[0]
                slice_B = segment_pair[1]
                L_A = slice_A.stop - slice_A.start
                L_B = slice_B.stop - slice_B.start
                rect_AB = Rectangle(
                    (slice_A.start, slice_B.start),L_A,L_B,
                    linewidth=linewidth,
                    edgecolor=colours[dom_ix % len(colours)],
                    facecolor="none",
                )
                
                rect_BA = Rectangle(
                    (slice_B.start, slice_A.start),L_B,L_A,
                    linewidth=linewidth,
                    edgecolor=colours[dom_ix % len(colours)],
                    facecolor="none",
                )
                
                ax.add_patch(rect_AB)
                ax.add_patch(rect_BA)


def wandb_visualise_cropped_pairwise_preds(learner, chain_ids, epoch, crop_size=256, **data_kwargs):
    """Cropped reconstructions (train or test set)."""
    # label_names = [
    #     # '5cebB', '3sp8B', '1y8qD', '2o3oH', '2lvsA', '3hkzA' # old version label names
    #     '5kqqA',  '4yqcA', '4a35A',  '3tszA', '3omxA'
    # ]
    dataset = TrainDataset(chain_ids, crop_size=crop_size, feature_dir=data_kwargs['feature_dir'],
                           label_dir=data_kwargs['label_dir'], has_pae=data_kwargs['has_pae'],
                           chains_csv=data_kwargs['chains_csv'],
                           add_sequence_separation=data_kwargs["add_sequence_separation"],
                           sequence_separation_onehot=data_kwargs["sequence_separation_onehot"],
                           add_sequence=data_kwargs["add_sequence"],
                           )
    generator = torch.utils.data.DataLoader(dataset, batch_size=len(chain_ids), shuffle=False)

    x, y = next(iter(generator))
    with torch.no_grad():
        results = learner.predict_pairwise(x)
        mask = learner.get_mask(x)
        losses, _ = learner.compute_loss(results, y, mask=mask, batch_average=False)

    results = results.cpu().numpy()
    losses = losses.cpu().numpy()
    y = y.cpu().numpy()
    fig, axs = plt.subplots(4, max(len(chain_ids), 2)) # this dimension must be minimum 2 even when the batch size is 1
    fig.suptitle(f"Epoch: {epoch}", fontsize=14)
    axs[0, 0].set_ylabel('True Labels')
    axs[3, 0].set_ylabel('Pred Labels')
    for i, (pred, loss, inp, labels, chain_id) in enumerate(zip(results, losses, x, y, chain_ids)):
        axs[0, i].imshow(labels, vmin=0, vmax=1)
        axs[0, i].xaxis.set_ticklabels([])
        axs[0, i].yaxis.set_ticklabels([])

        # x is b, channels, L, L
        axs[1, i].xaxis.set_ticklabels([])
        axs[1, i].yaxis.set_ticklabels([])
        pairwise_plot_with_domain_boxes(inp[0], domain_masks=[mask[i].cpu().numpy().astype(bool)], ax=axs[1, i])

        axs[2, i].xaxis.set_ticklabels([])
        axs[2, i].yaxis.set_ticklabels([])
        axs[2, i].imshow(inp[1])

        axs[3, i].xaxis.set_ticklabels([])
        axs[3, i].yaxis.set_ticklabels([])
        axs[3, i].imshow(pred)

        axs[0, i].set_title(f"{chain_id}_{loss:.3f}", fontsize=10)

    plt.savefig('progress.png')
    w_image = wandb.data_types.Image('progress.png')
    wandb.log({"reconstruction": w_image, "epoch": epoch})
    plt.close()


def wandb_visualise_domain_preds(learner, dataset, epoch, prefix=""):
    """No cropping, no padding, visualise domain boundaries (train or test set)."""
    fig, axs = plt.subplots(4, max(len(dataset), 2)) # this dimension must be minimum 2 even when the batch size is 1
    fig.suptitle(f"Epoch: {epoch}", fontsize=14)
    axs[0, 0].set_ylabel('True Labels')
    axs[3, 0].set_ylabel('Pred Labels')
    for i in range(len(dataset)):
        chain_id = dataset.chain_ids[i]
        x, y, domains, boundaries = dataset[i]
        x, y = x[None], y[None]  # c.f. DomainPredictionEvaluator
        with torch.no_grad():
            pair_pred, dom_pred, uncertainty = learner.predict(x)
            # note - no padding this time
            pair_loss, _ = learner.compute_loss(pair_pred, y, batch_average=False)
            pair_loss, pair_pred, dom_pred = pair_loss[0].item(), pair_pred[0].cpu().numpy(), dom_pred[0]  # remove batch dim
            # TODO compute MSE
            ndo = ndo_score(dom_pred, domains)
        if len(y.shape) > 2:
            x, y = x[0], y[0]  # remove batch dim

        axs[0, i].imshow(y, vmin=0, vmax=1)
        axs[0, i].xaxis.set_ticklabels([])
        axs[0, i].yaxis.set_ticklabels([])

        # x is b, channels, L, L
        axs[1, i].xaxis.set_ticklabels([])
        axs[1, i].yaxis.set_ticklabels([])
        axs[1, i].imshow(x[0])

        axs[2, i].xaxis.set_ticklabels([])
        axs[2, i].yaxis.set_ticklabels([])
        axs[2, i].imshow(x[1])

        # now we need to just extract tuples of ranges and pass to pairwise plot with domain boxes.
        axs[3, i].xaxis.set_ticklabels([])
        axs[3, i].yaxis.set_ticklabels([])

        # TODO make this take inds as input
        pairwise_plot_with_domain_boxes(pair_pred, get_domdict_masks(dom_pred, L=pair_pred.shape[-1]), ax=axs[3, i])

        axs[0, i].set_title(f"{chain_id}_{pair_loss:.3f}_{ndo:.3f}", fontsize=8)

    plt.savefig(f'{prefix}dompred_progress.png')
    w_image = wandb.data_types.Image(f'{prefix}dompred_progress.png')
    wandb.log({f"{prefix}dompred": w_image, "epoch": epoch})
    plt.close()

