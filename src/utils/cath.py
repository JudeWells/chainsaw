"""
Load / Process CATH metadata.
N.B. Domain splits are based on CATH classification in cath-domain-list.txt
CATH List File (CLF) Format 2.0
-------------------------------
This file format has an entry for each structural entry in CATH.
Column 1:  CATH domain name (seven characters)
Column 2:  Class number
Column 3:  Architecture number
Column 4:  Topology number
Column 5:  Homologous superfamily number
Column 6:  S35 sequence cluster number
Column 7:  S60 sequence cluster number
Column 8:  S95 sequence cluster number
Column 9:  S100 sequence cluster number
Column 10: S100 sequence count number
Column 11: Domain length
Column 12: Structure resolution (Angstroms)
           (999.000 for NMR structures and 1000.000 for obsolete PDB entries)
Numbers within each column are relative to the higher categories.
e.g. topology number 1 is relative to the C and A codes,
S35 sequence cluster is relative to CATH codes.
We *might* want to check that no chains in this set contain domains outside of the
nonredundant set on which redundancy control is based.
If it's the same pipeline as Ingraham, we take CHAINS to which non-redundant domains
are annotated, then annotate these chains with CATH nodes. This produces a dataset of
CATH nodes. The split procedure occurs on CATH nodes and is random 80/10/10.
Since each chain can contain multiple CAT codes, we first removed any redundant entries 
from train and then from validation. What does this mean?
Finally, we removed any chains from the test set that had CAT overlap with train and 
removed chains from the validation set with CAT overlap to train or test. 
Facebook description is clearer perhaps:
As each chain may be classified with more than one topology codes, we further removed 
chains with topology codes spanning different splits, so that there is no overlap in 
topology codes between train, validation, and test. This results in 16,153 chains in 
the train split, 1457 chains in the validation split, and 1797 chains in the test split.
https://github.com/jingraham/neurips19-graph-protein-design/blob/master/data/build_chain_dataset.py
"""
import time
from dataclasses import dataclass
import itertools
import json
import os
import numpy as np
import pandas as pd
import torch
import urllib
import logging

from src.utils.torch_utils import get_ids
from src.utils import pdbio

LOG = logging.getLogger(__name__)

ESM_CATH_SPLIT = "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
DOMAIN_BOUNDARIES = "data/cath/cath-domain-boundaries.txt"   # annotations of domains to chains
DOMAIN_CLASSIFICATIONS = "data/cath/cath-domain-list.txt"  # classification of domains
CATH_PDB_URL = "http://www.cathdb.info/version/v4_3_0/api/rest/id/{}.pdb"


CLASSIFICATION_COLS = [
    "domain_name",
    "class",
    "architecture",
    "topology",
    "superfamily",
    "S35_cluster",
    "S60_cluster",
    "S95_cluster",
    "S100_cluster",
    "S100_count",
    "length",
    "resolution",
]


@dataclass
class DomainAnnotation:

    segments: list
    
    def extract(self, seq, start_index=0, graceful=False):
        domain_seq = ""
        for seg in self.segments:
            domain_seq += seg.extract(seq)
        return domain_seq

    def start(self, start_index=0):
        slices = self.get_slices(start_index=start_index)
        return slices[0].start

    def end(self, start_index=0):
        slices = self.get_slices(start_index=start_index)
        return slices[-1].stop
    
    def slice_tokens(self, tokens, start_index=0, graceful=False):
        slices = self.get_slices(start_index=start_index)
        if not graceful:
            for _slice in slices:
                assert _slice.start <= tokens.shape[-1] and _slice.stop <= tokens.shape[-1]
        token_slices = [tokens[..., sl] for sl in slices]
        if torch.is_tensor(tokens):
            return torch.cat(token_slices, dim=-1)
        else:
            return np.concatenate(token_slices, -1)

    def get_slices(self, start_index=0):
        slices = []
        for seg in self.segments:
            slices.append(seg.get_slice(start_index=start_index))  
        return slices

    def get_mask(self, L, start_index=0):
        """L is length of full chain."""
        mask = np.zeros(L)
        for seg in self.get_slices(start_index=start_index):
            mask[seg] = 1
        return mask.astype(bool)

    def __len__(self):
        raise NotImplementedError()


@dataclass
class SegmentAnnotation:
    
    start: int
    end: int
    chain: str = "A"

    def get_slice(self, start_index=0):
        return slice(self.start-start_index, self.end-start_index+1)  # + 1 b.c. cath numbering inclusive

    def extract(self, seq, start_index=0, graceful=False):
        _slice = self.get_slice()
        if not graceful:
            assert _slice.start <= len(seq) and _slice.stop <= len(seq)
        return seq[_slice]


class NumberedSequence:

    """Since CATH uses PDB numbering which isn't guaranteed to behave sensibly,
    (residue numbers don't even have to be numbers, we can get numberings like
    100,100A,100B etc.: see 12e8H), we need to define our own slicing interface
    that is compatible with CATH numbering.
    1c12B is another one where I seem to have a serious issue (the sequence is much to short)
    6gmeB has no actual domain annotated to it for some reason - many others like this.
    """

    def __init__(self, sequence, residue_numbers, chain=None, label=None):
        assert len(sequence) == len(
            residue_numbers
        ), f"Seq len {len(sequence)} doesn't equal numbers len {len(residue_numbers)}"
        self.sequence = sequence
        self.residue_numbers = residue_numbers
        self.chain = chain
        self.label = label

    def resnum2seqnum(self, resnum):
        # n.b. this doesnt handle duplicates properly
        return self.residue_numbers.index(resnum)

    def __str__(self):
        return self.sequence

    def resseg2seqseg(self, res_seg):
        """Convert annotated residue numbers in segment annotation to numbers relative
        to sequence.
        n.b. since we are dealing with a segment annotation, we maintain CATH numbering
        conventions.
        """
        seq_start = self.resnum2seqnum(res_seg.start)
        seq_end = self.resnum2seqnum(res_seg.end)
        seq_seg = SegmentAnnotation(seq_start, seq_end, chain=res_seg.chain)
        return seq_seg

    def __getitem__(self, key):
        if isinstance(key, slice):
            seq_slice = self.resslice2seqslice(key)
            return self.sequence[seq_slice]
        elif isinstance(key, int):
            seq_ind = self.resnum2seqnum(key)
            return self.sequence[seq_ind]
        else:
            raise ValueError()


def domall_iterator(domall_file, max_lines=None, min_domains=0):
    """Parse domall file, e.g. cath-domain-boundaries.txt.
    This file is useful because against each PDB id it lists all domains
    associated with that PDB id.
    """
    with open(domall_file, "r") as f:
        for line in itertools.islice(f, max_lines or 1000000):
            if line.startswith("#"):
                continue
            else:
                fields = line.rstrip("\n").split()
                n_domains = int(fields[1][1:])
                if n_domains >= min_domains:
                    d = {"pdb": fields[0], "n_domains": n_domains, "domains": []}
                    col_start = 3
                    segment_cols = 6
                    for domain in range(n_domains):
                        n_segments = int(fields[col_start])
                        segments = []
                        for segment in range(n_segments):
                            segment_fields = fields[
                                col_start+1+segment*segment_cols:col_start+1+(segment+1)*segment_cols
                            ]
                            assert segment_fields[0] == segment_fields[3]
                            segment = SegmentAnnotation(
                                int(segment_fields[1]),
                                int(segment_fields[4]),
                                chain=segment_fields[0],
                            )
                            segments.append(segment)

                        d["domains"].append(DomainAnnotation(segments))
                        col_start += 1 + segment_cols*n_segments

                    yield d


def parse_domall(domall_file, max_lines=None, min_domains=0):
    return [d for d in domall_iterator(domall_file, max_lines=max_lines, min_domains=min_domains)]


def chain_annotations():
    return parse_domall(DOMAIN_BOUNDARIES)


def cath_split_ids(
    splits_file,
    exclude_missing_feats=False,
    feature_dir="features/2d_features",
    label_dir="features/pairwise",
):
    with open(splits_file, "r") as f:
        d = json.load(f)

    if exclude_missing_feats:
        featurised_chain_ids = set([c.replace(".npz", "") for c in get_ids(feature_dir=feature_dir, label_dir=label_dir)])
        LOG.info(list(featurised_chain_ids)[:10])

    featurised_split_ids = {}
    msg = ""
    for split_name, split_ids in d.items():
        split_size = len(split_ids)
        split_ids = [c.replace(".", "") for c in split_ids]
        if exclude_missing_feats:
            split_ids = list(set(split_ids).intersection(featurised_chain_ids))
            msg += f"Split {split_name}: {len(split_ids)} of {split_size}\t"
        else:
            msg += f"Split {split_name}: {split_size}\t"

        # we get random orderings irrespective of seeding unless we do this.
        featurised_split_ids[split_name] = sorted(split_ids)

    LOG.info(msg)

    return featurised_split_ids


def domain_classification():
    """See 'how does the numbering in the CATH classification work?' on website:

    http://cathdb.info/wiki/doku/?id=faq

    and the README.md file for domain-list
    """
    records = []
    with open(DOMAIN_CLASSIFICATIONS, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] != "#":
                row = {col: v for col, v in zip(CLASSIFICATION_COLS, line.split())}
                records.append(row)

    df = pd.DataFrame.from_records(records)
    df["chain_id"] = df["domain_name"].str[:-2]
    df["CAT"] = df["class"] + "." + df["architecture"] + "." + df["topology"]
    df["CATH"] = df["CAT"] + "." + df["superfamily"]
    df["S35"] = df["CATH"] + "." + df["S35_cluster"]
    df["S60"] = df["S35"] + "." + df["S60_cluster"]
    return df


def chain_classification():
    cath_chain_df_path = 'data/cath_chain_topology_class.pickle'
    if os.path.exists(cath_chain_df_path):
        chain_df = pd.read_pickle(cath_chain_df_path)
    else:
        dom_df = domain_classification()
        chain_df = dom_df.groupby("chain_id").agg(
                {
                    'CAT': lambda x: tuple(x),
                    'CATH': lambda x: tuple(x),
                    'S35': lambda x: tuple(x),
                    'S60': lambda x: tuple(x),
                    'domain_name': "count",
                }
            ).reset_index().rename({"domain_name": "n_domains"}, axis=1)
        chain_df["n_CAT"] = chain_df["CAT"].apply(lambda x: len(x))
        chain_df["n_CATH"] = chain_df["CATH"].apply(lambda x: len(x))
        chain_df["n_S35"] = chain_df["S35"].apply(lambda x: len(x))
        chain_df["n_S60"] = chain_df["S60"].apply(lambda x: len(x))
        chain_df.to_pickle(cath_chain_df_path)

    chain_df["pdb_id"] = chain_df["chain_id"].apply(lambda x: x[:-1])
    chain_df["chain_code"] = chain_df["chain_id"].apply(lambda x: x[-1])
    # note storing s60 as sorted in GENERAL is risky because it breaks 
    # identification of individual domains with clusters.
    chain_df["S60_comb"] = chain_df["S60"].apply(lambda x: tuple(sorted(x)))

    return chain_df


def make_splits_df(splits_file):
    chain_df = chain_classification()
    splits = cath_split_ids(splits_file)
    assert len(splits.keys()) == 3, "Can't handle non-overlapping splits"

    chain2split = {c: s for s, chains in splits.items() for c in chains}
    chain_df["split"] = chain_df["chain_id"].apply(lambda x: chain2split.get(x, ""))
    return chain_df[chain_df["split"]!=""].reset_index(drop=True).copy()


def esmif_chain_splits_df():
    """Load a dataframe containing chain ids, topology information and split information
    for chains within the ESM-IF CATH splits.
    """
    return make_splits_df("splits.json")


def renumber_domains(chain_seq: NumberedSequence, domain_annotations):
    """Given a list of domains in CATH/PDB numbering,
    convert to canonical numbering relative to the residues in chain_seq.
    """
    renumbered_domains = []
    for domain in domain_annotations:
        renumbered_segments = []
        for segment in domain.segments:
            # convert numbering from PDB numbering to standard, 0-based sequence numbering
            # this is important because PDB numbering can be a complete mess, including e.g.
            # 4 residues with number 100

            # n.b. the below will raise exceptions, e.g. 1ddqD first domain fails because of
            # UNK at resnum 4 which is not parsed by SeqIO
            seqnum_segment = chain_seq.resseg2seqseg(segment)
            renumbered_segments.append(seqnum_segment)
        renumbered_domains.append(DomainAnnotation(segments=renumbered_segments))

    return renumbered_domains


def pdbatom2seq(pdbfile, fill_gaps=False, filter_residues=False):
    """n.b. relies on custom pdbatomiterator in pdbio to return residue numbers."""
    with open(pdbfile) as handle:
        records = [
            r
            for r in pdbio.PdbAtomIterator(
                handle, fill_gaps=fill_gaps, filter_residues=filter_residues
            )
        ]
    assert len(records) == 1
    return NumberedSequence(str(records[0].seq), records[0].annotations["residues"])


def domains_from_label(label):
    """Where I have mapped domains into sequence numbering using scripts/cath/get_cath_pdbs,
    then the next step is to load the domain annotations as here.
    """
    components = label.split("|")
    pdb = components[0].split("_")[0]
    chain = pdb[-1]
    domain_annotations = components[1:]
    annotated_domains = []
    for domain_annotation in domain_annotations:
        segments = []
        try:
            segment_annotations = domain_annotation.split("_")[-1].split(",")
            for segment_annotation in segment_annotations:
                seg_start, seg_end = segment_annotation.split("-")
                segments.append(SegmentAnnotation(int(seg_start), int(seg_end), chain=chain))
            annotated_domains.append(DomainAnnotation(segments))
        except Exception as e:
            # label_issues.append(label)
            raise e
    return annotated_domains


def download_extract_chain_sequence(pdbid, pdb_dir="/tmp/cath", cleanup=False):
    os.makedirs(pdb_dir, exist_ok=True)
    url = CATH_PDB_URL.format(pdbid)
    filepath = os.path.join(pdb_dir, f"{pdbid}.pdb")
    max_retries = 5
    n_retries = 0
    while n_retries < max_retries:
        try:
            # urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name
            # resolution>
            urllib.request.urlretrieve(url, filename=filepath)
            break
        except Exception:
            time.sleep(2)
            n_retries += 1

    # download_file(url, filepath)
    seqrecord = pdbatom2seq(filepath)
    if cleanup:
        os.remove(filepath)
    return seqrecord
