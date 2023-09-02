"""Wrapper for unidoc.

Should produce same objects that chainsaw produces.
"""
import os
import hashlib
from src import featurisers
from src.post_processors import SSPostProcessor
from src.utils.secondary_structure import make_ss_matrix


# these functions are minor modifications from the python script in the unidoc source
def caculate_ss(pdbfile, chain, stride_executable, outdir='.'):
    binpath = os.path.join(bindir, 'stride')
    assert os.path.exists(pdbfile)
    output_path = os.path.join(outdir, 'pdb_ss')
    print(f"Running command: {'%s %s -r%s>%s' % (stride_executable, pdbfile, chain, output_path)}")
    return os.system('%s %s -r%s>%s' % (stride_executable, pdbfile, chain, output_path))


def parse_domain(pdbfile, chain, unidoc_executable, outfile, outdir='.'):
    ss_path = os.path.join(outdir, 'pdb_ss')
    output_path = os.path.join(outdir, out)
    print(f"Running command: {'%s %s %s %s > %s' % (unidoc_executable, pdbfile, chain, ss_path, output_path)}")
    return os.system('%s %s %s %s > %s' % (unidoc_executable, pdbfile, chain, ss_path, output_path))


# c.f. domdet evaluate_unidoc_preds/benchmark.benchmark 
def load_predictions(output_file, convert_to_one_based=True):
    with open(, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 1
    pred_str = lines[0].strip()
    dnames = []
    dom_bounds = []
    for i, dbounds in enumerate(pred_str.split('/')):
        for dseg in dbounds.split(','):
            dnames.append(f"d{i+1}")
            dom_bounds.append(dseg.replace('~', '-'))
    if convert_to_one_based:
        new_bounds = []
        for segment_bounds in dom_bounds:
            start, end = segment_bounds.split("-")
            new_bounds.append(f"{int(start)+1}-{int(end)+1}")
        dom_bounds = new_bounds
    bounds_str = '|'.join(dom_bounds)
    name_str = '|'.join(dnames)
    return {
        'dom_bounds_pdb_ix': bounds_str,
        'dom_names': name_str,
        'n_domains': len(set(dnames)),
    }


def convert_limits_to_numbers(dom_limit_list):
    processed_dom_limit_list = []
    for lim in dom_limit_list:
        dash_idx = [i for i, char in enumerate(lim) if char == '-']
        if len(dash_idx) == 1:
            start_index = int(lim.split('-')[0]) -1
            end_index = int(lim.split('-')[1])
        else:
            raise ValueError('Invalid format for domain limits', str(dom_limit_list))
        processed_dom_limit_list.append((start_index, end_index))
    return processed_dom_limit_list


def resolve_residue_in_multiple_domain(mapping, shared_res):
    """
    This is a stupid slow recursive solution: but I think it only applies to one
    case so going to leave it for now
    """
    for one_shared in shared_res:
        for domain, res in mapping.items():
            if one_shared in res:
                mapping[domain].remove(one_shared)
                return check_no_residue_in_multiple_domains(mapping)


def check_no_residue_in_multiple_domains(mapping, resolve_conflics=True):
    # ensures no residue index is associated with more than one domain
    for dom, res in mapping.items():
        for dom2, res2 in mapping.items():
            if dom == dom2:
                continue
            shared_res = set(res).intersection(set(res2))
            if len(shared_res):
                print(f'Found {len(shared_res)} shared residues')
                if resolve_conflics:
                    mapping = resolve_residue_in_multiple_domain(mapping, shared_res)
                else:
                    raise ValueError("SAME RESIDUE NUMBER FOUND IN MULTIPLE DOMAINS")
    return mapping


def make_domain_mapping_dict(d):
    dom_limit_list = d["dom_bounds_pdb_ix"].split('|')
    dom_names = d["dom_names"].split('|')
    dom_limit_list = convert_limits_to_numbers(dom_limit_list)
    dom_limit_array, dom_names = sort_domain_limits(dom_limit_list, dom_names)
    mapping = {}

    for i, d_lims in enumerate(dom_limit_array):
        dom_name = dom_names[i]
        pdb_start, pdb_end = d_lims
        if dom_name not in mapping:
            mapping[dom_name] = []
        mapping[dom_name] += list(range(pdb_start, pdb_end))
    check_no_residue_in_multiple_domains(mapping)
    return mapping


class UniDoc:

    """To improve UniDoc's suitability for use with AF we add the
    same ss and domain size filtering options used by Chainsaw.
    """

    def __init__(
        self,
        output_directory,
        stride_executable="stride",
        unidoc_executable="UniDoc_structure",
        cleanup=False,
        min_ss_components=None,
        min_domain_length=None,
        remove_disordered_domain_threshold=0,  # TODO check what defaults should be to ensure unidoc defaults
        trim_each_domain=True,
    ):
        self.output_directory = output_directory
        self.cleanup = cleanup
        self.stride_executable = stride_executable
        self.unidoc_executable = unidoc_executable
        self.post_processor = SSPostProcessor(
            min_ss_components=min_ss_components,
            min_domain_length=min_domain_length,
            remove_disordered_domain_threshold=remove_disordered_domain_threshold,
            trim_each_domain=trim_each_domain,
        )

    def domain_dict_from_preds(self, preds):
        raise NotImplementedError()

    def predict(self, pdb_path, chain_id):
        # to re-use secondary structure post-processing as written,
        # we also have to generate secondary structure features in same
        # way that chainsaw does...

        # we can use inference_time_create_features directly to do this,
        # this however unnecessarily calls stride a second time and
        # also unnecessarily loads the structure from the pdb.
        pdb_id = f"{os.path.basename(pdb_path)}{chain_id}"
        output_directory = os.path.join(self.output_directory, pdb_id)
        calculate_ss(pdb_path, chain_id, self.stride_executable, outdir=output_directory)
        parse_domain(pdb_path, chain_id, self.unidoc_executable, "unidoc_out.txt", outdir=output_directory)
        output_file = os.path.join(output_directory, "unidoc_out.txt")
        ss_file = os.path.join(output_directory, "pdb_ss")
        preds = load_predictions(output_file)
        domain_dict = make_domain_mapping_dict(preds)  # from domdet make_2d_features (c.f. benchmark.py)

        # TODO: find a way around this dist matrix loading (just used to get nres)
        model_structure = featurisers.get_model_structure(pdb_path)
        model_structure_seq = featurisers.get_model_structure_sequence(model_structure, chain=chain_id)
        model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()
        dist_matrix = featurisers.get_distance(model_structure, chain=chain_id)
        n_res = dist_matrix.shape[-1]
        helix, strand = make_ss_matrix(ss_file, nres=n_res)

        domain_dict = self.post_processor.post_process(domain_dict, helix, strand)

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
