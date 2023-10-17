import numpy as np

from src import featurisers
from src.domain_assignment.util import convert_domain_dict_strings
from src.utils.secondary_structure import make_ss_matrix


def centered_rolling_mean(arr, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd for a centered rolling mean.")

    # Create a window of ones of shape window_size
    window = np.ones(window_size) / window_size

    # Compute the rolling sum using convolution
    rolling_mean = np.convolve(arr, window, mode="same")

    return rolling_mean


class DPAMPostProcessor:

    """To be considered: DPAM uses PAE to define disordered residues.

    https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4548?saml_referrer
    """

    pass


class BasePostProcessor:
    def __init__(self, min_domain_length=30):
        self.min_domain_length = min_domain_length

    def remove_domains_with_short_length(self, domain_dict):
        """
        Remove domains where length is less than minimum
        """
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue

            if len(res) < self.min_domain_length:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict

    def domain_dict_to_chopping_str(self, domain_dict):
        # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
        domain_dict = {k: [r + 1 for r in v] for k, v in domain_dict.items()}
        names_str, bounds_str = convert_domain_dict_strings(domain_dict)

        if names_str == "":
            names = bounds = ()
        else:
            names = names_str.split("|")
            bounds = bounds_str.split("|")

        assert len(names) == len(bounds)

        # gather choppings into segments in domains
        chopping_segs_by_domain = {}
        for domain_id, chopping in zip(names, bounds):
            if domain_id not in chopping_segs_by_domain:
                chopping_segs_by_domain[domain_id] = []
            chopping_segs_by_domain[domain_id].append(chopping)

        # convert list of segments "start-end" into chopping string for the domain
        # (join distontiguous segs with "_")
        chopping_str_by_domain = {
            domid: "_".join(segs) for domid, segs in chopping_segs_by_domain.items()
        }

        # sort domain choppings by the start residue in first segment
        sorted_domain_chopping_strs = sorted(
            chopping_str_by_domain.values(), key=lambda x: int(x.split("-")[0])
        )

        # convert to string (join domains with ",")
        chopping_str = ",".join(sorted_domain_chopping_strs)

        num_domains = len(chopping_str_by_domain)
        if num_domains == 0:
            chopping_str = None

        return chopping_str


class PLDDTWindowPostProcessor(BasePostProcessor):
    """Look at pLDDT within local window

    We want the window to be centred, since then if a residue together with
    its neighbours in either direction are above the threshold then the residue is
    above the threshold.

    We probably want to implement a criterion like if the running plddt average is
    below a certain value we trim (for boundaries)

    And if the domain average is below a certain value we remove outright.

    N.B. assumes zero indexing

    TO TEST THIS, WE CAN CREATE A DUMMY DOMAIN_DICT, WHICH ASSIGNS EVERYTHING TO
    A SINGLE DOMAIN, AND SEE WHAT WE GET OUT.
    """

    def __init__(
        self, min_domain_length=30, window_size=5, strategy="mean", threshold=60
    ):
        super().__init__(min_domain_length=min_domain_length)
        self.window_size = window_size
        self.threshold = threshold

    def post_process(
        self, domain_dict, pdb_path, chain, ss_path=None, model_structure=None
    ):
        domain_dict = {k: list(v) for k, v in domain_dict.items()}
        if model_structure is None:
            model_structure = featurisers.get_model_structure(pdb_path)

        plddts = featurisers.extract_plddts(model_structure, chain)
        domain_dict = self.remove_disordered_domains(domain_dict, plddts)

        if self.min_domain_length > 0:
            domain_dict = self.remove_domains_with_short_length(domain_dict)

        return domain_dict

    def remove_disordered_domains(self, domain_dict, plddts):
        """Get rid of domains with insufficient secondary structure percentage"""
        windowed_plddts = centered_rolling_mean(plddts, self.window_size)
        plddt_linker_residues = np.argwhere(windowed_plddts < self.threshold)
        # TODO: consider adding option that avoids the introduction of discontinuities
        new_domain_dict = {}
        linker_res = domain_dict.get("linker", [])
        new_linker = np.setdiff1d(plddt_linker_residues, linker_res)
        new_domain_dict["linker"] = list(set(linker_res).union(set(new_linker)))
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            if len(res) == 0:
                continue

            updated_res = np.setdiff1d(res, plddt_linker_residues)
            if len(updated_res) > 0:
                new_domain_dict[dname] = list(updated_res)

        return new_domain_dict


class SSPostProcessor(BasePostProcessor):
    def __init__(
        self,
        min_ss_components=2,
        min_domain_length=30,
        remove_disordered_domain_threshold=0,
        trim_each_domain=True,
    ):
        self.remove_disordered_domain_threshold = remove_disordered_domain_threshold
        self.trim_each_domain = trim_each_domain
        self.min_domain_length = min_domain_length
        self.min_ss_components = min_ss_components

    def post_process(self, domain_dict, helix, sheet):
        x = x.cpu().numpy()
        domain_dict = {k: list(v) for k, v in domain_dict.items()}
        diag_helix = np.diagonal(helix)
        diag_sheet = np.diagonal(sheet)
        ss_residues = list(np.where(diag_helix > 0)[0]) + list(np.where(diag_sheet > 0)[0])

        domain_dict = self.trim_disordered_boundaries(domain_dict, ss_residues)

        if self.remove_disordered_domain_threshold > 0:
            domain_dict = self.remove_disordered_domains(domain_dict, ss_residues)

        if self.min_ss_components > 0:
            domain_dict = self.remove_domains_with_few_ss_components(domain_dict, helix, sheet)

        if self.min_domain_length > 0:
            domain_dict = self.remove_domains_with_short_length(domain_dict)

        return domain_dict

    def trim_disordered_boundaries(self, domain_dict, ss_residues):
        """Get rid of parts of domain not belonging to secondary structure components"""
        if not self.trim_each_domain:
            start = min(ss_residues)
            end = max(ss_residues)
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            if self.trim_each_domain:
                domain_specific_ss = set(ss_residues).intersection(set(res))
                if len(domain_specific_ss) == 0:
                    continue
                start = min(domain_specific_ss)
                end = max(domain_specific_ss)
            domain_dict["linker"] += [r for r in res if r < start or r > end]
            domain_dict[dname] = [r for r in res if r >= start and r <= end]
        return domain_dict

    def remove_disordered_domains(self, domain_dict, ss_residues):
        """Get rid of domains with insufficient secondary structure percentage"""
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            if len(res) == 0:
                continue
            if len(set(res).intersection(set(ss_residues))) / len(res) < self.remove_disordered_domain_threshold:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict

    def remove_domains_with_few_ss_components(self, domain_dict, helix, strand):
        """
        Remove domains where number of ss components is less than minimum
        eg if self.min_ss_components=2 domains made of only a single helix or sheet are removed
        achieve this by counting the number of unique string hashes in domain rows of x
        """
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue
            res = sorted(res)
            if hasattr(self, "ss_mod") and self.ss_mod:
                raise NotImplementedError("Unclear why we don't just use the same method?")
                # ss_mod features have value 2 on the boundary of the ss component
                helix_boundary_diag = np.diagonal(x[3][res,:][:,res])
                strand_boundary_diag = np.diagonal(x[4][res,:][:,res])
                helix_boundaries = sum(helix_boundary_diag == 1)
                sheet_boundaries = sum(strand_boundary_diag == 1)
                d_start = min(res)
                d_end = max(res)
                # adjust for cases where domain split occurrs within a single ss component
                if x[1, d_start, d_start] == 1 and helix_boundary_diag[0] == 0:
                    helix_boundaries += 1
                if x[2, d_start, d_start] == 1 and strand_boundary_diag[0] == 0:
                    sheet_boundaries += 1
                if x[1, d_end, d_end] == 1 and helix_boundary_diag[-1] == 0:
                    helix_boundaries += 1
                if x[2, d_end, d_end] == 1 and strand_boundary_diag[-1] == 0:
                    sheet_boundaries += 1
                n_helix = helix_boundaries / 2
                n_sheet = sheet_boundaries / 2
            else:
                helix = helix[res, :][:, res]
                strand = strand[res, :][:, res]
                helix = helix[np.any(helix, axis=1)]
                strand = strand[np.any(strand, axis=1)]
                n_helix = len(set(["".join([str(int(i)) for i in row]) for row in helix]))
                n_sheet = len(set(["".join([str(int(i)) for i in row]) for row in strand]))
            if len(res) == 0:
                continue
            if n_helix + n_sheet < self.min_ss_components:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict

    def remove_domains_with_short_length(self, domain_dict):
        """
        Remove domains where length is less than minimum
        """
        new_domain_dict = {}
        for dname, res in domain_dict.items():
            if dname == "linker":
                continue

            if len(res) < self.min_domain_length:
                domain_dict["linker"] += res
            else:
                new_domain_dict[dname] = res
        new_domain_dict["linker"] = domain_dict["linker"]
        return new_domain_dict

    def domain_dict_to_chopping_str(self, domain_dict):
         # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
        domain_dict = {k: [r + 1 for r in v] for k, v in domain_dict.items()}
        names_str, bounds_str = convert_domain_dict_strings(domain_dict)

        if names_str == "":
            names = bounds = ()
        else:
            names = names_str.split('|')
            bounds = bounds_str.split('|')

        assert len(names) == len(bounds)

        # gather choppings into segments in domains
        chopping_segs_by_domain = {}
        for domain_id, chopping in zip(names, bounds):
            if domain_id not in chopping_segs_by_domain:
                chopping_segs_by_domain[domain_id] = []
            chopping_segs_by_domain[domain_id].append(chopping)

        # convert list of segments "start-end" into chopping string for the domain 
        # (join distontiguous segs with "_")
        chopping_str_by_domain = {domid: '_'.join(segs) for domid, segs in chopping_segs_by_domain.items()}

        # sort domain choppings by the start residue in first segment
        sorted_domain_chopping_strs = sorted(chopping_str_by_domain.values(), key=lambda x: int(x.split('-')[0]))

        # convert to string (join domains with ",")
        chopping_str = ','.join(sorted_domain_chopping_strs)

        num_domains = len(chopping_str_by_domain)
        if num_domains == 0:
            chopping_str = None

        return chopping_str
