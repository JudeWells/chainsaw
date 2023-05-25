"""PDB parsing - to handle the missing residue information in BioPython PDBAtomIterator.

N.B. if I want to modify the PDB atom iterator base class I have to be
careful it also works on cif files.
"""
import warnings

from Bio import BiopythonParserWarning, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import logging
LOG = logging.getLogger(__name__)


warnings.filterwarnings("ignore")


def AtomIterator(pdb_id, structure, fill_gaps=True, filter_residues=True):
    """
    Patches AtomIterator to return residue numbers. And optionally NOT fill iin
    any gaps in the residue numbering (identified via non-consecutive residue numbers)
    with X. Finally we add the option not to filter out non-standard residues.

    When returning residue numbers and filling gaps, we return Nones in the correpsonding
    positions.

    filter_residues is also relevant in structure.as_protein()

    Notes on non_standard amino acids:
    https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format
    Other codes are used for modified amino acids (such as MSE for selenomethionine)
    and for modified nucleotides (such as CBR for bromocytosine)

    c.f. https://biopython.org/docs/dev/api/Bio.SeqIO.PdbIO.html
    > Where amino acids are missing from the structure, as indicated by residue
    numbering, the sequence is filled in with ‘X’ characters to match the size of
    the missing region, and None is included as the corresponding entry in the
    list record.annotations[“residues”].

    Return SeqRecords from Structure objects.
    Base function for sequence parsers that read structures Bio.PDB parsers.
    Once a parser from Bio.PDB has been used to load a structure into a
    Bio.PDB.Structure.Structure object, there is no difference in how the
    sequence parser interprets the residue sequence. The functions in this
    module may be used by SeqIO modules wishing to parse sequences from lists
    of residues.
    Calling functions must pass a Bio.PDB.Structure.Structure object.
    See Bio.SeqIO.PdbIO.PdbAtomIterator and Bio.SeqIO.PdbIO.CifAtomIterator for
    details.
    """
    model = structure[0]
    for chn_id, chain in sorted(model.child_dict.items()):
        # HETATM mod. res. policy: remove mod if in sequence, else discard
        residues = [
            res
            for res in chain.get_unpacked_list()
            if SeqIO.PdbIO._res2aacode(res.get_resname().upper()) != "X" or not filter_residues
        ]
        if not residues:
            continue
        # Identify missing residues in the structure
        # (fill the sequence with 'X' residues in these regions)
        gaps = []
        rnum_annotations = []
        # LOG.info([r.id for r in residues])  ('', rnum, '') each on the CATH ones - weird.
        rnumbers = [r.id[1] for r in residues]

        for i, rnum in enumerate(rnumbers[:-1]):
            if rnumbers[i + 1] != rnum + 1 and rnumbers[i + 1] != rnum:
                # It's a gap!
                gaps.append((i + 1, rnum, rnumbers[i + 1]))

        if gaps and fill_gaps:
            res_out = []
            prev_idx = 0
            for i, pregap, postgap in gaps:
                if postgap > pregap:
                    gapsize = postgap - pregap - 1
                    res_out.extend(SeqIO.PdbIO._res2aacode(x) for x in residues[prev_idx:i])
                    rnum_annotations.extend(rnumbers[prev_idx:i])
                    prev_idx = i
                    res_out.append("X" * gapsize)
                    rnum_annotations += [None] * gapsize
                else:
                    warnings.warn(
                        "Ignoring out-of-order residues after a gap",
                        BiopythonParserWarning,
                    )
                    # Keep the normal part, drop the out-of-order segment
                    # (presumably modified or hetatm residues, e.g. 3BEG)
                    res_out.extend(SeqIO.PdbIO._res2aacode(x) for x in residues[prev_idx:i])
                    rnum_annotations.extend(rnumbers[prev_idx:i])
                    break
            else:
                # Last segment
                res_out.extend([SeqIO.PdbIO._res2aacode(x) for x in residues[prev_idx:]])
                rnum_annotations.extend(rnumbers[prev_idx:])
        else:
            # No gaps or gap filling disabled.
            res_out = [SeqIO.PdbIO._res2aacode(x) for x in residues]
            rnum_annotations = rnumbers

        record_id = f"{pdb_id}:{chn_id}"
        # ENH - model number in SeqRecord id if multiple models?
        # id = "Chain%s" % str(chain.id)
        # if len(structure) > 1 :
        #     id = ("Model%s|" % str(model.id)) + id

        record = SeqRecord(Seq("".join(res_out)), id=record_id, description=record_id)
        # TODO: Test PDB files with DNA and RNA too:
        record.annotations["residues"] = rnum_annotations
        record.annotations["molecule_type"] = "protein"

        record.annotations["model"] = model.id
        record.annotations["chain"] = chain.id

        record.annotations["start"] = int(rnumbers[0])
        record.annotations["end"] = int(rnumbers[-1])
        yield record


def PdbAtomIterator(source, fill_gaps=True, filter_residues=True):
    """Return SeqRecord objects for each chain in a PDB file.
    Argument source is a file-like object or a path to a file.
    The sequences are derived from the 3D structure (ATOM records), not the
    SEQRES lines in the PDB file header.
    Unrecognised three letter amino acid codes (e.g. "CSD") from HETATM entries
    are converted to "X" in the sequence.
    In addition to information from the PDB header (which is the same for all
    records), the following chain specific information is placed in the
    annotation:
    record.annotations["residues"] = List of residue ID strings
    record.annotations["chain"] = Chain ID (typically A, B ,...)
    record.annotations["model"] = Model ID (typically zero)
    Where amino acids are missing from the structure, as indicated by residue
    numbering, the sequence is filled in with 'X' characters to match the size
    of the missing region, and  None is included as the corresponding entry in
    the list record.annotations["residues"].
    This function uses the Bio.PDB module to do most of the hard work. The
    annotation information could be improved but this extra parsing should be
    done in parse_pdb_header, not this module.
    This gets called internally via Bio.SeqIO for the atom based interpretation
    of the PDB file format:
    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("PDB/1A8O.pdb", "pdb-atom"):
    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...
    Record id 1A8O:A, chain A
    Equivalently,
    >>> with open("PDB/1A8O.pdb") as handle:
    ...     for record in PdbAtomIterator(handle):
    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...
    Record id 1A8O:A, chain A
    """
    # TODO - Add record.annotations to the doctest, esp the residues (not working?)

    # Only import PDB when needed, to avoid/delay NumPy dependency in SeqIO
    from Bio.PDB import PDBParser

    structure = PDBParser().get_structure(None, source)
    pdb_id = structure.header["idcode"]
    if not pdb_id:
        warnings.warn("'HEADER' line not found; can't determine PDB ID.", BiopythonParserWarning)
        pdb_id = "????"

    for record in AtomIterator(
        pdb_id, structure, fill_gaps=fill_gaps, filter_residues=filter_residues
    ):
        # The PDB header was loaded as a dictionary, so let's reuse it all
        record.annotations.update(structure.header)

        # ENH - add letter annotations -- per-residue info, e.g. numbers

        yield record
