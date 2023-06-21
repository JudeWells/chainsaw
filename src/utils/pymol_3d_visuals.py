import os

import subprocess

import logging
LOG = logging.getLogger(__name__)


def generate_pymol_image(pdb_path, chain, chopping, image_out_path, path_to_script,
                         pymol_executable):
    """
    This function generates a pymol script which describes how one protein should be visualized
    and saved as a png file. Script is then executed via subprocess.

    Delimit separate domains with commas , and discontinuous domains
    with underscores _. Residue ranges separated by hyphens -, e.g. 1-100,101-200_300-340
    :param pdb_path: string which is the path to a .pdb file
    :param chain: target chain
    :param bounds: domain boundaries expressed in residue author numbers in the pdb file  e.g. 1-100,101-200_300-340
    :param path_to_script: where the pymol bash script is saved
    :param pymol_executable: path to pymol executable
    :return: None
    """
    if chain is None:
        chain = "''"
    script_text = f"""
load {pdb_path}, structure_id
hide everything
bg_color white
select target, chain {chain}
as cartoon, target
set ray_opaque_background,1
color white, structure_id\n"""
    color_list = ['green', 'red', 'cyan', 'magenta', 'blue', 'orange', 'dirtyviolet', 'olive',
                  'limon', 'salmon', 'deepteal', 'yellow', 'sand', 'purpleblue', 'black']
    bounds = chopping.split(',')
    for i, bound in enumerate(bounds):
        if len(bound):
            domain_color = color_list[i % len(color_list)]
            for segment in bound.split('_'):
                start, end = segment.split('-')
                script_text += f"color {domain_color}, structure_id and target and resi {start}-{end}\n"
    script_text += "orient\n"
    script_text += f"png {image_out_path}, ray=1, width=20cm, dpi=300,\n"
    with open(path_to_script, 'w') as filehandle:
        filehandle.write(script_text)
    bash_command = f"{pymol_executable} -c {path_to_script}"
    status = subprocess.run(bash_command.split(), timeout=60)
    os.remove(path_to_script)
    return status

