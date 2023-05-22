import os
import subprocess

import pandas as pd
import glob

from src.utils.common import execute_bash_command


def generate_pymol_image(pdb_path, chain, names, bounds, image_out_path, path_to_script):
    """
    PyMol can only be called via command line, this script generates
    a pymol file which describes how one protein should be visualized
    and saved as a png file.
    :param pdb_path: string which is the path to a .pdb file
    :param chain: target chain
    :param names: names of the domains in format D1|D2|D1
    :param bounds: domain boundaries expressed in residue author numbers in the pdb file 1-10|11-50|51-60
    :param path_to_script: where the pymol bash script is saved
    :return: Bash command that executes the pymol script that the function created
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
    color_list = ['green', 'red', 'cyan', 'magenta', 'blue', 'orange', 'firebrick', 'limon', 'salmon', 'deepteal', 'sand', 'purpleblue', 'black']
    color_dict = {}
    names = names.split('|')
    bounds = bounds.split('|')
    for i, (name, bound) in enumerate(zip(names, bounds)):
        if len(name) > 0:
            start, end = bound.split('-')
            if name in color_dict:
                seg_color = color_dict[name]
            else:
                seg_color = color_list[i % len(color_list)]
                color_dict[name] = seg_color
            script_text += f"color {seg_color}, structure_id and target and resi {start}-{end}\n"
    script_text += "orient\n"
    script_text += f"png {image_out_path}, ray=1, width=20cm, dpi=300,\n"
    with open(path_to_script, 'w') as filehandle:
        filehandle.write(script_text)
    bash_command = f"/Applications/PyMOL.app/Contents/MacOS/PyMOL -c {path_to_script}"
    status = execute_bash_command(bash_command)

# def generate_3d_pymol_image(filepath, domain_boundaries, outdir):
#
#     pass


if __name__=="__main__":
    df = pd.read_csv('../../casp_10_11_12_13_14_w_sifts.csv')
    pdb_dir = '../../features/casp_data/targets'
    path_to_script  = os.path.join(os.getcwd(), 'image_gen.pml')
    for i, row in df.iterrows():
        try:
            pdb_path = glob.glob(f"{pdb_dir}/*/{row.casp_id}.pdb")[0]
        except:
            continue
        if not os.path.exists(pdb_path):
            continue
        path_to_script = os.path.join(os.getcwd(), row.casp_id + '_' + 'image_gen.pml')
        filepath = row.casp_id +'_' + row.dom_bounds_pdb_ix + '.png'
        generate_pymol_image(pdb_path, None, row.dom_names, row.dom_bounds_pdb_ix, filepath, path_to_script)
