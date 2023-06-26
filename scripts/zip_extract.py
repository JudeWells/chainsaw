#!/usr/bin/env python3
"""
Extracts PDB files from zip archives for all entries in provided index file
"""

import csv
from pathlib import Path
from zipfile import ZipFile
import click

# for info
CS_ZIP_DIR = '/SAN/bioinf/afdb_domain/zipfiles'

@click.command()
@click.option('--index_file', '-i', 
              required=True,
              type=click.Path(exists=True, file_okay=True),
              help='File containing the index of all entries to extract [model_id, md5sum, zipfile] (tsv)')
@click.option('--out_dir', '-o', 
              type=click.Path(exists=True, dir_okay=True),
              default='.',
              help='Directory into which files will be extracted (default: ".")')
@click.option('--zip_dir', '-z',
              type=click.Path(exists=True, dir_okay=True), 
              default='.',
              help='Directory containing the zip files (default: ".")')
def run(index_file, out_dir, zip_dir):

    index_fieldnames = ['model_id', 'nres', 'md5sum', 'zipfile']

    index_file = Path(index_file).absolute()
    out_dir = Path(out_dir).absolute()
    zip_dir = Path(zip_dir).absolute()

    with index_file.open('rt') as index_fp:

        index_reader = csv.DictReader(index_fp, fieldnames=index_fieldnames, delimiter='\t', strict=True)

        click.echo(f'INDEX:   {index_file}')
        click.echo(f'OUT_DIR: {out_dir}')
        click.echo(f'ZIP_DIR: {zip_dir}')

        for row in index_reader:

            zippath = zip_dir / (row['zipfile'] + '.zip')
            zipfile = ZipFile(zippath)

            member = row['model_id'] + '.pdb'

            click.echo(f' - extracting: {member} (from {zippath.name})')
            zipfile.extract(member, path=out_dir)

if __name__ == "__main__":
    run()
