"""
Tests for zip_extract.py
"""

from pathlib import Path
import zipfile
from click.testing import CliRunner
from .zip_extract import run


def test_run_no_args():
    runner = CliRunner()
    result = runner.invoke(run, [])
    assert result.exit_code == 2
    assert 'Missing option' in result.output

def test_run_missing_index_file():
    runner = CliRunner()
    result = runner.invoke(run, ['-i', 'no_index_file.txt'])
    assert result.exit_code == 2
    assert 'does not exist' in result.output

def test_run_basic_usage(tmpdir):
    
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmpdir):

        mock_index_fname = 'mock_index.txt'
        mock_pdb_fname = 'mock_afid.pdb'
        mock_zip_fname = 'mock_zip.zip'
        mock_out_dir = 'out'
        mock_pdb_contents = 'MOCK_PDB_CONTENTS\n'

        test_pdb_path = Path(mock_pdb_fname).absolute()
        test_index_path = Path(mock_index_fname).absolute()

        mock_index_cols = [
            mock_pdb_fname.replace('.pdb', ''), 
            'mock_nres',
            'mock_md5', 
            mock_zip_fname.replace('.zip', '')]

        with test_index_path.open('w') as f:
            f.write('\t'.join(mock_index_cols) + '\n')
        with test_pdb_path.open('w') as f:
            f.write(mock_pdb_contents)
        with zipfile.ZipFile(mock_zip_fname, 'w') as tmpzip:
            tmpzip.write(mock_pdb_fname)

        test_pdb_path.unlink()
        assert test_pdb_path.exists() is False

        out_dir = Path.cwd() / mock_out_dir
        out_dir.mkdir()

        cmd_args = ['-i', mock_index_fname, '--zip_dir', '.', '--out_dir', str(out_dir)]
        result = runner.invoke(run, cmd_args)
        print(f"ARGS: {cmd_args}")
        print(f"OUT:  {result.output}")
        print(f"ERR:  {result.exception}")

        assert result.exit_code == 0
        assert f'extracting: {mock_pdb_fname}' in result.output

        out_pdb_path = out_dir / mock_pdb_fname
        assert out_pdb_path.exists() is True
        with out_pdb_path.open('r') as f:
            assert f.read() == mock_pdb_contents