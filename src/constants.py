from pathlib import Path
import os


REPO_ROOT = Path(__file__).parent.parent.resolve()
STRIDE_EXE = os.environ.get('STRIDE_EXE', str(REPO_ROOT / "stride" / "stride"))