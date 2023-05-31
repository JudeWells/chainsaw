from pathlib import Path
from typing import Optional
from pydantic import BaseModel, validator, Field

def chain_id_from_pdb_path(pdb_path: Path):
    return pdb_path.stem

class PredictionResult(BaseModel):

    pdb_path: Path
    uncertainty: float
    chain_id: Optional[str]
    chopping: str = Field(..., min_length=0)
    domain_id: str = Field(..., min_length=0)
    
    @validator('chain_id', always=True, pre=True, allow_reuse=True)
    def set_chain_id(cls, v, values):
        if v is None:
            return chain_id_from_pdb_path(values['pdb_path'])
        return v
