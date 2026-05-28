import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import MACCSkeys

def compute_descriptors(smiles_list, out_array="descriptors.npy", out_names="descriptor_names.npy"):
    names = [d[0] for d in Descriptors.descList]
    calc = MolecularDescriptorCalculator(names)

    X = np.full((len(smiles_list), len(names)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            X[i] = calc.CalcDescriptors(mol)
        except Exception:
            pass  # leave row as NaN

    np.save(out_array, X)
    np.save(out_names, np.array(names))
    return X, names

def smiles_to_maccs(smiles_list, return_invalid=False):
    """Convert SMILES to MACCS key fingerprints (167-bit; index 0 is unused per RDKit convention).

    Invalid SMILES yield all-zero rows. Set return_invalid=True to also get their indices.
    """
    X = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    invalid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(i)
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        X[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord('0')
    return (X, invalid) if return_invalid else X

def generate_fingerprint(smiles_list):
    
    X_desc, names_desc = compute_descriptors(smiles_list)
    X_maccs = smiles_to_maccs(smiles_list)

    out = np.DataFrame(np.hstack([maccs, descriptors]))

    return out
    