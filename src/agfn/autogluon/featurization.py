"""Shared molecular featurization for AutoGluon tabular models.

Every AutoGluon model in this package consumes the same feature matrix: MACCS keys
concatenated with the full RDKit descriptor set. Keep model-specific logic out of here.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def compute_rdkit_descriptors(smiles_list, return_success=False):
    names = [d[0] for d in Descriptors.descList]
    calc = MolecularDescriptorCalculator(names)
    values = np.full((len(smiles_list), len(names)), np.nan, dtype=np.float32)
    success = np.zeros(len(smiles_list), dtype=bool)
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            values[i] = calc.CalcDescriptors(mol)
            success[i] = True
        except Exception:
            pass
    if return_success:
        return values, names, success
    return values, names


def smiles_to_maccs(smiles_list, return_invalid=False, return_success=False):
    values = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    invalid = []
    success = np.zeros(len(smiles_list), dtype=bool)
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid.append(i)
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            values[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
            success[i] = True
        except Exception:
            invalid.append(i)
            continue
    if return_success:
        return values, success
    return (values, invalid) if return_invalid else values


def generate_features(smiles_list, return_success=False):
    if return_success:
        descriptors, _, descriptor_success = compute_rdkit_descriptors(
            smiles_list, return_success=True)
        maccs, maccs_success = smiles_to_maccs(smiles_list, return_success=True)
        return pd.DataFrame(np.hstack([maccs, descriptors])), descriptor_success & maccs_success
    descriptors, _ = compute_rdkit_descriptors(smiles_list)
    maccs = smiles_to_maccs(smiles_list)
    return pd.DataFrame(np.hstack([maccs, descriptors]))
