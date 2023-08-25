from typing import Any, List

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


def get_fingerprint(mol: Mol, fp_type: str):
    """Fingerprint helper method. Fingerprint is returned after using object of class 
    `FingerprintCalculator`. 

    Parameters
    ----------
    mol : Mol
        RdKit mol object (None if invalid smile string smi)
    fp_type : str
        Fingerprint type (choices: AP, PHCO, BPF, BTF, PAT, ECFP4, ECFP6, FCFP4, FCFP6)
    """
    return FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)

def get_fingerprint_scores(
        test_smiles:List[str], ref_smiles:str, fp_type:str
        ) -> List[float]: 
    """Calculate the Tanimoto fingerprint (using fp_type fingerint) similarity between a list 
       of SMILES and a known target structure (ref_smiles). 

    Parameters
    ----------
    test_smiles : List[str]
        A list of valid (generated) SMILES strings 
    ref_smiles : str
        A valid SMILES string. Each smile in 'test_smiles' will be compared to this stucture
    fp_type : str
        Fingerprint type (choices: AP, PHCO, BPF, BTF, PAT, ECFP4, ECFP6, FCFP4, FCFP6)

    Returns
    -------
    List[float]
        List of fingerprint similarities
    """
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(ref_smiles)

    fp_target = get_fingerprint(target, fp_type)

    for item in test_smiles: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

class FingerprintCalculator:
    """Chemical Fingerprint Calculator."""

    def get_fingerprint(self, mol: Mol, fp_type: str) -> float:
        """Calculate the fingerprint for a molecule, given the fingerprint type. 

        Parameters
        ----------
        mol : Mol
            RdKit mol object (None if invalid smile string smi)
        fp_type : str
            Fingerprint type 
            (choices: AP, PHCO, BPF, BTF, PAT, ECFP4, ECFP6, FCFP4, FCFP6)  
        """
        method_name = '_get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def _get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def _get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def _get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def _get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def _get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def _get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def _get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def _get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def _get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)
