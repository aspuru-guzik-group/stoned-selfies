from typing import Union, List, Tuple

import numpy as np 

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from selfies import encoder, decoder, get_semantic_robust_alphabet


class StonedChemicalSubspace:
    """StonedChemicalSubspace."""

    def __init__(
            self, 
            seed_smiles:str,
            num_samples:int,
            num_mutations:Union[int, List[int]], 
            fingerprint_type:str='ECFP4', 
        ) -> None:
        """Implementation of the STONED algorithm to generate a local chemical subspace.

        Parameters
        ----------
        seed_smiles : str
            Generate new molecules from this seed SMILES string. 
        num_samples : int
            The number of generated molecules
        num_mutations : Union[int, List[int]]
            Number of mutations performed on a seed molecule. Can be an `int` or a list 
            of integers. For `int` = n, the seed molecule is mutated n times. 
            If a list of `int` are passed, the seed molecule will perform successive 
            mutations according the the values in the list.  
        fingerprint_type : str, optional
            The fingerprint used for the comparison metric, by default 'ECFP4'

        Notes
        -----
        Mutations are 'point' mutations - meaning the SELFIE character is mutated, or a
        SELFIE character is added or removed. See the `_mutate_selfies` method for more
        details. 
        """
        self.seed_smiles = seed_smiles
        self.num_samples = num_samples
        self.num_mutations = num_mutations
        self.fingerprint_type = fingerprint_type

    def generate(self) -> List[str]:
        """Generate the chemical subspace."""

        # generate randomized seed molecules
        rand_seed_smiles = self._get_randomize_smiles() # get seed smiles 
        rand_seed_selfies = [encoder(smi) for smi in rand_seed_smiles]

        # mutate SELFIES
        if isinstance(self.num_mutations, int): 
            mutated_selfies = self.get_mutated_SELFIES(rand_seed_selfies, self.num_mutations)
        elif isinstance(self.num_mutations, list): 
            for n in self.num_mutations: 
                mutated_selfies = self._get_mutated_SELFIES(rand_seed_selfies, num_mutations=n)
        else: 
            msg = "`num_mutations` should be an integer or a list of integers"
            raise ValueError(msg)
        
        # convert back to SMILES and return
        mutated_smiles = [decoder(_selfie) for _selfie in mutated_selfies]
        return mutated_smiles

    def _get_mutated_SELFIES(self, selfies_ls:List[str], num_mutations:int) -> List[str]:
        """Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times. 

        Parameters
        ----------
        selfies_ls : List[str]
            A list of SELFIES 
        num_mutations : int
            The number of mutations to perform on each SELFIES within 'selfies_ls'

        Returns
        -------
        List[str]
            A list of mutated SELFIES
        """
        for _ in range(num_mutations): 
            selfie_ls_mut_ls = []
            for str_ in selfies_ls: 
                str_chars = self._get_selfie_chars(str_)
                max_molecules_len = len(str_chars) + num_mutations
                
                selfie_mutated, _ = self._mutate_selfie(str_, max_molecules_len)
                selfie_ls_mut_ls.append(selfie_mutated)
            
            selfies_ls = selfie_ls_mut_ls.copy()
        return selfies_ls

    def _get_randomize_smiles(self) -> List[str]:
        """Returns a random (de-aromatized) SMILES given an rdkit mol object of a 
        molecule.

        Returns
        -------
        List[str]
            A list of randomized SMILES strings
        """
        seed_mol = Chem.MolFromSmiles(self.seed_smiles)
        Chem.Kekulize(seed_mol)
        
        # generate list of random smiles
        out = [] 
        for _ in range(self.num_samples): 
            _random_smiles = Chem.MolToSmiles(
                seed_mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True
            )
            out.append(_random_smiles)
        return out

    def _sanitize_smiles(self, smi:str) -> Tuple[Mol, str, bool]:
        """Return a canonical smile representation of a SMILES string.

        Parameters
        ----------
        smi : str
            A SMILES string

        Returns
        -------
        Tuple[Mol, str, bool]
            Rdkit Mol object,
            Canonicalized SMILES string
            True/False indicating successful sanitation.
        """
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return (mol, smi_canon, True)
        except:
            return (None, None, False)

    def _get_selfie_chars(self, selfie:str) -> str:
        """Obtain a list of all selfie characters in string SELFIE. 

        Parameters
        ----------
        selfie : str
            A SELFIE string - representing a molecule. 
        
        
        Returns
        -------
        str
            list of selfie characters present in molecule SELFIE. 
        
        Example
        -------
        >>> _get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
        ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']   
        """
        chars_selfie = [] # A list of all SELFIE sybols from string selfie
        while selfie != '':
            chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
            selfie = selfie[selfie.find(']')+1:]
        return chars_selfie

    def _mutate_selfie(
            self, selfie:str, max_molecules_len:int, write_fail_cases:bool=False
            ) -> Tuple[str, str]:
        """Return a mutated selfie string (only one mutation on SELFIE is performed)
        
        Mutations are done until a valid molecule is obtained 
        Rules of mutation: With a 33.3% probability, either: 
            1. Add a random SELFIE character in the string
            2. Replace a random SELFIE character with another
            3. Delete a random character

        Parameters
        ----------
        selfie : str
            SELFIE string to be mutated 
        max_molecules_len : int
            Mutations of SELFIE string are allowed up to this length
        write_fail_cases : bool, optional
            If true, failed mutations are recorded in "selfie_failure_cases.txt", by default False

        Returns
        -------
        Tuple[str, str]
            selfie_mutated: Mutated SELFIE string
            smiles_canon: Canonical smile of mutated SELFIE string
        """
        valid=False
        fail_counter = 0
        chars_selfie = self._get_selfie_chars(selfie)
        
        while not valid:
            fail_counter += 1
                    
            alphabet = list(get_semantic_robust_alphabet()) # 34 SELFIE characters 

            choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
            random_choice = np.random.choice(choice_ls, 1)[0]
            
            # Insert a character in a Random Location
            if random_choice == 1: 
                random_index = np.random.randint(len(chars_selfie)+1)
                random_character = np.random.choice(alphabet, size=1)[0]
                
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

            # Replace a random character 
            elif random_choice == 2:                         
                random_index = np.random.randint(len(chars_selfie))
                random_character = np.random.choice(alphabet, size=1)[0]
                if random_index == 0:
                    selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
                    
            # Delete a random character
            elif random_choice == 3: 
                random_index = np.random.randint(len(chars_selfie))
                if random_index == 0:
                    selfie_mutated_chars = chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
                    
            else: 
                raise Exception('Invalid Operation trying to be performed')

            selfie_mutated = "".join(x for x in selfie_mutated_chars)
            sf = "".join(x for x in chars_selfie)
            
            try:
                smiles = decoder(selfie_mutated)
                mol, smiles_canon, done = self._sanitize_smiles(smiles)
                if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                    done = False
                if done:
                    valid = True
                else:
                    valid = False
            except:
                valid=False
                if fail_counter > 1 and write_fail_cases == True:
                    f = open("selfie_failure_cases.txt", "a+")
                    f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                    f.close()
        
        return (selfie_mutated, smiles_canon)


if __name__ == '__main__':
    smi = "ON=C1C=COC(O)=C1"
    smi = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'   # Celecoxib

    mol_target = Chem.MolFromSmiles(smi)


    generator = StonedChemicalSubspace(smi, num_samples=10, num_mutations=1)
    out = generator.generate() 

    print(len(out), out)
