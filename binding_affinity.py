#!/usr/bin/env python3

import sys
import os
from rdkit import Chem
from smiles_rewarder import calculate_binding_affinity, SuppressLibraryOutput

def main():
    # Check if a SMILES string was provided
    if len(sys.argv) != 2:
        print("Usage: python binding_affinity.py <SMILES_STRING>")
        print("Example: python binding_affinity.py 'CC(=O)OC1=CC=CC=C1C(=O)O'")
        sys.exit(1)
    
    smiles = sys.argv[1]
    
    # Validate SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: '{smiles}' is not a valid SMILES string.")
        sys.exit(1)
    
    print(f"Calculating binding affinity for: {smiles}")
    
    # Calculate binding affinity
    with SuppressLibraryOutput():
        affinity = calculate_binding_affinity(smiles)
    
    if affinity is None:
        print("Failed to calculate binding affinity.")
        sys.exit(1)
    
    # Print results
    print(f"\nResults for {smiles}:")
    print(f"Binding Affinity: {affinity:.2f} kcal/mol")
    print(f"(Lower values indicate stronger binding)")
    
    # Also calculate the positive score used in the reward function
    positive_score = 0 - affinity
    print(f"Positive Score: {positive_score:.2f}")

if __name__ == "__main__":
    main() 