import torch
from rdkit import Chem
from rdkit.Chem import QED, AllChem
import re
import os
import tempfile
from contextlib import contextmanager
from vina import Vina
from rdkit import RDLogger
from meeko import MoleculePreparation
RDLogger.DisableLog('rdApp.*')


class SuppressLibraryOutput:
    def __enter__(self):
        self.original_stdout_fd = os.dup(1)
        self.original_stderr_fd = os.dup(2)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), 1)
        os.dup2(self.devnull.fileno(), 2)
        self.original_env = {}
        for var in ['VINA_SILENT', 'MEEKO_SILENT', 'RDKIT_SILENT']:
            self.original_env[var] = os.environ.get(var)
            os.environ[var] = '1'
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.original_stdout_fd, 1)
        os.dup2(self.original_stderr_fd, 2)
        os.close(self.original_stdout_fd)
        os.close(self.original_stderr_fd)
        self.devnull.close()
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


class VinaCache:
    _instance = None
    
    def __init__(self):
        if VinaCache._instance is not None:
            raise Exception("This class is a singleton!")
        
        with SuppressLibraryOutput():
            self.vina = Vina(sf_name='vina')
            self.vina.set_receptor("/root/orl2/7L11.pdbqt")
            self.vina.compute_vina_maps(center=(-21, -3, -29), box_size=(24, 24, 24))
        self.affinity_cache = {}
        VinaCache._instance = self
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VinaCache()
        return cls._instance


def calculate_binding_affinity(smiles_string):
    """Calculate binding affinity between a ligand (SMILES) and protein using AutoDock Vina."""
    try:
        vina_cache = VinaCache.get_instance()
        if smiles_string in vina_cache.affinity_cache:
            return vina_cache.affinity_cache[smiles_string]

        with SuppressLibraryOutput():
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            center_coords = (-21, -3, -29)
            mol_center = conf.GetPositions().mean(axis=0)
            translation = [c - m for c, m in zip(center_coords, mol_center)]
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (pos.x + translation[0], 
                                       pos.y + translation[1],
                                       pos.z + translation[2]))
            
            ligand_pdb = None
            ligand_pdbqt = None
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_ligand:
                    ligand_pdb = temp_ligand.name
                    ligand_pdbqt = ligand_pdb.replace('.pdb', '.pdbqt')
                    
                    Chem.MolToPDBFile(mol, ligand_pdb)
                    preparator = MoleculePreparation()
                    preparator.prepare(mol)
                    preparator.write_pdbqt_file(ligand_pdbqt)
                
                    v = VinaCache.get_instance().vina
                    v.set_ligand_from_file(ligand_pdbqt)
                    v.dock(exhaustiveness=8, n_poses=20)
                    
                    best_affinity = v.energies()[0][0]
                    vina_cache.affinity_cache[smiles_string] = best_affinity
                    
                    return best_affinity
            finally:
                if ligand_pdb and os.path.exists(ligand_pdb):
                    os.unlink(ligand_pdb)
                if ligand_pdbqt and os.path.exists(ligand_pdbqt):
                    os.unlink(ligand_pdbqt)
                    
    except Exception as e:
        print(f"Docking error for {smiles_string}: {str(e)}")
        return None


def extract_solution(solution_str):
    """Extract SMILES strings from the solution."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        return [s.strip() for s in final_answer.split(',') if s.strip()]
    return []


def reward_func(queries, prompts, labels=None):
    rewards = []
    
    print(f"\nCalculating rewards for {len(queries)} samples")
    
    # Load current record from file
    record_file = "records.txt"
    try:
        with open(record_file, 'r') as f:
            lines = f.readlines()
            current_record = float(lines[-1].split(',')[1]) if lines else float('-inf')
    except (FileNotFoundError, IndexError, ValueError):
        current_record = float('-inf')
    
    for query, prompt in zip(queries, prompts):
        response = query[len(prompt):]
        smiles_strings = extract_solution(response)
        
        max_score = 0.0
        best_molecule_score = 0.0  # Track molecule score separately
        best_smiles = None
        
        word_count_reward = 0.0
        
        if smiles_strings:
            for smiles in smiles_strings:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        qed_score = QED.default(mol)
                        docking_score = calculate_binding_affinity(smiles)
                        
                        if docking_score is not None:
                            positive_docking_score = 0 - docking_score
                            molecule_score = 5 * qed_score + 3 * positive_docking_score

                            # Calculate word count reward separately
                            words = response.split()
                            word_count_reward = len(words) * 0.05
                            
                            # Total score includes word count reward for RL training
                            total_score = molecule_score + word_count_reward
                            
                            if total_score > max_score:
                                max_score = total_score
                                best_molecule_score = molecule_score  # Store molecule score without word count
                                best_smiles = smiles
                
                except Exception as e:
                    print(f"Error processing molecule {smiles}: {str(e)}")
                    continue
        
        # Record if we have a new best molecule score (without word count reward)
        if best_molecule_score > current_record and best_smiles is not None:
            try:
                with open(record_file, 'a') as f:
                    f.write(f"{best_smiles},{best_molecule_score}\n")
                current_record = best_molecule_score
            except Exception as e:
                print(f"Failed to write to records file: {str(e)}")
        
        # For RL training reward, we still use the total score including word count
        rewards.append(max(max_score, 0))
    
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # Show detailed breakdown for first 3 examples
    num_examples = min(3, len(queries))
    for i in range(num_examples):
        response = queries[i][len(prompts[i]):]
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {response}")
        print(f"Extracted SMILES: {extract_solution(response)}")
        print(f"Reward: {rewards[i]:.2f}")
    
    # Show summary statistics
    print(f"\nReward statistics:")
    print(f"Min reward: {rewards.min():.2f}")
    print(f"Max reward: {rewards.max():.2f}")
    print(f"Mean reward: {rewards.mean():.2f}")
    
    return rewards
