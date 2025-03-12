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
import time
import json
RDLogger.DisableLog('rdApp.*')

# Simple file-based prompt logging that's compatible with Ray serialization
_PROMPT_LOG_FILE = "prompt_history.log"
_PROMPT_BUFFER_SIZE = 100  # Number of prompts to buffer before writing

import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="sk-ant-api03-Fl3jVaFGa1KNeps5DgoTkBYBy9Z1tVN65NUksAQHYI5IRFkPv2mmEVfwjvVeFcv7JeA8zaMpFoMsFgpbNJSrWpcU6A-19hrLgAA",
)

class PromptLogger:
    """A serializable prompt logger that buffers prompts and periodically writes them to disk."""
    
    def __init__(self):
        self.buffer = []
        self.buffer_size = _PROMPT_BUFFER_SIZE
    
    def add_prompt(self, prompt_id, prompt, response, reward=None, best_smiles=None, molecule_score=None):
        """Add a prompt to the buffer and flush if needed."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.buffer.append({
            "timestamp": timestamp,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "best_smiles": best_smiles,
            "molecule_score": molecule_score
        })
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered prompts to the log file."""
        if not self.buffer:
            return
            
        try:
            # Write each prompt as a separate JSON line for easier processing
            with open(_PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
                for entry in self.buffer:
                    # Write as JSON line with delimiter for readability
                    f.write("\n" + "="*80 + "\n")
                    
                    # Add reward information prominently at the top
                    reward_info = ""
                    if entry['reward'] is not None:
                        reward_info = f" | REWARD: {entry['reward']:.4f}"
                        if entry['best_smiles'] is not None:
                            reward_info += f" | BEST SMILES: {entry['best_smiles']}"
                        if entry['molecule_score'] is not None:
                            reward_info += f" | MOLECULE SCORE: {entry['molecule_score']:.4f}"
                    
                    f.write(f"PROMPT ID: {entry['prompt_id']} | TIMESTAMP: {entry['timestamp']}{reward_info}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"PROMPT:\n{entry['prompt']}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"RESPONSE:\n{entry['response']}\n")
                    f.write("="*80 + "\n\n")
            
            # Clear the buffer after successful write
            self.buffer = []
        except Exception as e:
            print(f"Error writing prompts to log file: {str(e)}")

# Create a new logger for each reward_func call to avoid serialization issues
def get_prompt_logger():
    return PromptLogger()

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
    
    # Use re.DOTALL to match across newlines
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        final_answer = matches[-1].group(1).strip()
        # Split by commas and strip each SMILES string
        return [s.strip() for s in final_answer.split(',') if s.strip()]
    return []

def chemistry_penalty(response):
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8192,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"On a scale from 0 (not chemistry related) to 100 (PhD in chemistry talking about their dissertation), how chemistry-related is the following text:\n\n```\n{response}\n```\n\nReturn your number in brackets, e.g. [32] or [57] etc."
                    }
                ]
            }
        ]
    )
    print(message.content)

    return float(message.content[0].text.split('[')[1].split(']')[0])


def reward_func(queries, prompts, labels=None):
    rewards = []
    
    print(f"\nCalculating rewards for {len(queries)} samples")
    
    # Create a new logger instance for this call (avoids serialization issues)
    prompt_logger = get_prompt_logger()
    
    # Load current record from file
    record_file = "records.txt"
    try:
        with open(record_file, 'r') as f:
            lines = f.readlines()
            current_record = float(lines[-1].split(',')[1]) if lines else float('-inf')
    except (FileNotFoundError, IndexError, ValueError):
        current_record = float('-inf')
    
    for i, (query, prompt) in enumerate(zip(queries, prompts)):
        response = query[len(prompt):]
        
        # Log the prompt and response to our buffer - we'll update with reward later
        prompt_id = f"{int(time.time())}_{i}"
        
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

                            # If the number of words is greater than 300, penalize for chemistry
                            if len(words) > 300:
                                penalty = chemistry_penalty(response[len(prompt):])
                                total_score -= penalty * 0.3
                            
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
        reward_value = max(max_score, 0)
        rewards.append(reward_value)
        
        # Now log the prompt with the calculated reward
        prompt_logger.add_prompt(
            prompt_id=prompt_id, 
            prompt=prompt, 
            response=response,
            reward=reward_value,
            best_smiles=best_smiles,
            molecule_score=best_molecule_score if best_molecule_score > 0 else None
        )
    
    # Make sure to flush any remaining prompts before returning
    prompt_logger.flush()
    
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
