import networkx as nx
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from typing import Optional
import os
import datetime

def extract_solution(solution_str: str) -> Optional[nx.Graph]:
    """Extract graph from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        try:
            graph_str = matches[-1].strip()
            graph_dict = literal_eval(graph_str)
            graph = nx.Graph(graph_dict)
            return graph
        except Exception as e:
            print(f"Error extracting graph: {e}")
            return None
    return None

def reward(G: Optional[nx.Graph]) -> float:
    """
    Calculates the reward for a given graph.
    For Conjecture 2.1: Minimizes lambda_1 + mu.
    
    Parameters:
    G : networkx.Graph or None
        The graph to evaluate
    
    Returns:
    float
        The reward score. Higher is better.
    """
    if not G:
        return -420
    
    # G is assumed to be connected in the conjecture
    if not nx.is_connected(G):
        return -420
    
    # Calculate the eigenvalues of G
    evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
    evalsRealAbs = np.abs(evals)
    lambda1 = np.max(evalsRealAbs)
    
    # Calculate the matching number of G
    maxMatch = nx.max_weight_matching(G)
    mu = len(maxMatch)
    
    # Calculate the reward
    # Since we want to minimize lambda_1 + mu, we return the negative of this
    # We add to this the conjectured best value
    myScore = math.sqrt(len(G.nodes()) - 1) + 1 - lambda1 - mu
    
    # Apply penalty for small graphs
    # Graphs with 19 or more nodes have no penalty
    # Smaller graphs are penalized proportionally to how small they are
    num_nodes = len(G.nodes())
    if num_nodes < 19:
        # Penalty increases as the graph gets smaller
        size_penalty = (19 - num_nodes) / 19
        myScore -= size_penalty * 5  # Scale the penalty appropriately
    
    # Check for counterexample and record best graph
    if myScore > 0:
        print("Potential counterexample found!")
        print(f"Graph: {G.edges()}")
        print(f"lambda1: {lambda1}, mu: {mu}")
        print(f"Score: {myScore}")
        
        # Record the graph to records.txt
        with open("records.txt", "a") as f:
            f.write(f"NEW BEST GRAPH FOUND!\n")
            f.write(f"Graph: {dict(G.edges())}\n")
            f.write(f"Nodes: {num_nodes}, Edges: {len(G.edges())}\n")
            f.write(f"lambda1: {lambda1}, mu: {mu}\n")
            f.write(f"Score: {myScore}\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write("-" * 50 + "\n")
    
    # Keep track of the best graph seen so far
    try:
        with open("records.txt", "r") as f:
            best_score = -float('inf')
            for line in f:
                if line.startswith("Score:"):
                    try:
                        score = float(line.split(":")[1].strip())
                        best_score = max(best_score, score)
                    except:
                        pass
        
        # If this is a new best graph (or first graph), record it
        if myScore > best_score:
            with open("records.txt", "a") as f:
                f.write(f"NEW RECORD GRAPH!\n")
                f.write(f"Graph: {dict(G.edges())}\n")
                f.write(f"Nodes: {num_nodes}, Edges: {len(G.edges())}\n")
                f.write(f"lambda1: {lambda1}, mu: {mu}\n")
                f.write(f"Score: {myScore}\n")
                f.write(f"Date: {datetime.datetime.now()}\n")
                f.write("-" * 50 + "\n")
    except:
        # If the file doesn't exist yet or there's an error reading it
        with open("records.txt", "a") as f:
            f.write(f"FIRST RECORDED GRAPH!\n")
            f.write(f"Graph: {dict(G.edges())}\n")
            f.write(f"Nodes: {num_nodes}, Edges: {len(G.edges())}\n")
            f.write(f"lambda1: {lambda1}, mu: {mu}\n")
            f.write(f"Score: {myScore}\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write("-" * 50 + "\n")
    
    return myScore

def reward_func(queries, prompts):
    rewards = []
    
    print(f"\nCalculating rewards for {len(queries)} samples")
    for query, prompt in zip(queries, prompts):
        response = query[len(prompt):]
        G = extract_solution(response)
        rewards.append(reward(G))
    rewards = torch.tensor(rewards, dtype=torch.float32)
      
    # Show detailed breakdown for first 3 examples
    num_examples = min(3, len(queries))
    for i in range(num_examples):
        response = queries[i][len(prompts[i]):]
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {response}")
        print(f"Reward: {rewards[i]:.3f}")
    
    # Show summary statistics
    print(f"\nReward statistics:")
    print(f"Min reward: {rewards.min():.3f}")
    print(f"Max reward: {rewards.max():.3f}")
    print(f"Mean reward: {rewards.mean():.3f}")
    return rewards
    