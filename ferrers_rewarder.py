import networkx as nx
import re
import math
import torch
from ast import literal_eval
from typing import Optional

def extract_solution(solution_str: str) -> Optional[nx.Graph]:
    """Extract graph from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    answer_pattern = r'<answer>.*?</answer>'
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


def T(G: nx.Graph):
    return nx.algorithms.tree.mst.number_of_spanning_trees(G)

def F(G: nx.Graph) -> float:
    return ferrers_invariant(G)

def ferrers_invariant(G: nx.Graph) -> float:
    """
    Compute the Ferrers invariant of a bipartite graph.
    
    Parameters:
    G : networkx.Graph
        A bipartite graph
    
    Returns:
    float
        The Ferrers invariant of the graph
    """
    if not nx.is_connected(G):
        return 0.0
    if not nx.is_bipartite(G):
        raise ValueError("The graph is not bipartite.")
    
    X, Y = nx.bipartite.sets(G)
    product_degrees = 1
    for v in G.nodes():
        product_degrees *= G.degree(v)
    
    return product_degrees / (len(X) * len(Y))


def reward(G: Optional[nx.Graph]) -> float:
    if not G:
        return -3.0
    if not nx.is_connected(G):
        return -2.0
    if not nx.is_bipartite(G):
        return -1.0
    return math.log(T(G)) - math.log(F(G)) - 0.9/T(G)


def reward_func(queries, prompts):
    rewards = []
    
    print(f"\nCalculating rewards for {len(queries)} samples")
    for query, prompt in zip(queries, prompts):
        response = query[len(prompt):]
        G = extract_solution(response)
        rewards.append(reward(G))
    return torch.tensor(rewards, dtype=torch.float32)
    