import torch


def reward_func(queries, prompts):
    # Calculate response lengths by subtracting prompt lengths from query lengths
    response_lengths = [len(query) - len(prompt) for query, prompt in zip(queries, prompts)]
    
    # Convert to tensor and normalize to reasonable reward values
    rewards = torch.tensor(response_lengths, dtype=torch.float32)
    
    return rewards
