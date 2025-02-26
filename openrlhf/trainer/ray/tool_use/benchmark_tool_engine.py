"""
Benchmark script for the tool-enabled VLLM engine.

This script measures the performance of the tool-enabled VLLM engine
with and without tool use enabled.
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from openrlhf.trainer.ray.tool_use.vllm_tool_engine import create_tool_vllm_engines
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def tokenize_prompts(tokenizer, prompts: List[str], max_length: int = 4096):
    """Tokenize prompts using the provided tokenizer."""
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return inputs["input_ids"].tolist()


def run_benchmark(
    model: str,
    tensor_parallel_size: int,
    num_engines: int,
    num_actors: int,
    batch_size: int,
    num_batches: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    prompts: List[str],
    tool_use_enabled: bool,
) -> Tuple[float, float, int]:
    """
    Run a benchmark with the given configuration.
    
    Args:
        model: Path to the model to use
        tensor_parallel_size: Tensor parallel size
        num_engines: Number of VLLM engines
        num_actors: Number of actors
        batch_size: Batch size per actor
        num_batches: Number of batches to process
        max_model_len: Maximum model length
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        prompts: List of prompts to use
        tool_use_enabled: Whether to enable tool use
        
    Returns:
        Tuple of (total_time, throughput, total_tokens)
    """
    print(f"\nRunning benchmark with configuration:")
    print(f"  Model: {model}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Number of engines: {num_engines}")
    print(f"  Number of actors: {num_actors}")
    print(f"  Batch size per actor: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Tool use enabled: {tool_use_enabled}")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create VLLM engines
    engines = create_tool_vllm_engines(
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        pretrain=model,
        seed=42,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=max_model_len,
        num_total_actors=num_actors,
        gpu_memory_utilization=gpu_memory_utilization,
        tool_use_enabled=tool_use_enabled,
    )
    
    print(f"Created {len(engines)} VLLM engines")
    
    # Import the tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,  # Generate shorter responses for benchmarking
    )
    
    # Prepare prompts
    all_prompts = []
    for i in range(num_actors):
        # Repeat prompts to fill the batch
        batch_prompts = prompts * (batch_size // len(prompts) + 1)
        batch_prompts = batch_prompts[:batch_size]
        all_prompts.append(batch_prompts)
    
    # Tokenize prompts
    all_prompt_token_ids = []
    for actor_prompts in all_prompts:
        prompt_token_ids = tokenize_prompts(tokenizer, actor_prompts, max_model_len)
        all_prompt_token_ids.append(prompt_token_ids)
    
    # Assign actors to engines
    actor_to_engine = {}
    for i in range(num_actors):
        engine_idx = i % num_engines
        actor_to_engine[i] = engines[engine_idx]
    
    # Run the benchmark
    total_tokens = 0
    start_time = time.time()
    
    for batch in range(num_batches):
        print(f"Processing batch {batch+1}/{num_batches}")
        
        # Send requests to engines
        for actor_rank in range(num_actors):
            engine = actor_to_engine[actor_rank]
            prompt_token_ids = all_prompt_token_ids[actor_rank]
            ray.get(engine.add_requests.remote(actor_rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        
        # Get responses from engines
        all_responses = []
        for actor_rank in range(num_actors):
            engine = actor_to_engine[actor_rank]
            responses = ray.get(engine.get_responses.remote(actor_rank))
            all_responses.extend(responses)
        
        # Count tokens
        for response in all_responses:
            for output in response.outputs:
                total_tokens += output.token_ids.shape[0]
    
    total_time = time.time() - start_time
    throughput = total_tokens / total_time
    
    print(f"Benchmark completed in {total_time:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/second")
    
    return total_time, throughput, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark tool-enabled VLLM engine")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--num-engines", type=int, default=1, help="Number of VLLM engines")
    parser.add_argument("--num-actors", type=int, default=1, help="Number of actors")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per actor")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to process")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Write a Python function to calculate the factorial of a number. Output the code in <PYTHON></PYTHON> tags.",
        "Explain the concept of quantum computing to a high school student. If you ever want to write python code, output the code in <PYTHON></PYTHON> tags.",
        "Write a short story about a robot that develops consciousness. If you ever want to use python code, output the code in <PYTHON></PYTHON> tags.",
        "What are the main differences between Python and JavaScript? If you ever want to use python code to demonstrate, output the code in <PYTHON></PYTHON> tags.",
    ], help="Prompts to use for benchmarking")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Run benchmark with tool use disabled
    total_time_no_tools, throughput_no_tools, total_tokens_no_tools = run_benchmark(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        num_engines=args.num_engines,
        num_actors=args.num_actors,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompts=args.prompts,
        tool_use_enabled=False,
    )
    
    # Run benchmark with tool use enabled
    total_time_tools, throughput_tools, total_tokens_tools = run_benchmark(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        num_engines=args.num_engines,
        num_actors=args.num_actors,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompts=args.prompts,
        tool_use_enabled=True,
    )
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Tokens':<10} {'Throughput':<15}")
    print("-" * 80)
    print(f"{'Without Tool Use':<20} {total_time_no_tools:<10.2f} {total_tokens_no_tools:<10} {throughput_no_tools:<15.2f}")
    print(f"{'With Tool Use':<20} {total_time_tools:<10.2f} {total_tokens_tools:<10} {throughput_tools:<15.2f}")
    print("-" * 80)
    
    # Calculate overhead
    overhead = (total_time_tools - total_time_no_tools) / total_time_no_tools * 100
    print(f"Tool use overhead: {overhead:.2f}%")


if __name__ == "__main__":
    main() 