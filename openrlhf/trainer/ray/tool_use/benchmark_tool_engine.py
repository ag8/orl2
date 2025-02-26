"""
Benchmark script for the tool-enabled VLLM engine.

This script measures the performance of the tool-enabled VLLM engine
with the best performing configuration from the regular VLLM engine benchmark.
It separates text generation (GPU) from tool execution (CPU) for better performance.
"""

import argparse
import os
import time
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from openrlhf.trainer.ray.tool_use.vllm_tool_engine import create_tool_vllm_engines
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
class ToolExecutorActor:
    """Ray actor for executing Python code in parallel."""
    
    def __init__(self, max_output_length: int = 1000):
        self.max_output_length = max_output_length
    
    def execute_code(self, code: str) -> str:
        """Execute Python code and return the output."""
        import io
        import sys
        import traceback
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Execute the code in the global namespace
                exec(code, globals())
            
            # Get the output
            output = stdout.getvalue()
            
            # If there's no stdout but there is stderr, use stderr
            if not output and stderr.getvalue():
                output = stderr.getvalue()
                
        except Exception as e:
            # Capture the exception
            output = f"Error: {str(e)}\n{traceback.format_exc()}"
        
        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + f"\n... (output truncated, exceeded {self.max_output_length} characters)"
        
        return output
    
    def process_text(self, text: str) -> str:
        """Process text by executing Python code blocks and injecting the output."""
        # Extract code blocks
        pattern = r"<PYTHON>(.*?)</PYTHON>"
        blocks = []
        
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append({
                "start": match.start(),
                "end": match.end(),
                "code": match.group(1).strip()
            })
        
        # If no blocks, return the original text
        if not blocks:
            return text
        
        # Process blocks in reverse order to avoid messing up indices
        blocks.reverse()
        
        # Process each block
        for block in blocks:
            start_idx = block["start"]
            end_idx = block["end"]
            code = block["code"]
            
            # Execute the code
            output = self.execute_code(code)
            
            # Insert the output after the code block
            text = (
                text[:end_idx] + 
                "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>" + 
                text[end_idx:]
            )
        
        return text


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


def extract_python_blocks(text: str) -> List[Dict[str, Any]]:
    """Extract Python code blocks from text."""
    pattern = r"<PYTHON>(.*?)</PYTHON>"
    blocks = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        blocks.append({
            "start": match.start(),
            "end": match.end(),
            "code": match.group(1).strip()
        })
    
    return blocks


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
    num_tool_executors: int = 16,  # Number of parallel tool executors
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
        num_tool_executors: Number of parallel tool executors
        
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
    print(f"  Number of tool executors: {num_tool_executors if tool_use_enabled else 'N/A'}")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create VLLM engines with tool use DISABLED (we'll handle tool execution separately)
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
        tool_use_enabled=False,  # Always disable tool use in the engine
    )
    
    print(f"Created {len(engines)} VLLM engines")
    
    # Create tool executor actors if tool use is enabled
    tool_executors = []
    if tool_use_enabled:
        print(f"Creating {num_tool_executors} tool executor actors")
        for _ in range(num_tool_executors):
            tool_executors.append(ToolExecutorActor.remote(max_output_length=1000))
    
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
    total_generation_time = 0
    total_tool_execution_time = 0
    start_time = time.time()
    
    for batch in range(num_batches):
        print(f"Processing batch {batch+1}/{num_batches}")
        
        # PHASE 1: Text Generation (GPU)
        generation_start_time = time.time()
        
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
        
        generation_time = time.time() - generation_start_time
        total_generation_time += generation_time
        print(f"  Generation completed in {generation_time:.2f} seconds")
        
        # Count tokens from generation
        for response in all_responses:
            for output in response.outputs:
                try:
                    # Handle different types of token_ids
                    if hasattr(output.token_ids, 'shape'):
                        total_tokens += output.token_ids.shape[0]
                    elif isinstance(output.token_ids, (tuple, list)):
                        total_tokens += len(output.token_ids)
                    else:
                        # Try to get length or count in some other way
                        try:
                            total_tokens += len(output.token_ids)
                        except (TypeError, AttributeError):
                            # If all else fails, count tokens in the text
                            total_tokens += len(tokenizer.encode(output.text))
                except Exception as e:
                    logger.error(f"Error counting tokens: {e}")
                    # Make a rough estimate based on words
                    total_tokens += len(output.text.split())
        
        # PHASE 2: Tool Execution (CPU) - only if tool use is enabled
        if tool_use_enabled:
            tool_execution_start_time = time.time()
            
            # Extract all texts that need processing
            texts_to_process = []
            for response in all_responses:
                for output in response.outputs:
                    texts_to_process.append(output.text)
            
            # Process texts in parallel using tool executors
            executor_idx = 0
            processing_refs = []
            
            for text in texts_to_process:
                # Check if the text contains Python code blocks
                if "<PYTHON>" in text:
                    # Assign to a tool executor in a round-robin fashion
                    executor = tool_executors[executor_idx]
                    processing_refs.append(executor.process_text.remote(text))
                    executor_idx = (executor_idx + 1) % len(tool_executors)
                else:
                    # No Python code blocks, no processing needed
                    processing_refs.append(text)
            
            # Wait for all processing to complete
            processed_texts = []
            for ref in processing_refs:
                if isinstance(ref, str):
                    # This was a text without Python code blocks
                    processed_texts.append(ref)
                else:
                    # This was a Ray object reference
                    processed_texts.append(ray.get(ref))
            
            tool_execution_time = time.time() - tool_execution_start_time
            total_tool_execution_time += tool_execution_time
            print(f"  Tool execution completed in {tool_execution_time:.2f} seconds")
    
    total_time = time.time() - start_time
    throughput = total_tokens / total_time
    
    print(f"Benchmark completed in {total_time:.2f} seconds")
    print(f"  Generation time: {total_generation_time:.2f} seconds ({total_generation_time/total_time*100:.1f}%)")
    if tool_use_enabled:
        print(f"  Tool execution time: {total_tool_execution_time:.2f} seconds ({total_tool_execution_time/total_time*100:.1f}%)")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/second")
    
    return total_time, throughput, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark tool-enabled VLLM engine")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to process")
    parser.add_argument("--num-tool-executors", type=int, default=16, help="Number of parallel tool executors")
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Write a Python function to sleep for a random amount of time between 1 and 10 seconds, and then print 'Hello, world!'. Output the code in <PYTHON></PYTHON> tags.",
        "Write a Python function to sleep for a random amount of time between 5 and 15 seconds, and then print the 7th Fibonacci number. Output the code in <PYTHON></PYTHON> tags.",
        "Write a Python function to sleep for a random amount of time between 3 and 8 seconds, and then print the sum of the first 1000 prime numbers. Output the code in <PYTHON></PYTHON> tags.",
        "Write a Python function to sleep for a random amount of time between 2 and 6 seconds, and then print the square root of 42069. Output the code in <PYTHON></PYTHON> tags.",
    ], help="Prompts to use for benchmarking")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Use only the best configuration from the previous benchmark
    # tensor_parallel_size, num_engines, num_actors, batch_size
    tp_size, num_engines, num_actors, batch_size = 1, 2, 2, 16  # Best performing config: 2 engines, each on 1 GPU
    
    # Run benchmark with tool use enabled
    total_time, throughput, total_tokens = run_benchmark(
        model=args.model,
        tensor_parallel_size=tp_size,
        num_engines=num_engines,
        num_actors=num_actors,
        batch_size=batch_size,
        num_batches=args.num_batches,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompts=args.prompts,
        tool_use_enabled=True,
        num_tool_executors=args.num_tool_executors,
    )
    
    # Print summary
    print("\nBenchmark Summary (Tool Use Enabled):")
    print("-" * 80)
    print(f"{'Configuration':<40} {'Time (s)':<10} {'Tokens':<10} {'Throughput':<15}")
    print("-" * 80)
    
    config = f"TP={tp_size}, Engines={num_engines}, Actors={num_actors}, Batch={batch_size}"
    print(f"{config:<40} {total_time:<10.2f} {total_tokens:<10} {throughput:<15.2f}")
    
    # Compare with the previous benchmark result
    print("\nComparison with Regular VLLM Engine:")
    print("-" * 80)
    print(f"{'Configuration':<40} {'Regular':<15} {'With Tools':<15} {'Overhead %':<15}")
    print("-" * 80)
    
    # Previous benchmark result for the best configuration
    regular_throughput = 1715.77  # From the previous benchmark
    
    overhead = (1 - throughput / regular_throughput) * 100
    print(f"{config:<40} {regular_throughput:<15.2f} {throughput:<15.2f} {overhead:<15.2f}")


if __name__ == "__main__":
    main() 