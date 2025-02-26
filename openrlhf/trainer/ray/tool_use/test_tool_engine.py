"""
Test script for the tool-enabled VLLM engine.

This script initializes a tool-enabled VLLM engine and tests basic generation
functionality to ensure it works correctly before adding tool use features.
"""

import argparse
import os
import ray
import torch
from typing import List
from vllm import SamplingParams

from openrlhf.trainer.ray.tool_use.vllm_tool_engine import create_tool_vllm_engines
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Test the tool-enabled VLLM engine")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--tool-use-enabled", action="store_true", help="Enable tool use")
    parser.add_argument("--num-tool-executors", type=int, default=32, help="Number of parallel tool executors")
    parser.add_argument("--prompts", type=str, nargs="+", default=["Write a Python function to calculate the factorial of a number. Make sure to output the code in <PYTHON></PYTHON> tags.", "Write a Python function to fuckin destroy the primen umber conjecture LOL!!!!!. Make sure to output the code in <PYTHON></PYTHON> tags."], 
                        help="Prompts to test with")
    return parser.parse_args()


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


def main():
    args = parse_args()
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    logger.info(f"Creating tool-enabled VLLM engine with model: {args.model}")
    logger.info(f"Tool use enabled: {args.tool_use_enabled}")
    logger.info(f"Number of tool executors: {args.num_tool_executors if args.tool_use_enabled else 'N/A'}")
    
    # Create a tool-enabled VLLM engine
    engines = create_tool_vllm_engines(
        num_engines=1,
        tensor_parallel_size=args.tensor_parallel_size,
        pretrain=args.model,
        seed=42069,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        num_total_actors=1,
        tool_use_enabled=args.tool_use_enabled,
        num_tool_executors=args.num_tool_executors,
    )
    
    engine = engines[0]
    
    # Import the tokenizer from the model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Tokenize the prompts
    prompt_token_ids = tokenize_prompts(tokenizer, args.prompts, args.max_model_len)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )
    
    # Send the request to the engine
    logger.info("Sending request to the engine")
    ray.get(engine.add_requests.remote(0, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
    
    # Get the response
    logger.info("Getting response from the engine")
    responses = ray.get(engine.get_responses.remote(0))
    
    # Print the responses
    logger.info("Responses:")
    for i, response in enumerate(responses):
        prompt = args.prompts[i]
        output = response.outputs[0].text
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output}")
        logger.info("-" * 50)
    
    # Shut down Ray
    ray.shutdown()


if __name__ == "__main__":
    main() 