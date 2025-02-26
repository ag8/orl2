"""
Benchmark script for VLLM engine configurations in OpenRLHF.

This script measures the performance of different VLLM configurations,
including tensor parallelism and multiple engines.
"""

import argparse
import os
import numpy as np
import ray
import time
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple


@ray.remote
class LLMRayActor:
    """
    A Ray actor that wraps the VLLM engine.
    This is a direct copy of the LLMRayActor from OpenRLHF.
    """

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.2"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.responses = {}

        self.llm = LLM(*args, **kwargs)

    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = prompt_token_ids
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)

            if len(requests) > 0:
                # For now we assume that all requests have the same sampling params
                responses = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=requests)
                
                # Process responses with tool executor if enabled
                if self.tool_use_enabled and self.tool_executor:
                    # Process each response individually
                    for i in range(len(responses)):
                        output_text = responses[i].outputs[0].text
                        processed_text = self.tool_executor.process_text(output_text)
                        # Update the response with the processed text
                        responses[i].outputs[0].text = processed_text
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.responses[actor_rank] = responses[offset : offset + num]
                offset += num

            self.actor_counter = 0
            self.requests = {}

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.responses.pop(actor_rank)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
):
    """
    Create VLLM engines.
    This is a direct copy of create_vllm_engines from OpenRLHF.
    """
    import vllm

    assert vllm.__version__ >= "0.7.0", "OpenRLHF only supports vllm >= 0.7.0"

    vllm_engines = []
    num_gpus = int(tensor_parallel_size == 1)
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    for i in range(num_engines):
        bundle_indices = None
        scheduling_strategy = None

        # Hybrid engine
        if shared_pg is not None:
            assert vllm.__version__ >= "0.7.2", "Only vllm >= 0.7.2 supports hybrid engine"

            if tensor_parallel_size > 1:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i * tensor_parallel_size
                )
                bundle_indices = np.arange(i * tensor_parallel_size, (i + 1) * tensor_parallel_size).tolist()
            else:
                num_gpus = 0.2
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i
                )
        # Distributed RLHF
        elif tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices if shared_pg else None,
            )
        )

    return vllm_engines


def tokenize_prompts(tokenizer, prompts, max_length=4096):
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
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create VLLM engines
    engines = create_vllm_engines(
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        pretrain=model,
        seed=42,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=max_model_len,
        num_total_actors=num_actors,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    print(f"Created {len(engines)} VLLM engines")
    
    # Import the tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,  # Generate shorter responses for benchmarking
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
    for batch_prompts in all_prompts:
        token_ids = tokenize_prompts(tokenizer, batch_prompts, max_model_len)
        all_prompt_token_ids.append(token_ids)
    
    # Run the benchmark
    print(f"Starting benchmark...")
    start_time = time.time()
    total_tokens = 0
    
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx+1}/{num_batches}...")
        
        # Send requests to engines
        refs = []
        for i in range(num_actors):
            engine_idx = i % num_engines
            engine = engines[engine_idx]
            refs.append(engine.add_requests.remote(i, sampling_params=sampling_params, prompt_token_ids=all_prompt_token_ids[i]))
        
        # Wait for all requests to be processed
        ray.get(refs)
        
        # Get responses from engines
        all_responses = []
        for i in range(num_actors):
            engine_idx = i % num_engines
            engine = engines[engine_idx]
            responses = ray.get(engine.get_responses.remote(i))
            all_responses.append(responses)
        
        # Count tokens
        for actor_responses in all_responses:
            for response in actor_responses:
                for output in response.outputs:
                    total_tokens += len(output.token_ids)
    
    end_time = time.time()
    total_time = end_time - start_time
    throughput = total_tokens / total_time
    
    print(f"Benchmark completed:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} tokens/second")
    
    # Shut down Ray
    ray.shutdown()
    
    return total_time, throughput, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VLLM engine configurations")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--prompts", type=str, nargs="+", default=[
        "Write a Python function to calculate the factorial of a number. Output the code in <PYTHON></PYTHON> tags.",
        "Explain the concept of quantum computing to a high school student. If you ever want to write python code, output the code in <PYTHON></PYTHON> tags.",
        "Write a short story about a robot that develops consciousness. If you ever want to use python code, output the code in <PYTHON></PYTHON> tags.",
        "What are the main differences between Python and JavaScript? If you ever want to use python code to demonstrate, output the code in <PYTHON></PYTHON> tags.",
    ], help="Prompts to use for benchmarking")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Define configurations to benchmark
    configs = [
        # tensor_parallel_size, num_engines, num_actors, batch_size
        (1, 2, 2, 16),  # 2 engines, each on 1 GPU
        (2, 1, 2, 16),  # 1 engine with tensor parallelism across 2 GPUs
        (1, 1, 1, 32),  # 1 engine on 1 GPU with larger batch
    ]
    
    results = []
    
    for tp_size, num_engines, num_actors, batch_size in configs:
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
        )
        
        results.append({
            "config": f"TP={tp_size}, Engines={num_engines}, Actors={num_actors}, Batch={batch_size}",
            "total_time": total_time,
            "throughput": throughput,
            "total_tokens": total_tokens,
        })
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"{'Configuration':<40} {'Time (s)':<10} {'Tokens':<10} {'Throughput':<15}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x["throughput"], reverse=True):
        print(f"{result['config']:<40} {result['total_time']:<10.2f} {result['total_tokens']:<10} {result['throughput']:<15.2f}")


if __name__ == "__main__":
    main() 