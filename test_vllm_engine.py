"""
Test script for the VLLM engine in OpenRLHF.

This script creates a direct copy of the VLLM engine as used in OpenRLHF
and tests it in a standalone way that mirrors how it's actually used.
"""

import argparse
import os
import numpy as np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def parse_args():
    parser = argparse.ArgumentParser(description="Test the VLLM engine in OpenRLHF")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--num-engines", type=int, default=1, help="Number of VLLM engines")
    parser.add_argument("--num-actors", type=int, default=1, help="Number of actors")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per actor")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--prompts", type=str, nargs="+", default=["Write a Python function to calculate the factorial of a number."], 
                        help="Prompts to test with")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    print(f"Creating VLLM engines with model: {args.model}")
    print(f"Number of engines: {args.num_engines}")
    print(f"Number of actors: {args.num_actors}")
    print(f"Batch size per actor: {args.batch_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    
    # Create VLLM engines
    engines = create_vllm_engines(
        num_engines=args.num_engines,
        tensor_parallel_size=args.tensor_parallel_size,
        pretrain=args.model,
        seed=42,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        num_total_actors=args.num_actors,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    print(f"Created {len(engines)} VLLM engines")
    
    # Import the tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create batches of prompts
    all_prompts = []
    for i in range(args.num_actors):
        # Repeat prompts to fill the batch
        batch_prompts = args.prompts * (args.batch_size // len(args.prompts) + 1)
        batch_prompts = batch_prompts[:args.batch_size]
        all_prompts.append(batch_prompts)
    
    # Tokenize prompts
    all_prompt_token_ids = []
    for batch_prompts in all_prompts:
        token_ids = tokenize_prompts(tokenizer, batch_prompts, args.max_model_len)
        all_prompt_token_ids.append(token_ids)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )
    
    # Send requests to engines
    print("Sending requests to engines...")
    refs = []
    for i in range(args.num_actors):
        engine_idx = i % args.num_engines
        engine = engines[engine_idx]
        refs.append(engine.add_requests.remote(i, sampling_params=sampling_params, prompt_token_ids=all_prompt_token_ids[i]))
    
    # Wait for all requests to be processed
    ray.get(refs)
    
    # Get responses from engines
    print("Getting responses from engines...")
    all_responses = []
    for i in range(args.num_actors):
        engine_idx = i % args.num_engines
        engine = engines[engine_idx]
        responses = ray.get(engine.get_responses.remote(i))
        all_responses.append(responses)
    
    # Print responses
    print("\nResponses:")
    for actor_idx, responses in enumerate(all_responses):
        print(f"Actor {actor_idx}:")
        for i, response in enumerate(responses):
            prompt = all_prompts[actor_idx][i]
            output = response.outputs[0].text
            
            print(f"  Prompt: {prompt}")
            print(f"  Output: {output}")
            print("  " + "-" * 50)
    
    # Shut down Ray
    ray.shutdown()


if __name__ == "__main__":
    main() 