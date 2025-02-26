"""
Tool-enabled version of the VLLM engine.

This is initially a copy of the original VLLM engine, which we'll modify
to support executing Python code during generation.
"""

import os
import time
import re
from typing import List, Dict, Any

import numpy as np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.utils.logging_utils import init_logger
from openrlhf.trainer.ray.tool_use.tool_executor import ToolExecutor

logger = init_logger(__name__)


@ray.remote
class ToolExecutorActor:
    """Ray actor for executing Python code in parallel."""
    
    def __init__(self, max_output_length: int = 1000):
        self.max_output_length = max_output_length
        self.executor = ToolExecutor(max_output_length=max_output_length)
    
    def execute_code(self, code: str) -> str:
        """Execute Python code and return the output."""
        return self.executor.execute_code(code)
    
    def process_text(self, text: str) -> str:
        """Process text by executing Python code blocks and injecting the output."""
        return self.executor.process_text(text)
    
    def process_block(self, text: str, start_idx: int, end_idx: int, code: str) -> str:
        """Process a single code block and return the modified text."""
        # Execute the code
        output = self.execute_code(code)
        
        # Insert the output after the code block
        return (
            text[:end_idx] + 
            "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>" + 
            text[end_idx:]
        )


@ray.remote
class ToolLLMRayActor:
    """
    A Ray actor that wraps the VLLM engine with tool use capabilities.
    Initially, this is a copy of LLMRayActor with minimal changes.
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

        # Initialize tool use capabilities
        self.tool_use_enabled = kwargs.pop("tool_use_enabled", False)
        self.num_tool_executors = kwargs.pop("num_tool_executors", 32)  # Default to 32 parallel executors
        
        # Create tool executor actors if tool use is enabled
        self.tool_executors = []
        if self.tool_use_enabled:
            logger.info(f"Creating {self.num_tool_executors} tool executor actors")
            for _ in range(self.num_tool_executors):
                self.tool_executors.append(ToolExecutorActor.remote(max_output_length=1000))

        self.llm = LLM(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()
        
    def extract_python_blocks(self, text: str) -> List[Dict[str, Any]]:
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
                
                # Process responses with tool executors if enabled
                if self.tool_use_enabled and self.tool_executors:
                    generation_time = time.time()
                    logger.info(f"Processing {len(responses)} responses with tool executors")
                    
                    # Process each response in parallel
                    processing_tasks = []
                    response_indices = []
                    
                    for i, response in enumerate(responses):
                        output_text = response.outputs[0].text
                        
                        # Check if the response contains Python code blocks
                        if "<PYTHON>" in output_text and "</PYTHON>" in output_text:
                            # Assign to a tool executor in a round-robin fashion
                            executor_idx = i % len(self.tool_executors)
                            executor = self.tool_executors[executor_idx]
                            
                            # Process the entire text with all code blocks
                            processing_tasks.append(
                                executor.process_text.remote(output_text)
                            )
                            response_indices.append(i)
                    
                    # Wait for all processing to complete
                    if processing_tasks:
                        logger.info(f"Waiting for {len(processing_tasks)} tool execution tasks to complete")
                        processed_texts = ray.get(processing_tasks)
                        
                        # Update responses with processed texts and continue generation
                        for idx, processed_text in zip(response_indices, processed_texts):
                            # First update the response text with the processed text
                            responses[idx].outputs[0].text = processed_text
                            
                            # Now continue generation with the processed text as the new prompt
                            # We need to tokenize the processed text and continue generation
                            try:
                                continued_prompt_tokens = self.llm.llm_engine.tokenizer.encode(processed_text)
                                
                                # Generate continuation
                                logger.info(f"Continuing generation for response {idx} after tool execution")
                                continued_response = self.llm.generate(
                                    sampling_params=sampling_params,
                                    prompt_token_ids=[continued_prompt_tokens]
                                )[0]
                                
                                # Update the response with the continued generation
                                responses[idx].outputs[0].text = continued_response.outputs[0].text
                            except Exception as e:
                                logger.error(f"Error continuing generation after tool execution: {str(e)}")
                                # Keep the processed text with tool outputs if continuation fails
                        
                        tool_execution_time = time.time() - generation_time
                        logger.info(f"Tool execution and continued generation completed in {tool_execution_time:.4f} seconds")
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


def create_tool_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=0.8,
    vllm_enable_sleep=False,
    tool_use_enabled=False,
    num_tool_executors=32,
):
    """
    Create VLLM engines with tool use capabilities.
    This is initially a copy of create_vllm_engines with minimal changes.
    
    Args:
        num_engines: Number of VLLM engines to create
        tensor_parallel_size: Tensor parallel size for each engine
        pretrain: Path to the pretrained model
        seed: Random seed
        enable_prefix_caching: Whether to enable prefix caching
        enforce_eager: Whether to enforce eager execution
        max_model_len: Maximum model length
        num_total_actors: Total number of actors
        shared_pg: Shared placement group
        gpu_memory_utilization: GPU memory utilization
        vllm_enable_sleep: Whether to enable sleep mode
        tool_use_enabled: Whether to enable tool use
        num_tool_executors: Number of parallel tool executors
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
            ToolLLMRayActor.options(
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
                enable_sleep_mode=vllm_enable_sleep,
                tool_use_enabled=tool_use_enabled,
                num_tool_executors=num_tool_executors,
            )
        )

    return vllm_engines 