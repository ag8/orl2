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
        print(f"TOOL_DEBUG: Initialized ToolExecutorActor with max_output_length={max_output_length}")
    
    def execute_code(self, code: str) -> str:
        """Execute Python code and return the output."""
        print(f"TOOL_DEBUG: ToolExecutorActor executing code: {code[:50]}...")
        result = self.executor.execute_code(code)
        print(f"TOOL_DEBUG: ToolExecutorActor execution result: {result[:50]}...")
        return result
    
    def process_text(self, text: str) -> str:
        """Process text by executing Python code blocks and injecting the output."""
        print(f"TOOL_DEBUG: ToolExecutorActor processing text: {text[:50]}...")
        result = self.executor.process_text(text)
        print(f"TOOL_DEBUG: ToolExecutorActor processing result contains PYTHON-OUTPUT: {'Yes' if '<PYTHON-OUTPUT>' in result else 'No'}")
        return result
    
    def process_block(self, text: str, start_idx: int, end_idx: int, code: str) -> str:
        """Process a single code block and return the modified text."""
        # Execute the code
        print(f"TOOL_DEBUG: ToolExecutorActor processing block at {start_idx}-{end_idx}")
        output = self.execute_code(code)
        
        # Insert the output after the code block
        result = (
            text[:end_idx] + 
            "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>"
        )
        
        print(f"TOOL_DEBUG: ToolExecutorActor block processing result contains PYTHON-OUTPUT: {'Yes' if '<PYTHON-OUTPUT>' in result else 'No'}")
        return result


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
        
        # Print a clear marker to verify this engine is being used
        print("*" * 80)
        print("INITIALIZING TOOL-ENABLED VLLM ENGINE")
        print(f"Tool use enabled: {self.tool_use_enabled}")
        print(f"Number of tool executors: {self.num_tool_executors}")
        print("*" * 80)
        
        # Create tool executor actors if tool use is enabled
        self.tool_executors = []
        if self.tool_use_enabled:
            print(f"Creating {self.num_tool_executors} tool executor actors")
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
                
                # EXTREME DEBUG: Replace all response texts with a fixed string
                for i, response in enumerate(responses):
                    # Replace the entire text with a fixed string that would be impossible to miss
                    response.outputs[0].text = "TOOL_ENGINE_REPLACEMENT_TEXT_POTATO_POTATO_POTATO"
                    print(f"TOOL_DEBUG: Replaced response {i} with fixed string")
                
                # Process responses with tool executors if enabled
                if self.tool_use_enabled and self.tool_executors:
                    generation_time = time.time()
                    print(f"TOOL_DEBUG: Processing {len(responses)} responses with tool executors")
                    
                    # EXTREME DEBUG: Skip all tool execution and just keep our fixed string
                    print(f"TOOL_DEBUG: EXTREME DEBUG MODE - Skipping all tool execution and keeping fixed string")
                    
                    # Skip all the tool execution code
                    tool_execution_time = time.time() - generation_time
                    print(f"TOOL_DEBUG: Tool execution skipped in {tool_execution_time:.4f} seconds")
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
        responses = self.responses.pop(actor_rank)
        
        # EXTREME DEBUG: Make sure our fixed string is preserved
        for i, response in enumerate(responses):
            # Double-check that our fixed string is still there
            if "POTATO" not in response.outputs[0].text:
                # If not, replace it again
                response.outputs[0].text = "TOOL_ENGINE_REPLACEMENT_TEXT_POTATO_POTATO_POTATO"
                print(f"TOOL_DEBUG: Re-added fixed string to response {i} before returning to actor")
            else:
                print(f"TOOL_DEBUG: Fixed string still present in response {i}")
        
        return responses


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

    # Print a clear marker to verify this function is being called
    print("*" * 80)
    print("CREATING TOOL-ENABLED VLLM ENGINES")
    print(f"Number of engines: {num_engines}")
    print(f"Tool use enabled: {tool_use_enabled}")
    print(f"Number of tool executors: {num_tool_executors}")
    print("*" * 80)

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