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
            "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>\n\nCool, so here\'s what this output means:"
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
                
                # Process responses with tool executors if enabled
                if self.tool_use_enabled and self.tool_executors:
                    generation_time = time.time()
                    print(f"TOOL_DEBUG: Processing {len(responses)} responses with tool executors")
                    
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
                            
                            print(f"TOOL_DEBUG: Response {i} contains Python code blocks, assigning to executor {executor_idx}")
                            
                            # Process the entire text with all code blocks
                            processing_tasks.append(
                                executor.process_text.remote(output_text)
                            )
                            response_indices.append(i)
                        else:
                            print(f"TOOL_DEBUG: Response {i} does not contain Python code blocks")
                    
                    # Wait for all processing to complete
                    if processing_tasks:
                        print(f"TOOL_DEBUG: Waiting for {len(processing_tasks)} tool execution tasks to complete")
                        processed_texts = ray.get(processing_tasks)
                        
                        # Check if processed texts contain PYTHON-OUTPUT tags
                        for idx, processed_text in enumerate(processed_texts):
                            print(f"TOOL_DEBUG: Processed text {idx} contains PYTHON-OUTPUT tags: {'Yes' if '<PYTHON-OUTPUT>' in processed_text else 'No'}")
                        
                        # Update responses with processed texts
                        for idx, processed_text in zip(response_indices, processed_texts):
                            responses[idx].outputs[0].text = processed_text
                            print(f"TOOL_DEBUG: Updated response {idx} with processed text")
                        
                        # Batch the continued generation
                        try:
                            # Prepare batch of prompts for continued generation
                            continued_prompt_tokens = []
                            continued_indices = []
                            processed_texts_map = {}  # Store processed texts for later use
                            
                            for idx, processed_text in zip(response_indices, processed_texts):
                                try:
                                    tokens = self.llm.llm_engine.tokenizer.encode(processed_text)
                                    continued_prompt_tokens.append(tokens)
                                    continued_indices.append(idx)
                                    processed_texts_map[idx] = processed_text  # Store for later
                                    print(f"TOOL_DEBUG: Tokenized processed text for response {idx}, token length: {len(tokens)}")
                                except Exception as e:
                                    print(f"TOOL_DEBUG: Error tokenizing processed text for response {idx}: {str(e)}")
                            
                            if continued_prompt_tokens:
                                # Generate continuations in batch
                                print(f"TOOL_DEBUG: Continuing generation for {len(continued_prompt_tokens)} responses after tool execution")
                                
                                # Log the batch size for debugging
                                print(f"TOOL_DEBUG: Batch size for continued generation: {len(continued_prompt_tokens)}")
                                
                                # Generate continuations in batch
                                continued_responses = self.llm.generate(
                                    sampling_params=sampling_params,
                                    prompt_token_ids=continued_prompt_tokens
                                )
                                
                                # Log the number of responses received
                                print(f"TOOL_DEBUG: Received {len(continued_responses)} continued responses")
                                
                                # Update the responses by preserving the processed text with PYTHON-OUTPUT tags
                                for i, idx in enumerate(continued_indices):
                                    processed_text = processed_texts_map[idx]
                                    continued_text = continued_responses[i].outputs[0].text
                                    
                                    # Log the first 100 chars of both texts for debugging
                                    print(f"TOOL_DEBUG: Response {idx} - Processed text first 100 chars: {processed_text[:100]}...")
                                    print(f"TOOL_DEBUG: Response {idx} - Continued text first 100 chars: {continued_text[:100]}...")
                                    
                                    # IMPORTANT: Print the full processed text to see if it contains PYTHON-OUTPUT tags
                                    print(f"TOOL_DEBUG: FULL PROCESSED TEXT FOR RESPONSE {idx}:")
                                    print("=" * 80)
                                    print(processed_text)
                                    print("=" * 80)
                                    
                                    # Check if the processed text contains PYTHON-OUTPUT tags
                                    if "<PYTHON-OUTPUT>" in processed_text:
                                        print(f"TOOL_DEBUG: Response {idx} - Processed text contains PYTHON-OUTPUT tags")
                                        # Find the last PYTHON-OUTPUT tag
                                        last_output_end = processed_text.rfind("</PYTHON-OUTPUT>")
                                        
                                        if last_output_end != -1:
                                            # Get everything up to and including the last output tag
                                            prefix = processed_text[:last_output_end + len("</PYTHON-OUTPUT>")]
                                            
                                            # Find where this same content ends in the continued text
                                            # The model might have regenerated some of the content
                                            if prefix in continued_text:
                                                # The continued text contains the full prefix
                                                suffix = continued_text[len(prefix):]
                                                final_text = prefix + suffix
                                                print(f"TOOL_DEBUG: Response {idx} - Found exact prefix match, appending suffix")
                                            else:
                                                # Try to find the last output tag in the continued text
                                                continued_last_tag = continued_text.rfind("</PYTHON-OUTPUT>")
                                                
                                                if continued_last_tag != -1:
                                                    # Use everything from the continued text after its last output tag
                                                    suffix = continued_text[continued_last_tag + len("</PYTHON-OUTPUT>"):]
                                                    final_text = prefix + suffix
                                                    print(f"TOOL_DEBUG: Response {idx} - Found output tag in continued text, appending suffix")
                                                else:
                                                    # The continued text doesn't have the output tags
                                                    # This is problematic - the model might have lost context
                                                    # Just use the processed text to preserve the outputs
                                                    final_text = processed_text
                                                    print(f"TOOL_DEBUG: Response {idx} - Continued text lost output tags, using processed text")
                                        else:
                                            # This shouldn't happen if we found PYTHON-OUTPUT earlier
                                            final_text = continued_text
                                            print(f"TOOL_DEBUG: Response {idx} - Inconsistent state: PYTHON-OUTPUT found but no closing tag")
                                    else:
                                        # No PYTHON-OUTPUT tags, just use the continued text
                                        final_text = continued_text
                                        print(f"TOOL_DEBUG: Response {idx} - No PYTHON-OUTPUT tags found, using continued text")
                                    
                                    # Update the response
                                    responses[idx].outputs[0].text = final_text
                                    
                                    # Log if the final text contains PYTHON-OUTPUT tags
                                    if "<PYTHON-OUTPUT>" in final_text:
                                        print(f"TOOL_DEBUG: Response {idx} - Final text contains PYTHON-OUTPUT tags")
                                    else:
                                        print(f"TOOL_DEBUG: Response {idx} - Final text does NOT contain PYTHON-OUTPUT tags")
                                
                        except Exception as e:
                            print(f"TOOL_DEBUG: Error in batch continued generation: {str(e)}")
                            # Keep the processed texts with tool outputs if continuation fails
                        
                        tool_execution_time = time.time() - generation_time
                        print(f"TOOL_DEBUG: Tool execution and continued generation completed in {tool_execution_time:.4f} seconds")
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