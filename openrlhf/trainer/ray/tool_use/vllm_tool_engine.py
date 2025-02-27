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
    
    def __init__(self, max_output_length: int = 10000, timeout_seconds: int = 30):
        self.max_output_length = max_output_length
        self.timeout_seconds = timeout_seconds
        self.executor = ToolExecutor(max_output_length=max_output_length, timeout_seconds=timeout_seconds)
        print(f"TOOL_DEBUG: Initialized ToolExecutorActor with max_output_length={max_output_length}, timeout={timeout_seconds}s")
    
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
        self.max_output_length = kwargs.pop("max_output_length", 10000)  # Default to 10000 characters
        self.timeout_seconds = kwargs.pop("timeout_seconds", 30)  # Default to 30 seconds timeout
        
        # Print a clear marker to verify this engine is being used
        print("*" * 80)
        print("INITIALIZING TOOL-ENABLED VLLM ENGINE")
        print(f"Tool use enabled: {self.tool_use_enabled}")
        print(f"Number of tool executors: {self.num_tool_executors}")
        print(f"Max output length: {self.max_output_length}")
        print(f"Timeout seconds: {self.timeout_seconds}")
        print("*" * 80)
        
        # Create tool executor actors if tool use is enabled
        self.tool_executors = []
        if self.tool_use_enabled:
            print(f"Creating {self.num_tool_executors} tool executor actors with max_output_length={self.max_output_length}, timeout={self.timeout_seconds}s")
            for _ in range(self.num_tool_executors):
                self.tool_executors.append(ToolExecutorActor.remote(max_output_length=self.max_output_length, timeout_seconds=self.timeout_seconds))

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

    def _update_token_ids(self, response, new_text, original_token_ids):
        """
        Helper method to update token IDs for a response to match new text.
        Handles truncation, padding, and error conditions.
        
        Args:
            response: The response object to update
            new_text: The new text to tokenize
            original_token_ids: The original token IDs for reference
            
        Returns:
            The updated token IDs
        """
        original_length = len(original_token_ids)
        
        try:
            # Try to tokenize the new text
            new_token_ids = self.llm.llm_engine.tokenizer.encode(new_text)
            
            # For continuations, we want to allow longer token sequences
            # Only truncate if it's extremely long (more than 3x the original)
            max_allowed_length = max(original_length * 3, 4096)  # todo: make this match the maximum generation length from the configuration
            
            if len(new_token_ids) > max_allowed_length:
                print(f"TOOL_DEBUG: Truncating extremely long token IDs from {len(new_token_ids)} to {max_allowed_length}")
                new_token_ids = new_token_ids[:max_allowed_length]
            elif len(new_token_ids) > original_length:
                print(f"TOOL_DEBUG: Allowing longer token sequence: {len(new_token_ids)} vs original {original_length}")
            # If they're too short, pad with a safe token
            elif len(new_token_ids) < original_length:
                safe_token = 1  # Default safe token
                if len(original_token_ids) > 0:
                    safe_token = original_token_ids[0]
                padding = [safe_token] * (original_length - len(new_token_ids))
                print(f"TOOL_DEBUG: Padding token IDs from {len(new_token_ids)} to {original_length}")
                new_token_ids = new_token_ids + padding
            
            return new_token_ids
            
        except Exception as e:
            print(f"TOOL_DEBUG: Error updating token IDs: {str(e)}")
            # If tokenization fails, use a safe fallback
            safe_token = 1
            if len(original_token_ids) > 0:
                safe_token = original_token_ids[0]
            return [safe_token] * original_length

    def _process_responses_with_code_execution(self, responses, sampling_params):
        """
        Process responses that may contain Python code blocks:
        1. Extract code blocks
        2. Execute the code in parallel
        3. Insert output after code blocks
        4. Generate continuations
        
        Args:
            responses: List of LLM responses
            sampling_params: Sampling parameters for continuation generation
            
        Returns:
            Updated responses with code execution outputs
        """
        if not self.tool_use_enabled or not self.tool_executors:
            return responses
            
        generation_time = time.time()
        print(f"TOOL_DEBUG: Processing {len(responses)} responses with tool executors")
        
        # Find responses containing Python code blocks
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
                processing_tasks.append(executor.process_text.remote(output_text))
                response_indices.append(i)
            else:
                print(f"TOOL_DEBUG: Response {i} does not contain Python code blocks")
        
        # Wait for all processing to complete
        if not processing_tasks:
            return responses
            
        print(f"TOOL_DEBUG: Waiting for {len(processing_tasks)} tool execution tasks to complete")
        processed_texts = ray.get(processing_tasks)
        
        # Update responses with processed texts (containing <PYTHON-OUTPUT> tags)
        for idx, processed_text in zip(response_indices, processed_texts):
            # Store the original token IDs for length reference
            original_token_ids = responses[idx].outputs[0].token_ids
            
            # Update the text with executed code outputs
            responses[idx].outputs[0].text = processed_text
            print(f"TOOL_DEBUG: Updated response {idx} with processed text")
            
            # Update token IDs to match the new text
            new_token_ids = self._update_token_ids(
                responses[idx], processed_text, original_token_ids
            )
            responses[idx].outputs[0].token_ids = new_token_ids
        
        # Generate continuations for responses with Python outputs
        self._generate_continuations(responses, response_indices, processed_texts, sampling_params)
        
        tool_execution_time = time.time() - generation_time
        print(f"TOOL_DEBUG: Tool execution and continued generation completed in {tool_execution_time:.4f} seconds")
        
        return responses

    def _generate_continuations(self, responses, response_indices, processed_texts, sampling_params):
        """
        Generate continuations for responses that have Python outputs.
        
        Args:
            responses: List of LLM responses
            response_indices: Indices of responses that were processed
            processed_texts: List of processed texts with Python outputs
            sampling_params: Sampling parameters for generation
        """
        try:
            # Prepare prompts for continuation generation
            continuation_prompts = []
            continuation_indices = []
            processed_texts_map = {}  # Store processed texts for later use
            
            for idx, processed_text in zip(response_indices, processed_texts):
                if "<PYTHON-OUTPUT>" not in processed_text:
                    print(f"TOOL_DEBUG: Response {idx} - No PYTHON-OUTPUT tags found, skipping continuation")
                    continue
                    
                try:
                    # Create a continuation prompt with analyzed output
                    last_output_end = processed_text.rfind("</PYTHON-OUTPUT>")
                    
                    if last_output_end == -1:
                        print(f"TOOL_DEBUG: Response {idx} - Inconsistent state: PYTHON-OUTPUT found but no closing tag")
                        continue
                        
                    # Take everything up to and including the last PYTHON-OUTPUT tag
                    continuation_prompt = processed_text[:last_output_end + len("</PYTHON-OUTPUT>")]
                    # Add prompt for continuation
                    continuation_prompt += "\n\nCool, so here's what this output means: "
                    
                    # Tokenize the continuation prompt
                    tokens = self.llm.llm_engine.tokenizer.encode(continuation_prompt)
                    
                    continuation_prompts.append(tokens)
                    continuation_indices.append(idx)
                    processed_texts_map[idx] = {
                        "processed_text": processed_text,
                        "continuation_prompt": continuation_prompt
                    }
                    print(f"TOOL_DEBUG: Created continuation prompt for response {idx}, token length: {len(tokens)}")
                except Exception as e:
                    print(f"TOOL_DEBUG: Error creating continuation prompt for response {idx}: {str(e)}")
            
            if not continuation_prompts:
                print("TOOL_DEBUG: No continuation prompts created, skipping batch generation")
                return
            
            # Create a new sampling_params object with a higher max_tokens value for continuation
            # This ensures we have enough tokens for a substantial continuation
            from copy import deepcopy
            from vllm import SamplingParams
            
            # Clone the original sampling params
            continuation_sampling_params = sampling_params.clone()
            
            # Get the original max_tokens value
            original_max_tokens = getattr(continuation_sampling_params, "max_tokens", 1024)
            
            # Set a higher max_tokens value for continuation (at least 1000 tokens)
            # If the original max_tokens is already high, use that
            continuation_max_tokens = max(1000, original_max_tokens)
            
            # Update the max_tokens value
            continuation_sampling_params.max_tokens = continuation_max_tokens
            
            # Set a minimum number of tokens to generate
            continuation_sampling_params.min_tokens = 100
            
            print(f"TOOL_DEBUG: Original max_tokens: {original_max_tokens}, Continuation max_tokens: {continuation_max_tokens}")
            print(f"TOOL_DEBUG: Continuation sampling parameters: {continuation_sampling_params}")
                
            # Generate continuations in batch with the new sampling params
            print(f"TOOL_DEBUG: Continuing generation for {len(continuation_prompts)} responses")
            continued_responses = self.llm.generate(
                sampling_params=continuation_sampling_params,
                prompt_token_ids=continuation_prompts
            )
            
            # Update responses with continuations
            for i, idx in enumerate(continuation_indices):
                try:
                    data = processed_texts_map[idx]
                    processed_text = data["processed_text"]
                    continuation_prompt = data["continuation_prompt"]
                    
                    # The continued_text is the model's response - it's already just the continuation
                    # We don't need to extract anything from it
                    continuation = continued_responses[i].outputs[0].text
                    
                    print(f"TOOL_DEBUG: Response {idx} - Got continuation of length {len(continuation)}")
                    print(f"TOOL_DEBUG: Response {idx} - Continuation preview: {continuation[:100]}...")
                    
                    # Check if the continuation is too short
                    if len(continuation) < 50:
                        print(f"TOOL_DEBUG: WARNING - Response {idx} has a very short continuation ({len(continuation)} chars)")
                    
                    # Create the final text with the processed output and the continuation
                    last_output_end = processed_text.rfind("</PYTHON-OUTPUT>")
                    final_text = processed_text[:last_output_end + len("</PYTHON-OUTPUT>")]
                    final_text += "\n\nCool, so here's what this output means: " + continuation
                    
                    # Update the response and its token IDs
                    original_token_ids = responses[idx].outputs[0].token_ids
                    responses[idx].outputs[0].text = final_text
                    responses[idx].outputs[0].token_ids = self._update_token_ids(
                        responses[idx], final_text, original_token_ids
                    )
                    
                    # Log the final text length
                    print(f"TOOL_DEBUG: Response {idx} - Final text length: {len(final_text)}")
                    print(f"TOOL_DEBUG: Response {idx} - Updated with continuation")
                except Exception as e:
                    print(f"TOOL_DEBUG: Error processing continuation for response {idx}: {str(e)}")
                    # Use a default continuation if there's an error
                    try:
                        processed_text = processed_texts_map[idx]["processed_text"]
                        last_output_end = processed_text.rfind("</PYTHON-OUTPUT>")
                        final_text = processed_text[:last_output_end + len("</PYTHON-OUTPUT>")]
                        final_text += "\n\nBased on this output, I can analyze that "
                        
                        # Update the response and its token IDs
                        original_token_ids = responses[idx].outputs[0].token_ids
                        responses[idx].outputs[0].text = final_text
                        responses[idx].outputs[0].token_ids = self._update_token_ids(
                            responses[idx], final_text, original_token_ids
                        )
                        print(f"TOOL_DEBUG: Response {idx} - Used default continuation due to error")
                    except Exception as nested_e:
                        print(f"TOOL_DEBUG: Failed to apply default continuation for response {idx}: {str(nested_e)}")
                
        except Exception as e:
            print(f"TOOL_DEBUG: Error in batch continuation generation: {str(e)}")
            import traceback
            print(f"TOOL_DEBUG: {traceback.format_exc()}")

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
                    responses = self._process_responses_with_code_execution(responses, sampling_params)
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
        
        # Verify that both text and token IDs are properly set for each response
        for i, response in enumerate(responses):
            # Check if the text and token IDs are consistent
            text_length = len(response.outputs[0].text)
            token_ids_length = len(response.outputs[0].token_ids)
            
            print(f"TOOL_DEBUG: Response {i} - Text length: {text_length}, Token IDs length: {token_ids_length}")
            
            # If there's a mismatch, log a warning
            if text_length > 0 and token_ids_length == 0:
                print(f"TOOL_DEBUG: WARNING - Response {i} has text but no token IDs")
                # Create token IDs from the text as a fallback
                try:
                    token_ids = self.llm.llm_engine.tokenizer.encode(response.outputs[0].text)
                    response.outputs[0].token_ids = token_ids
                    print(f"TOOL_DEBUG: Created token IDs for response {i}, length: {len(token_ids)}")
                except Exception as e:
                    print(f"TOOL_DEBUG: Error creating token IDs for response {i}: {str(e)}")
                    # Use a safe fallback
                    response.outputs[0].token_ids = [1] * min(100, text_length)  # Use a reasonable default length
            
            # Final verification
            print(f"TOOL_DEBUG: Final response {i} - Text: {response.outputs[0].text[:50]}..., Token IDs length: {len(response.outputs[0].token_ids)}")
        
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
    max_output_length=10000,
    timeout_seconds=30,
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
        max_output_length: Maximum length of captured output from tool execution
        timeout_seconds: Maximum execution time in seconds for tool execution
    """
    import vllm

    # Print a clear marker to verify this function is being called
    print("*" * 80)
    print("CREATING TOOL-ENABLED VLLM ENGINES")
    print(f"Number of engines: {num_engines}")
    print(f"Tool use enabled: {tool_use_enabled}")
    print(f"Number of tool executors: {num_tool_executors}")
    print(f"Max output length: {max_output_length}")
    print(f"Timeout seconds: {timeout_seconds}")
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
                max_output_length=max_output_length,
                timeout_seconds=timeout_seconds,
            )
        )

    return vllm_engines 