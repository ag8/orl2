import os
import json
import re
import logging
from datetime import datetime

import numpy as np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


# <tools>
def get_safe_globals():
    """Create a safe globals dictionary with allowed modules."""
    safe_modules = {
        'math': __import__('math'),
        'random': __import__('random'),
        'datetime': __import__('datetime'),
        'json': __import__('json'),
        'statistics': __import__('statistics'),
        'collections': __import__('collections'),
        'numpy': __import__('numpy'),
        're': __import__('re'),
        'requests': __import__('requests'),
        
        # Core chemistry libraries
        'rdkit': try_import('rdkit'),
        'rdkit.Chem': try_import('rdkit.Chem', fromlist=['Chem']),
        'rdkit.Chem.AllChem': try_import('rdkit.Chem.AllChem', fromlist=['AllChem']),
        'rdkit.Chem.Draw': try_import('rdkit.Chem.Draw', fromlist=['Draw']),
        
        # Molecular docking and analysis
        'openbabel': try_import('openbabel'),
        
        # Bio-informatics
        'Bio': try_import('Bio'),  # BioPython
        'Bio.PDB': try_import('Bio.PDB', fromlist=['PDB']),
        
        # Atomic Simulation Environment
        'ase': try_import('ase'),
    }
    
    return {
        '__builtins__': __builtins__,
        **{k: v for k, v in safe_modules.items() if v is not None}
    }

def try_import(module_name, fromlist=None):
    """Safely try to import a module, return None if not available."""
    try:
        if fromlist:
            return __import__(module_name, fromlist=fromlist)
        return __import__(module_name)
    except ImportError:
        logger.warning(f"{module_name} not available - some chemistry operations will be limited")
        return None

def run_python_code(code: str):
    from io import StringIO
    import sys
    
    # Create string buffer to capture output
    output_buffer = StringIO()
    # Store the original stdout
    original_stdout = sys.stdout
    
    try:
        # Log the exact code being executed with unique delimiters
        logger.info("-----BEGIN EXTRACTED CODE-----")
        logger.info(code)
        logger.info("-----END EXTRACTED CODE-----")
        
        # Redirect stdout to our buffer
        sys.stdout = output_buffer
        
        # Create a dictionary to store local variables
        local_dict = {}
        
        # Execute the code with safe globals
        exec(code, get_safe_globals(), local_dict)
        
        # Get any printed output
        printed_output = output_buffer.getvalue()
        
        # Log the exact execution result with unique delimiters
        logger.info("-----BEGIN EXECUTION RESULT-----")
        
        # If there was printed output, return that
        if printed_output.strip():
            result = printed_output.strip()
        # If there's a specific 'result' variable, return that
        elif 'result' in local_dict:
            result = str(local_dict['result'])
        # Otherwise return the last assigned variable
        elif local_dict:
            result = str(list(local_dict.values())[-1])
        # If nothing was printed or assigned, return success message
        else:
            result = "Code executed successfully"
            
        logger.info(result)
        logger.info("-----END EXECUTION RESULT-----")
        
        return result
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        logger.info("-----BEGIN EXECUTION RESULT-----")
        logger.info(error_msg)
        logger.info("-----END EXECUTION RESULT-----")
        return error_msg
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout
        output_buffer.close()


tools = {
    "run_python_code": run_python_code,
    "eval_python_code": run_python_code,  # Keep the alias for backward compatibility
}
# </tools>

@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:

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

        # Set up detailed logging for tool calls
        self.tool_logger = logging.getLogger('tool_calls')
        self.tool_logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create a new log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/tool_calls_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        
        # Set format for logging
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.tool_logger.addHandler(file_handler)

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
                
                # Instead of trying to decode the tokens, we'll use empty strings as initial prompts
                # since we're working with token IDs directly
                current_prompts = [""] * len(responses)
                
                self.tool_logger.info("="*80)
                self.tool_logger.info("NEW BATCH OF REQUESTS")
                self.tool_logger.info("="*80)
                self.tool_logger.info(f"Processing {len(responses)} responses")
                self.tool_logger.info("="*80)

                # Create patterns for all tool types
                tool_patterns = {
                    'PYTHON': (tools['run_python_code'], re.compile(r"<PYTHON>(.*?)</PYTHON>", re.DOTALL | re.IGNORECASE)),
                    'eval_python_code': (tools['eval_python_code'], re.compile(r"<EVAL>(.*?)</EVAL>", re.DOTALL | re.IGNORECASE)),
                }

                for i, response in enumerate(responses):
                    self.tool_logger.info(f"\nPROCESSING RESPONSE {i}")
                    self.tool_logger.info("-"*40)
                    self.tool_logger.info("Initial generation:")
                    self.tool_logger.info(f"{response.outputs[0].text}\n")
                    
                    original_response = response
                    tool_call_count = 0
                    tool_call_budget = 3
                    current_prompt = current_prompts[i]
                    
                    while tool_call_budget > 0:
                        # Find the earliest tool call of any type
                        earliest_match = None
                        earliest_pos = float('inf')
                        matched_tool = None
                        matched_func = None
                        
                        for tool_name, (func, pattern) in tool_patterns.items():
                            match = pattern.search(response.outputs[0].text)
                            if match and match.start() < earliest_pos:
                                earliest_match = match
                                earliest_pos = match.start()
                                matched_tool = tool_name
                                matched_func = func
                        
                        if earliest_match:
                            tool_call_count += 1
                            self.tool_logger.info(f"TOOL EXECUTION {tool_call_count} - {matched_tool}")
                            self.tool_logger.info("-"*30)
                            
                            code = earliest_match.group(1).strip()
                            self.tool_logger.info(f"Extracted code:\n{code}")
                            tool_call_budget -= 1
                            
                            self.tool_logger.info(f"Executing {matched_tool}...")
                            result = matched_func(code)
                            self.tool_logger.info(f"Execution result:\n{result}\n")
                            
                            # Build new prompt as a text string
                            prompt = (current_prompt + 
                                    response.outputs[0].text[:earliest_match.end()] + 
                                    f"\n<OUTPUT>{result}</OUTPUT>\n")
                            
                            self.tool_logger.info("Regenerating with new prompt:")
                            self.tool_logger.info("-"*30)
                            self.tool_logger.info(f"{prompt}")
                            self.tool_logger.info("-"*30)
                            
                            # Generate with text prompt instead of token IDs
                            response = self.llm.generate(sampling_params=sampling_params, prompts=[prompt])[0]
                            current_prompt = prompt  # Update current prompt for next iteration
                            self.tool_logger.info(f"New generation:\n{response.outputs[0].text}\n")
                        else:
                            break
                    
                    self.tool_logger.info(f"\nFINAL STATE OF RESPONSE {i}")
                    self.tool_logger.info("-"*40)
                    self.tool_logger.info(f"Total tool executions: {tool_call_count}")
                    self.tool_logger.info("Original response:")
                    self.tool_logger.info(f"{original_response.outputs[0].text}\n")
                    self.tool_logger.info("Final response:")
                    self.tool_logger.info(f"{response.outputs[0].text}\n")
                    self.tool_logger.info("="*80)
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
    vllm_enable_sleep=False,
):
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
                enable_sleep_mode=vllm_enable_sleep,
            )
        )

    return vllm_engines
