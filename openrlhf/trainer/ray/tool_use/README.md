# Tool Use for VLLM in OpenRLHF

This module adds tool use capabilities to VLLM in OpenRLHF, allowing models to execute Python code during generation.

## Overview

The implementation consists of:

1. **ToolExecutor**: A class that executes Python code and captures the output
2. **ToolLLMRayActor**: A modified version of LLMRayActor that supports tool use
3. **Parallel execution**: Multiple tool executors running in parallel for better performance

## How It Works

When tool use is enabled:

1. The model generates text as usual
2. When Python code blocks are detected (enclosed in `<PYTHON></PYTHON>` tags), the code is executed
3. The output is injected back into the generation as `<PYTHON-OUTPUT></PYTHON-OUTPUT>` blocks
4. The model can then continue generating based on the output

## Using Tool Use with PPO Training

To enable tool use in your PPO training, add the following parameters to your command:

```bash
python -m openrlhf.cli.train_ppo_ray \
    --enable_tool_use \
    --num_tool_executors 32 \
    --tool_config_path /path/to/tool_config.json \
    ... other parameters ...
```

You can also submit a Ray job:

```bash
ray job submit --address="http://127.0.0.1:8265" \
-- python3 -m openrlhf.cli.train_ppo_ray \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 1 \
--reward_num_nodes 1 \
--reward_num_gpus_per_node 1 \
--critic_num_nodes 1 \
--critic_num_gpus_per_node 1 \
--actor_num_nodes 1 \
--actor_num_gpus_per_node 1 \
--vllm_num_engines 1 \
--vllm_tensor_parallel_size 1 \
--enable_tool_use \
--num_tool_executors 32 \
--tool_config_path /root/orl2/openrlhf/trainer/ray/tool_use/tool_config.json \
... other parameters ...
```

## Tool Configuration

You can customize the tool behavior by creating a JSON configuration file. Here's an example:

```json
{
    "tools": [
        {
            "name": "python_executor",
            "description": "Execute Python code and return the output",
            "enabled": true,
            "max_output_length": 1000,
            "timeout_seconds": 10
        }
    ],
    "execution": {
        "max_tools_per_generation": 1,
        "parallel_execution": true,
        "num_executors": 32
    },
    "security": {
        "allowed_modules": [
            "math",
            "random",
            "datetime",
            "time",
            "re",
            "json",
            "collections",
            "itertools",
            "functools",
            "numpy"
        ],
        "restricted_functions": [
            "eval",
            "exec",
            "compile",
            "open",
            "file",
            "__import__",
            "input"
        ]
    }
}
```

## Testing

### Testing the Tool Executor

To test the tool executor independently:

```bash
python -m openrlhf.trainer.ray.tool_use.test_tool_use
```

### Testing the Tool-Enabled VLLM Engine

To test the tool-enabled VLLM engine:

```bash
python -m openrlhf.trainer.ray.tool_use.test_tool_engine --model <model_path> --tool-use-enabled --num-tool-executors 32
```

## Benchmarking

To benchmark the performance of the tool-enabled VLLM engine:

```bash
python -m openrlhf.trainer.ray.tool_use.benchmark_tool_engine --model <model_path> --num-batches 3 --num-tool-executors 32
```

## Development Roadmap

1. ✅ Create a basic implementation that can execute Python code
2. ✅ Set up testing infrastructure
3. 🔄 Implement tool use in the VLLM engine
4. 🔄 Add support for streaming generation with tool use
5. 🔄 Benchmark performance
6. 🔄 Add support for other types of tools

## Implementation Details

### ToolExecutor

The `ToolExecutor` class:
- Extracts Python code blocks from text
- Executes the code in a restricted environment
- Captures stdout/stderr
- Returns the output to be injected back into the generation

### ToolLLMRayActor

The `ToolLLMRayActor` class:
- Extends the original `LLMRayActor` class
- Adds tool use capabilities
- Processes generated text to execute Python code
- Injects the output back into the generation

## Security Considerations

The `ToolExecutor` runs Python code in a restricted environment:
- Only a limited set of modules can be imported
- Dangerous operations are blocked
- Execution time is limited
- Output length is limited

## Future Work

- Add support for other types of tools (e.g., web search, API calls)
- Implement streaming generation with tool use
- Add support for interactive tools
- Improve error handling and security 