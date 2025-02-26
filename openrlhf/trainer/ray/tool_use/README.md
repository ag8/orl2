# Tool Use for VLLM in OpenRLHF

This module adds tool use capabilities to VLLM in OpenRLHF, allowing models to execute Python code during generation.

## Overview

The implementation consists of:

1. **ToolExecutor**: A class that executes Python code in a sandbox and captures the output
2. **ToolLLMRayActor**: A modified version of LLMRayActor that supports tool use
3. **Test scripts**: Scripts to verify the implementation works correctly

## How It Works

When tool use is enabled:

1. The model generates text as usual
2. When Python code blocks are detected (enclosed in `<PYTHON></PYTHON>` tags), the code is executed
3. The output is injected back into the generation as `<PYTHON-OUTPUT></PYTHON-OUTPUT>` blocks
4. The model can then continue generating based on the output

## Testing

### Testing the Tool Executor

To test the tool executor independently:

```bash
python -m openrlhf.trainer.ray.tool_use.test_tool_use
```

This script runs several tests to verify that Python code execution works correctly.

### Testing the Tool-Enabled VLLM Engine

To test the tool-enabled VLLM engine:

```bash
python -m openrlhf.trainer.ray.tool_use.test_tool_engine --model <model_path> --tool-use-enabled
```

Replace `<model_path>` with the path to your model.

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