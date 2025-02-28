"""
Tool executor for executing Python code.

This module provides a class for executing Python code and capturing the output.
"""

import io
import re
import sys
import traceback
import subprocess
import tempfile
import os
import signal
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional, Any, Tuple, Set

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ToolExecutor:
    """
    Execute Python code and capture the output.
    
    This class provides methods for:
    1. Extracting Python code blocks from text
    2. Executing the code in an isolated subprocess
    3. Capturing the output
    4. Injecting the output back into the text
    5. Supporting multiple iterations of code execution (up to a limit)
    """
    
    def __init__(self, max_output_length: int = 10000, timeout_seconds: int = 30, max_executions: int = 3):
        """
        Initialize the ToolExecutor.
        
        Args:
            max_output_length: Maximum length of captured output.
            timeout_seconds: Maximum execution time in seconds.
            max_executions: Maximum number of Python code executions per generation.
        """
        self.max_output_length = max_output_length
        self.timeout_seconds = timeout_seconds
        self.max_executions = max_executions
        self.temp_file_path = None
        logger.info(f"Initialized ToolExecutor with max_output_length: {max_output_length}, timeout: {timeout_seconds}s, max_executions: {max_executions}")
    
    def extract_python_blocks(self, text: str, exclude_positions: Set[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Extract Python code blocks from text.
        
        Args:
            text: The text to extract code blocks from.
            exclude_positions: Set of (start, end) positions to exclude from extraction.
            
        Returns:
            A list of dictionaries, each containing:
                - start: The start index of the code block
                - end: The end index of the code block
                - code: The code inside the block
        """
        pattern = r"<PYTHON>(.*?)</PYTHON>"
        blocks = []
        exclude_positions = exclude_positions or set()
        
        for match in re.finditer(pattern, text, re.DOTALL):
            start_idx = match.start()
            end_idx = match.end()
            
            # Skip this block if it's in the exclude list
            if any(start == start_idx and end == end_idx for start, end in exclude_positions):
                continue
                
            blocks.append({
                "start": start_idx,
                "end": end_idx,
                "code": match.group(1).strip()
            })
        
        return blocks
    
    def execute_code(self, code: str, is_first_execution: bool = False) -> str:
        """
        Execute Python code in an isolated subprocess and capture the output.
        
        Args:
            code: The Python code to execute.
            is_first_execution: Whether this is the first execution in a sequence.
            
        Returns:
            The captured output (stdout + stderr).
        """
        print(f"TOOL_DEBUG: Executing Python code: {code[:100]}..." if len(code) > 100 else f"TOOL_DEBUG: Executing Python code: {code}")
        
        # For the first execution, create a new temp file
        # For subsequent executions, append to the existing file to maintain state
        if is_first_execution or self.temp_file_path is None:
            # Create a temporary file to hold the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                # Write a wrapper that captures all output
                temp_file.write("""
import sys
import traceback

# Redirect stdout and stderr to capture all output
class CaptureOutput:
    def __init__(self):
        self.value = ""
    
    def write(self, text):
        self.value += text
    
    def flush(self):
        pass

stdout_capture = CaptureOutput()
stderr_capture = CaptureOutput()

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # Execute the user code
""")
                # Indent the user code
                indented_code = "\n".join(f"    {line}" for line in code.split("\n"))
                temp_file.write(indented_code)
                
                # Add code to print the captured output
                temp_file.write("""
    
    # Print the captured output
    output = stdout_capture.value
    
    # If there's no stdout but there is stderr, use stderr
    if not output and stderr_capture.value:
        output = stderr_capture.value
    
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(output, end='')
    
except Exception as e:
    # Capture the exception
    error_output = f"Error: {str(e)}\\n{traceback.format_exc()}"
    
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(error_output, end='')
""")
                self.temp_file_path = temp_file.name
        else:
            # For subsequent executions, append to the existing file to maintain state
            with open(self.temp_file_path, 'a') as temp_file:
                temp_file.write("\n\n# New execution\n")
                temp_file.write("try:\n")
                # Indent the user code
                indented_code = "\n".join(f"    {line}" for line in code.split("\n"))
                temp_file.write(indented_code)
                
                # Add code to print the captured output
                temp_file.write("""
    
    # Print the captured output
    output = stdout_capture.value
    
    # If there's no stdout but there is stderr, use stderr
    if not output and stderr_capture.value:
        output = stderr_capture.value
    
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(output, end='')
    
except Exception as e:
    # Capture the exception
    error_output = f"Error: {str(e)}\\n{traceback.format_exc()}"
    
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(error_output, end='')
""")
        
        try:
            # Execute the code in a separate process with timeout
            result = subprocess.run(
                [sys.executable, self.temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            # Get the output
            output = result.stdout
            
            # If there's no stdout but there is stderr, use stderr
            if not output and result.stderr:
                output = result.stderr
                
        except subprocess.TimeoutExpired:
            output = f"Error: Code execution timed out after {self.timeout_seconds} seconds."
            # Clear the temp file on timeout error
            self._clear_temp_file()
        except Exception as e:
            output = f"Error in subprocess execution: {str(e)}"
            # Clear the temp file on execution error
            self._clear_temp_file()
        
        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + f"\n... (output truncated, exceeded {self.max_output_length} characters)"
        
        print(f"TOOL_DEBUG: Code execution output: {output[:100]}..." if len(output) > 100 else f"TOOL_DEBUG: Code execution output: {output}")
        
        # Check if the output contains an error message
        if output.startswith("Error:") or "Traceback (most recent call last)" in output:
            print(f"TOOL_DEBUG: Error detected in code execution. Clearing temporary file.")
            self._clear_temp_file()
        
        return output
    
    def _clear_temp_file(self):
        """
        Clear the temporary file and reset the path.
        This is called when an error occurs during code execution.
        """
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
                print(f"TOOL_DEBUG: Deleted temporary file {self.temp_file_path} due to error")
            except Exception as e:
                print(f"TOOL_DEBUG: Failed to delete temporary file: {str(e)}")
            finally:
                self.temp_file_path = None
    
    def process_text(self, text: str) -> str:
        """
        Process text by executing Python code blocks and injecting the output.
        
        Args:
            text: The text to process.
            
        Returns:
            The processed text with code outputs injected.
        """
        # Reset the temp file path for a new processing session
        self.temp_file_path = None
        
        # Track which blocks have been processed
        processed_blocks = set()
        
        # Track the number of executions performed
        executions_performed = 0
        
        # Process the text
        result_text = text
        
        # Continue processing until we've reached the maximum number of executions
        # or there are no more blocks to process
        while executions_performed < self.max_executions:
            # Extract code blocks, excluding those we've already processed
            blocks = self.extract_python_blocks(result_text, processed_blocks)
            
            # If no blocks, break the loop
            if not blocks:
                print(f"TOOL_DEBUG: No new Python blocks found in text. Stopping after {executions_performed} executions.")
                break
            
            # Process the last block
            block = blocks[-1]  # Take only the last block
            start_idx = block["start"]
            end_idx = block["end"]
            code = block["code"]
            
            print(f"TOOL_DEBUG: Processing block at positions {start_idx}-{end_idx} (execution {executions_performed + 1}/{self.max_executions})")
            
            # Mark this block as processed
            processed_blocks.add((start_idx, end_idx))
            
            # Execute the code, marking the first execution in the sequence
            is_first = (executions_performed == 0)
            output = self.execute_code(code, is_first_execution=is_first)
            
            # Insert the output after the code block
            result_text = (
                result_text[:end_idx] + 
                "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>" + 
                result_text[end_idx:]
            )
            
            print(f"TOOL_DEBUG: Inserted <PYTHON-OUTPUT> tag after position {end_idx}")
            
            # Increment the execution count
            executions_performed += 1
            
            # Log the processed text to help with debugging
            print(f"TOOL_DEBUG: Processed text with 1 Python block. Executions performed: {executions_performed}/{self.max_executions}")
        
        # If we've reached the maximum number of executions and there are still more blocks,
        # add a warning message
        if executions_performed >= self.max_executions:
            # Check if there are more blocks to process
            more_blocks = self.extract_python_blocks(result_text, processed_blocks)
            if more_blocks:
                logger.info(f"Reached maximum number of Python executions ({self.max_executions}). Adding warning message.")
                
                # Add warning message after the last Python output
                last_python_output_end = result_text.rfind("</PYTHON-OUTPUT>")
                if last_python_output_end != -1:
                    last_python_output_end += len("</PYTHON-OUTPUT>")
                    warning_message = "\n\n[WARNING: You have reached your Python execution quota. No further Python code will be executed.]\n\nAll right,"
                    result_text = result_text[:last_python_output_end] + warning_message + result_text[last_python_output_end:]
                    print(f"TOOL_DEBUG: Added warning message about reaching Python execution quota.")
        
        # Clean up temp file if it exists
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
                self.temp_file_path = None
            except:
                pass
        
        return result_text 
