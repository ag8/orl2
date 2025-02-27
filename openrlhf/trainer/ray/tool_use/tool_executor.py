"""
Tool executor for executing Python code.

This module provides a class for executing Python code and capturing the output.
"""

import io
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional, Any

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ToolExecutor:
    """
    Execute Python code and capture the output.
    
    This class provides methods for:
    1. Extracting Python code blocks from text
    2. Executing the code
    3. Capturing the output
    4. Injecting the output back into the text
    """
    
    def __init__(self, max_output_length: int = 10000):
        """
        Initialize the ToolExecutor.
        
        Args:
            max_output_length: Maximum length of captured output.
        """
        self.max_output_length = max_output_length
        logger.info(f"Initialized ToolExecutor with max_output_length: {max_output_length}")
    
    def extract_python_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Python code blocks from text.
        
        Args:
            text: The text to extract code blocks from.
            
        Returns:
            A list of dictionaries, each containing:
                - start: The start index of the code block
                - end: The end index of the code block
                - code: The code inside the block
        """
        pattern = r"<PYTHON>(.*?)</PYTHON>"
        blocks = []
        
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append({
                "start": match.start(),
                "end": match.end(),
                "code": match.group(1).strip()
            })
        
        return blocks
    
    def execute_code(self, code: str) -> str:
        """
        Execute Python code and capture the output.
        
        Args:
            code: The Python code to execute.
            
        Returns:
            The captured output (stdout + stderr).
        """
        print(f"TOOL_DEBUG: Executing Python code: {code[:100]}..." if len(code) > 100 else f"TOOL_DEBUG: Executing Python code: {code}")
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Execute the code in the global namespace
                exec(code, globals())
            
            # Get the output
            output = stdout.getvalue()
            
            # If there's no stdout but there is stderr, use stderr
            if not output and stderr.getvalue():
                output = stderr.getvalue()
                
        except Exception as e:
            # Capture the exception
            output = f"Error: {str(e)}\n{traceback.format_exc()}"
        
        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + f"\n... (output truncated, exceeded {self.max_output_length} characters)"
        
        print(f"TOOL_DEBUG: Code execution output: {output[:100]}..." if len(output) > 100 else f"TOOL_DEBUG: Code execution output: {output}")
        
        return output
    
    def process_text(self, text: str) -> str:
        """
        Process text by executing Python code blocks and injecting the output.
        
        Args:
            text: The text to process.
            
        Returns:
            The processed text with code outputs injected.
        """
        # Extract code blocks
        blocks = self.extract_python_blocks(text)
        
        # If no blocks, return the original text
        if not blocks:
            print(f"TOOL_DEBUG: No Python blocks found in text: {text[:100]}...")
            return text
        
        print(f"TOOL_DEBUG: Found {len(blocks)} Python blocks in text")
        
        # Process blocks in reverse order to avoid messing up indices
        blocks.reverse()
        
        # Process each block
        for block in blocks:
            start_idx = block["start"]
            end_idx = block["end"]
            code = block["code"]
            
            print(f"TOOL_DEBUG: Processing block at positions {start_idx}-{end_idx}")
            
            # Execute the code
            output = self.execute_code(code)
            
            # Insert the output after the code block
            text = (
                text[:end_idx] + 
                "\n<PYTHON-OUTPUT>\n" + output + "\n</PYTHON-OUTPUT>" + 
                text[end_idx:]
            )
            
            print(f"TOOL_DEBUG: Inserted <PYTHON-OUTPUT> tag after position {end_idx}")
        
        # Log the processed text to help with debugging
        print(f"TOOL_DEBUG: Processed text with {len(blocks)} Python blocks. First 100 chars: {text[:100]}...")
        print(f"TOOL_DEBUG: Does processed text contain <PYTHON-OUTPUT> tags? {'Yes' if '<PYTHON-OUTPUT>' in text else 'No'}")
        
        return text 