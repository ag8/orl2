"""
Test script for the ToolExecutor class.

This script tests the functionality of the ToolExecutor class
to ensure it correctly executes Python code and processes text.
"""

import argparse
from openrlhf.trainer.ray.tool_use.tool_executor import ToolExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Test the tool executor")
    parser.add_argument("--max-output-length", type=int, default=1000, help="Maximum output length")
    return parser.parse_args()


def test_basic_execution():
    """Test basic Python code execution."""
    executor = ToolExecutor()
    
    # Test a simple calculation
    text = """Let me calculate 2+2:
<PYTHON>
print(2 + 2)
</PYTHON>
"""
    
    result = executor.process_text(text)
    print("Basic execution test:")
    print(result)
    print("-" * 50)


def test_multiple_blocks():
    """Test multiple Python code blocks in a single text."""
    executor = ToolExecutor()
    
    text = """Let's calculate some values:
<PYTHON>
print("The sum of 1 to 10 is:", sum(range(1, 11)))
</PYTHON>

Now let's calculate a factorial:
<PYTHON>
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

print("5! =", factorial(5))
</PYTHON>
"""
    
    result = executor.process_text(text)
    print("Multiple blocks test:")
    print(result)
    print("-" * 50)


def test_error_handling():
    """Test error handling in Python code execution."""
    executor = ToolExecutor()
    
    text = """Let's try some code that will cause an error:
<PYTHON>
print(1/0)
</PYTHON>
"""
    
    result = executor.process_text(text)
    print("Error handling test:")
    print(result)
    print("-" * 50)


def test_module_imports():
    """Test importing and using modules."""
    executor = ToolExecutor()
    
    text = """Let's use some Python modules:
<PYTHON>
import math
import random

# Calculate the square root of 16
print("Square root of 16:", math.sqrt(16))

# Generate a random number between 1 and 100
print("Random number:", random.randint(1, 100))

# Use datetime
import datetime
print("Current date and time:", datetime.datetime.now())
</PYTHON>
"""
    
    result = executor.process_text(text)
    print("Module imports test:")
    print(result)
    print("-" * 50)


def main():
    args = parse_args()
    
    # Create a tool executor with the specified parameters
    executor = ToolExecutor(max_output_length=args.max_output_length)
    
    # Run all tests
    test_basic_execution()
    test_multiple_blocks()
    test_error_handling()
    test_module_imports()
    
    # Test with custom input
    print("Custom test:")
    print("Enter text with Python code blocks (type 'exit' to quit):")
    
    while True:
        try:
            text = input("> ")
            if text.lower() == "exit":
                break
            
            result = executor.process_text(text)
            print(result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main() 