"""
Main adapter module for connecting the ALPR system to CodeProject.AI.
This file will be gradually replaced by the new modular structure.
"""

# Import from our new package structure
from alpr.adapter import ALPRAdapter

# Start the adapter
if __name__ == "__main__":
    ALPRAdapter().start_loop()
