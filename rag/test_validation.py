#!/usr/bin/env python3
"""
Test script for content validation functionality.
This script can be run independently to test the validation logic.
"""

import re
import os

def count_alphanumeric_chars(text):
    """
    Count the number of alphanumeric characters in a text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        int: Number of alphanumeric characters
    """
    if not text:
        return 0
    # Remove all non-alphanumeric characters and count
    alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', text)
    return len(alphanumeric_only)

def is_content_valid(content, min_alphanumeric_chars=100):
    """
    Validate if content has enough alphanumeric characters.
    
    Args:
        content (str): The content to validate
        min_alphanumeric_chars (int): Minimum number of alphanumeric characters required
        
    Returns:
        bool: True if content is valid, False otherwise
    """
    if not content or not isinstance(content, str):
        return False
    
    alphanumeric_count = count_alphanumeric_chars(content)
    return alphanumeric_count >= min_alphanumeric_chars

def test_content_validation():
    """
    Test function to demonstrate content validation.
    """
    test_cases = [
        "This is a short text with only 50 alphanumeric characters.",
        "This is a longer text with more than 100 alphanumeric characters including numbers 12345 and more text to reach the minimum requirement for validation.",
        "Short text",
        "This text has exactly 100 alphanumeric characters including numbers 1234567890 and letters to test the boundary condition properly.",
        "",
        "   ",  # Only spaces
        "!@#$%^&*()",  # Only symbols
        "This is a valid text with 123 numbers and more than 100 alphanumeric characters total including spaces and punctuation marks."
    ]
    
    print("Testing content validation:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        alphanumeric_count = count_alphanumeric_chars(text)
        is_valid = is_content_valid(text)
        print(f"Test {i}:")
        print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Alphanumeric chars: {alphanumeric_count}")
        print(f"  Valid: {is_valid}")
        print()

if __name__ == "__main__":
    test_content_validation() 