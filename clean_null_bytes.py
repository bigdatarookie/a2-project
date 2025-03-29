#!/usr/bin/env python
"""
clean_null_bytes.py - Remove null bytes from Python source files
"""
import os
import sys

def clean_nulls_from_file(filepath):
    """Remove null bytes from a file."""
    try:
        # Read file content
        with open(filepath, 'rb') as f:
            content = f.read()

        # Check if file contains null bytes
        if b'\x00' in content:
            print(f"Fixing null bytes in {filepath}")
            # Remove null bytes
            cleaned_content = content.replace(b'\x00', b'')

            # Write cleaned content back
            with open(filepath, 'wb') as f:
                f.write(cleaned_content)

            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def clean_directory(directory):
    """Clean all Python files in directory recursively."""
    fixed_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if clean_nulls_from_file(filepath):
                    fixed_files += 1

    return fixed_files

if __name__ == '__main__':
    # Default to current directory if none specified
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'

    print(f"Scanning {directory} for Python files with null bytes...")
    fixed = clean_directory(directory)

    if fixed:
        print(f"Fixed {fixed} files with null bytes.")
    else:
        print("No files with null bytes found.")
