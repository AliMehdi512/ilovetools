#!/usr/bin/env python3
"""
Script to fix the missing Optional import in config_converter.py
Run this after installing ilovetools to fix the import error.
"""

import os
import sys

def fix_optional_import():
    """Fix the missing Optional import in config_converter.py"""
    
    # Find the ilovetools installation
    try:
        import ilovetools
        base_path = os.path.dirname(ilovetools.__file__)
        file_path = os.path.join(base_path, 'conversion', 'config_converter.py')
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the import line
        old_import = "from typing import Any, Dict, List, Union"
        new_import = "from typing import Any, Dict, List, Union, Optional"
        
        if old_import in content and new_import not in content:
            content = content.replace(old_import, new_import)
            
            # Write back
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✓ Fixed Optional import in {file_path}")
            return True
        elif new_import in content:
            print("✓ Optional import already fixed")
            return True
        else:
            print("Error: Could not find import line to fix")
            return False
            
    except ImportError:
        print("Error: ilovetools not installed")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_optional_import()
    sys.exit(0 if success else 1)
