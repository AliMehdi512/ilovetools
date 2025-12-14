#!/usr/bin/env python3
"""One-line fix for Optional import issue"""
import os, sys
try:
    import ilovetools
    f = os.path.join(ilovetools.__path__[0], 'conversion', 'config_converter.py')
    c = open(f).read()
    if 'from typing import Any, Dict, List, Union, Optional' not in c:
        open(f, 'w').write(c.replace('from typing import Any, Dict, List, Union', 'from typing import Any, Dict, List, Union, Optional'))
        print(f"✓ Fixed: {f}")
    else:
        print("✓ Already fixed")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
