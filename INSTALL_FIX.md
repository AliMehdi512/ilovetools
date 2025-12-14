# Installation Fix for v0.2.6

## Issue
Version 0.2.6 has a missing `Optional` import in `config_converter.py` that causes an import error.

## Quick Fix (Choose One)

### Option 1: One-Line Python Fix
```bash
python fix_import.py
```

### Option 2: Manual Command Line Fix
```bash
python -c "import os; f=os.path.join(__import__('ilovetools').__path__[0],'conversion','config_converter.py'); c=open(f).read(); open(f,'w').write(c.replace('from typing import Any, Dict, List, Union','from typing import Any, Dict, List, Union, Optional')) if 'Optional' not in c else None; print('Fixed!')"
```

### Option 3: Manual File Edit
1. Find ilovetools installation:
   ```bash
   python -c "import ilovetools; print(ilovetools.__file__)"
   ```

2. Open `ilovetools/conversion/config_converter.py`

3. Change line 8 from:
   ```python
   from typing import Any, Dict, List, Union
   ```
   
   To:
   ```python
   from typing import Any, Dict, List, Union, Optional
   ```

## Verify Fix
```python
from ilovetools.ml import sigmoid, relu, softmax
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
print(sigmoid(x))  # Should work!
```

## Status
This will be fixed in v0.2.7. For now, use one of the above methods after installing.
