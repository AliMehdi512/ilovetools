# Hotfix Required

The `config_converter.py` file is missing `Optional` in the imports.

## Fix:
Change line 8 from:
```python
from typing import Any, Dict, List, Union
```

To:
```python
from typing import Any, Dict, List, Union, Optional
```

This will be fixed in the next release.
