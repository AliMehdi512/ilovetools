# Quick Fix for Import Error

If you encounter `NameError: name 'Optional' is not defined`, run this command:

```bash
python -c "import os; f=os.path.join(__import__('ilovetools').__path__[0],'conversion','config_converter.py'); c=open(f).read(); open(f,'w').write(c.replace('from typing import Any, Dict, List, Union','from typing import Any, Dict, List, Union, Optional'))"
```

Or manually edit the file:
1. Find your ilovetools installation: `python -c "import ilovetools; print(ilovetools.__path__)"`
2. Open `conversion/config_converter.py`
3. Change line 8 from `from typing import Any, Dict, List, Union` to `from typing import Any, Dict, List, Union, Optional`

This will be fixed in the next PyPI release.
