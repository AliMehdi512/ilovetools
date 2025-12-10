"""
ilovetools - A comprehensive Python utility library
"""

__version__ = "0.2.9.post3"
# release marker: post3 (tiny, harmless change to alter build artifact hash for PyPI upload)
__author__ = "Ali Mehdi"
__email__ = "ali.mehdi.dev579@gmail.com"

# Import all modules for easy access
from . import ai
from . import data
from . import ml
from . import files
from . import text
from . import image
from . import audio
from . import web
from . import security
from . import database
from . import datetime
from . import validation
from . import conversion
from . import automation
from . import utils

__all__ = [
    "ai",
    "data",
    "ml",
    "files",
    "text",
    "image",
    "audio",
    "web",
    "security",
    "database",
    "datetime",
    "validation",
    "conversion",
    "automation",
    "utils",
]
