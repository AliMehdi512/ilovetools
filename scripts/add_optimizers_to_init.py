# Script to add optimizers to __init__.py

optimizers_import = """
from .optimizers import (
    # Adam Variants
    adam_optimizer,
    adamw_optimizer,
    adamax_optimizer,
    nadam_optimizer,
    amsgrad_optimizer,
    # RMSprop Variants
    rmsprop_optimizer,
    rmsprop_momentum_optimizer,
    # Modern Optimizers
    radam_optimizer,
    lamb_optimizer,
    lookahead_optimizer,
    adabelief_optimizer,
    # Utilities
    create_optimizer_state,
    get_optimizer_function,
    # Aliases
    adam,
    adamw,
    adamax,
    nadam,
    amsgrad,
    rmsprop,
    rmsprop_mom,
    radam,
    lamb,
    lookahead,
    adabelief,
)
"""

optimizers_exports = """    # Advanced Optimizers - Adam Variants
    'adam_optimizer',
    'adamw_optimizer',
    'adamax_optimizer',
    'nadam_optimizer',
    'amsgrad_optimizer',
    # Advanced Optimizers - RMSprop Variants
    'rmsprop_optimizer',
    'rmsprop_momentum_optimizer',
    # Advanced Optimizers - Modern
    'radam_optimizer',
    'lamb_optimizer',
    'lookahead_optimizer',
    'adabelief_optimizer',
    # Advanced Optimizers - Utilities
    'create_optimizer_state',
    'get_optimizer_function',
    # Advanced Optimizers - Aliases
    'adam',
    'adamw',
    'adamax',
    'nadam',
    'amsgrad',
    'rmsprop',
    'rmsprop_mom',
    'radam',
    'lamb',
    'lookahead',
    'adabelief',
"""

print("Add this after the activations import:")
print(optimizers_import)
print("\nAdd this at the end of __all__ list (before the closing bracket):")
print(optimizers_exports)
