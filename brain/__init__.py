"""brain package — Python wrapper around the Rust brain_core extension.

Re-exports all functions from the compiled Rust module (brain_core)
so callers can simply `import brain` and get everything.
"""

from brain_core import *  # noqa: F401,F403
