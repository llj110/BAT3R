"""Compatibility import; implementation lives in bat3r.bat3r_update."""
from bat3r.bat3r_update import *  # noqa: F401,F403
from bat3r import bat3r_update as _implementation

def __getattr__(name):
    return getattr(_implementation, name)
