"""Compatibility import; implementation lives in bat3r.bat3r_inference."""
from bat3r.bat3r_inference import *  # noqa: F401,F403
from bat3r import bat3r_inference as _implementation

def __getattr__(name):
    return getattr(_implementation, name)
