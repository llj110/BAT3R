"""Compatibility entry point. Prefer bat3r_demo.py."""
from bat3r_demo import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("bat3r_demo", run_name="__main__")
