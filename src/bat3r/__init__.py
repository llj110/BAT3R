"""BAT3R implementation. Upstream utilities remain under dust3r and croco."""

__all__ = ["BAT3R", "BAT3RConfig"]

def __getattr__(name):
    if name in __all__:
        from .bat3r_model import BAT3R, BAT3RConfig
        return {"BAT3R": BAT3R, "BAT3RConfig": BAT3RConfig}[name]
    raise AttributeError(name)
