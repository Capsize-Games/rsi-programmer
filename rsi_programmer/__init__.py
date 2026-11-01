"""Program synthesis policies, interpreters and source emitters."""


def __getattr__(name: str) -> object:
    """Keep source emission usable without loading neural dependencies."""
    if name == "SearchPolicy":
        from rsi_programmer.policy import SearchPolicy
        return SearchPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SearchPolicy"]
