import importlib.util


def package_exists(package_name: str) -> bool:
    """
    Check if a package exists in the current environment.
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


def is_debug_mode_enabled() -> bool:
    """
    Check if debug mode is enabled.
    """
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return bool(gettrace())
