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


def cuda_is_available():
    if package_exists("torch"):
        return __import__("torch").cuda.is_available()
    elif package_exists("nvidia_smi"):
        try:
            __import__("nvidia_smi").nvmlInit()
            return True
        except:
            return False
    else:
        return False
