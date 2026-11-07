"""Small shared helpers."""
import os
import tempfile
from pathlib import Path


def atomic_write(path, text, encoding="utf-8"):
    """Write text to path atomically (temp file in the same dir + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


ID_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"


def safe_child(base, name, suffix=""):
    """base/name+suffix if `name` is an allowlisted identifier and the resolved path stays inside base; else ValueError."""
    import re
    if not re.fullmatch(ID_PATTERN, str(name)) or ".." in str(name):
        raise ValueError(f"invalid identifier {name!r}")
    base = Path(base).resolve()
    p = (base / f"{name}{suffix}").resolve()
    if p.parent != base:
        raise ValueError(f"invalid identifier {name!r}")
    return p
