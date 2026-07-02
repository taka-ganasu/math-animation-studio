from __future__ import annotations

import py_compile
from pathlib import Path


class ValidationError(RuntimeError):
    pass


def validate_python_syntax(path: Path) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        raise ValidationError(f"Generated Python failed syntax validation: {exc}") from exc
