"""Provide safe fallbacks for optional LangChain dependencies."""

from __future__ import annotations

from functools import update_wrapper
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class _ShimTool:  # pragma: no cover - thin wrapper
    def __init__(self, func: F, name: Optional[str] = None):
        self._func = func
        self.name = name or func.__name__
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def invoke(self, input_data: Any | None = None, /, **kwargs: Any) -> Any:
        if kwargs:
            return self._func(**kwargs)
        if input_data is None:
            return self._func()
        if isinstance(input_data, dict):
            return self._func(**input_data)
        return self._func(input_data)


try:
    from langchain_core.tools import tool as _tool  # type: ignore
except ImportError:  # pragma: no cover

    def _tool(*decorator_args: Any, **decorator_kwargs: Any):
        """Minimal stand-in for ``langchain_core.tools.tool`` when unavailable."""

        name = decorator_kwargs.get("name")
        if decorator_args and isinstance(decorator_args[0], str):
            name = decorator_args[0]

        # Handle ``@tool`` without params
        if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
            return _ShimTool(decorator_args[0], name=name)

        def decorator(func: F) -> _ShimTool:
            return _ShimTool(func, name=name)

        return decorator


tool = _tool
