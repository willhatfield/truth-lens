"""Reusable error-wrapping and safe-response builders."""

from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def build_error_response(response_class: Type[T], error_msg: str) -> T:
    """Construct a default instance of *response_class* with its error field set.

    The response class must accept ``error`` as a keyword argument.
    All other fields will use their declared defaults.
    """
    return response_class(error=error_msg)
