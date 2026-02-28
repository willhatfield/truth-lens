"""Reusable warning-wrapping and safe-response builders."""

from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def build_warning_response(
    response_class: Type[T],
    analysis_id: str,
    warning_msg: str,
) -> T:
    """Construct a default instance of response_class with a warning.

    The response class must accept analysis_id and warnings as keyword
    arguments.  All other fields use their declared defaults.
    """
    return response_class(
        analysis_id=analysis_id,
        warnings=[warning_msg],
    )
