"""
This type stub file was generated by pyright.
"""

from collections.abc import Callable
from typing import Any, TypeVar, overload

_T = TypeVar('_T')
def kwarg_doc(text: str) -> Callable[[_T], _T]:
    ...

class Substitution:
    @overload
    def __init__(self, *args: str) -> None:
        ...
    
    @overload
    def __init__(self, **kwargs: str) -> None:
        ...
    
    def __call__(self, func: _T) -> _T:
        ...
    
    def update(self, *args, **kwargs):
        ...
    


class _ArtistKwdocLoader(dict[str, str]):
    def __missing__(self, key: str) -> str:
        ...
    


class _ArtistPropertiesSubstitution:
    def __init__(self) -> None:
        ...
    
    def register(self, **kwargs) -> None:
        ...
    
    def __call__(self, obj: _T) -> _T:
        ...
    


def copy(source: Any) -> Callable[[_T], _T]:
    ...

dedent_interpd: _ArtistPropertiesSubstitution
interpd: _ArtistPropertiesSubstitution
