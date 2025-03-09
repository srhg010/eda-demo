"""
This type stub file was generated by pyright.
"""

import numpy as np
from collections.abc import Callable
from typing import Literal
from numpy.typing import ArrayLike
from .path import Path

class NonIntersectingPathException(ValueError):
    ...


def get_intersection(cx1: float, cy1: float, cos_t1: float, sin_t1: float, cx2: float, cy2: float, cos_t2: float, sin_t2: float) -> tuple[float, float]:
    ...

def get_normal_points(cx: float, cy: float, cos_t: float, sin_t: float, length: float) -> tuple[float, float, float, float]:
    ...

def split_de_casteljau(beta: ArrayLike, t: float) -> tuple[np.ndarray, np.ndarray]:
    ...

def find_bezier_t_intersecting_with_closedpath(bezier_point_at_t: Callable[[float], tuple[float, float]], inside_closedpath: Callable[[tuple[float, float]], bool], t0: float = ..., t1: float = ..., tolerance: float = ...) -> tuple[float, float]:
    ...

class BezierSegment:
    def __init__(self, control_points: ArrayLike) -> None:
        ...
    
    def __call__(self, t: ArrayLike) -> np.ndarray:
        ...
    
    def point_at_t(self, t: float) -> tuple[float, ...]:
        ...
    
    @property
    def control_points(self) -> np.ndarray:
        ...
    
    @property
    def dimension(self) -> int:
        ...
    
    @property
    def degree(self) -> int:
        ...
    
    @property
    def polynomial_coefficients(self) -> np.ndarray:
        ...
    
    def axis_aligned_extrema(self) -> tuple[np.ndarray, np.ndarray]:
        ...
    


def split_bezier_intersecting_with_closedpath(bezier: ArrayLike, inside_closedpath: Callable[[tuple[float, float]], bool], tolerance: float = ...) -> tuple[np.ndarray, np.ndarray]:
    ...

def split_path_inout(path: Path, inside: Callable[[tuple[float, float]], bool], tolerance: float = ..., reorder_inout: bool = ...) -> tuple[Path, Path]:
    ...

def inside_circle(cx: float, cy: float, r: float) -> Callable[[tuple[float, float]], bool]:
    ...

def get_cos_sin(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    ...

def check_if_parallel(dx1: float, dy1: float, dx2: float, dy2: float, tolerance: float = ...) -> Literal[-1, False, 1]:
    ...

def get_parallels(bezier2: ArrayLike, width: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    ...

def find_control_points(c1x: float, c1y: float, mmx: float, mmy: float, c2x: float, c2y: float) -> list[tuple[float, float]]:
    ...

def make_wedged_bezier2(bezier2: ArrayLike, width: float, w1: float = ..., wm: float = ..., w2: float = ...) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    ...

