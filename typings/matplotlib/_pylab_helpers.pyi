"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.figure import Figure

class Gcf:
    figs: OrderedDict[int, FigureManagerBase]
    @classmethod
    def get_fig_manager(cls, num: int) -> FigureManagerBase | None:
        ...
    
    @classmethod
    def destroy(cls, num: int | FigureManagerBase) -> None:
        ...
    
    @classmethod
    def destroy_fig(cls, fig: Figure) -> None:
        ...
    
    @classmethod
    def destroy_all(cls) -> None:
        ...
    
    @classmethod
    def has_fignum(cls, num: int) -> bool:
        ...
    
    @classmethod
    def get_all_fig_managers(cls) -> list[FigureManagerBase]:
        ...
    
    @classmethod
    def get_num_fig_managers(cls) -> int:
        ...
    
    @classmethod
    def get_active(cls) -> FigureManagerBase | None:
        ...
    
    @classmethod
    def set_active(cls, manager: FigureManagerBase) -> None:
        ...
    
    @classmethod
    def draw_all(cls, force: bool = ...) -> None:
        ...
    


