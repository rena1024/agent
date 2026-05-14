from tools.base import ToolRegistry
from tools.calculator import Calculator
from tools.search import Search

try:
    from tools.retrieval import Retrieval
except Exception:  # noqa: BLE001
    Retrieval = None  # type: ignore

registry = ToolRegistry()
registry.register(Calculator())
registry.register(Search())
if Retrieval:
    registry.register(Retrieval())

__all__ = ["registry", "Calculator", "Search", "Retrieval"]
