from tools.base import ToolRegistry
from tools.calculator import Calculator
from tools.search import Search
from tools.retrieval import Retrieval

registry = ToolRegistry()
registry.register(Calculator())
registry.register(Search())
registry.register(Retrieval())

__all__ = ["registry", "Calculator", "Search", "Retrieval"]
