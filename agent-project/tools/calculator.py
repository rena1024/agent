"""Calculator tool for arithmetic with multi-number support."""

import ast
import operator as op
from typing import Any, Dict

from tools.base import Tool


# Allowed operators
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate arithmetic expressions (+ - * / ^) with multiple terms."

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        expr = str(tool_input.get("expression", "")).strip()
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval(tree.body)
            return {"status": "ok", "output": str(value)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "output": f"{exc}"}
