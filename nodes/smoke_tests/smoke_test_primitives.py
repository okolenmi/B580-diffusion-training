"""Contract + behavior check for nodes/primitive/ -- pure Python, no torch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.primitive.values import (BoolConstantNode, FloatConstantNode,
                                     IntConstantNode, StringConstantNode)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    check(FloatConstantNode().build()["value"] == 0.0, "float default should be 0.0")
    check(FloatConstantNode().build(value=3.5)["value"] == 3.5, "float should pass through")
    check(IntConstantNode().build(value=7)["value"] == 7, "int should pass through")
    check(StringConstantNode().build(value="hi")["value"] == "hi", "string should pass through")
    check(BoolConstantNode().build()["value"] is False, "bool default should be False")
    check(BoolConstantNode().build(value=True)["value"] is True, "bool should pass through")
    print("All primitive node checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
