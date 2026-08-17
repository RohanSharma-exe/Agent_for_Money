from __future__ import annotations

import unittest
from typing import Any

from local_opportunity_agent.tools import (
    Tool,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
)


class EchoTool(Tool):
    name = "echo"
    description = "Returns the supplied arguments."

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=arguments,
        )


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_register_and_get_tool(self) -> None:
        tool = EchoTool()

        self.registry.register(tool)

        self.assertIs(
            self.registry.get("echo"),
            tool,
        )

    def test_execute_tool(self) -> None:
        self.registry.register(EchoTool())

        result = self.registry.execute(
            "echo",
            {"message": "hello"},
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.tool_name,
            "echo",
        )
        self.assertEqual(
            result.data["message"],
            "hello",
        )

    def test_names_are_sorted(self) -> None:
        class ZTool(EchoTool):
            name = "z_tool"

        class ATool(EchoTool):
            name = "a_tool"

        self.registry.register(ZTool())
        self.registry.register(ATool())

        self.assertEqual(
            self.registry.names(),
            ["a_tool", "z_tool"],
        )

    def test_descriptions(self) -> None:
        self.registry.register(EchoTool())

        self.assertEqual(
            self.registry.descriptions(),
            {
                "echo": "Returns the supplied arguments.",
            },
        )

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(ToolRegistryError):
            self.registry.get("missing")

    def test_duplicate_tool_is_rejected(self) -> None:
        self.registry.register(EchoTool())

        with self.assertRaises(ToolRegistryError):
            self.registry.register(EchoTool())

    def test_empty_tool_name_is_rejected(self) -> None:
        class EmptyTool(EchoTool):
            name = ""

        with self.assertRaises(ToolRegistryError):
            self.registry.register(EmptyTool())


if __name__ == "__main__":
    unittest.main()
