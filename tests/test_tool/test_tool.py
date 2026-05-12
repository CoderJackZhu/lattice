import asyncio

import pytest

from lattice.tool.tool import Tool, ToolContext, ToolOutput, tool
from lattice.tool.toolkit import ToolKit
from lattice.tool.builtins.shell import shell
from lattice.tool.builtins.file import list_dir, read_file, write_file


def test_tool_decorator_no_parens():
    @tool
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert isinstance(greet, Tool)
    assert greet.name == "greet"
    assert greet.description == "greet"


def test_tool_decorator_with_description():
    @tool(description="Say hello")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet.description == "Say hello"


def test_tool_decorator_generates_schema():
    @tool(description="Add numbers")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    schema = add.to_schema()
    assert schema.name == "add"
    assert "a" in schema.parameters["properties"]
    assert "b" in schema.parameters["properties"]
    assert "a" in schema.parameters.get("required", [])
    assert "b" not in schema.parameters.get("required", [])


def test_tool_decorator_default_values():
    @tool(description="test")
    async def func(x: str, y: int = 5) -> str:
        return f"{x}-{y}"

    schema = func.to_schema()
    props = schema.parameters["properties"]
    assert props["y"]["default"] == 5


async def test_tool_execute():
    @tool(description="concat")
    async def concat(a: str, b: str) -> str:
        return a + b

    params = concat.parameters(a="hello", b=" world")
    ctx = ToolContext()
    result = await concat.execute(params, ctx)
    assert result.content == "hello world"


async def test_tool_execute_sync_function():
    @tool(description="sync tool")
    def sync_add(a: int, b: int) -> str:
        return str(a + b)

    params = sync_add.parameters(a=3, b=4)
    ctx = ToolContext()
    result = await sync_add.execute(params, ctx)
    assert result.content == "7"


async def test_shell_tool():
    assert shell.name == "shell"
    params = shell.parameters(command="echo hello")
    ctx = ToolContext()
    result = await shell.execute(params, ctx)
    assert "hello" in result.content


async def test_list_dir_tool():
    assert list_dir.name == "list_dir"
    params = list_dir.parameters(path=".")
    ctx = ToolContext()
    result = await list_dir.execute(params, ctx)
    assert "pyproject.toml" in result.content


async def test_read_file_tool(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello content")

    params = read_file.parameters(path=str(test_file))
    ctx = ToolContext()
    result = await read_file.execute(params, ctx)
    assert result.content == "hello content"


async def test_write_file_tool(tmp_path):
    test_file = tmp_path / "out.txt"
    params = write_file.parameters(path=str(test_file), content="written!")
    ctx = ToolContext()
    result = await write_file.execute(params, ctx)
    assert "Successfully" in result.content
    assert test_file.read_text() == "written!"


def test_toolkit():
    @tool(description="t1")
    async def tool_a(x: str) -> str:
        return x

    @tool(description="t2")
    async def tool_b(y: int) -> str:
        return str(y)

    kit = ToolKit("test_kit", [tool_a, tool_b])
    assert len(kit.get_tools()) == 2
    schemas = kit.get_schemas()
    assert len(schemas) == 2
    assert schemas[0].name == "tool_a"


async def test_toolkit_execute():
    @tool(description="echo")
    async def echo(msg: str) -> str:
        return msg

    kit = ToolKit("test", [echo])
    params = echo.parameters(msg="hi")
    ctx = ToolContext()
    result = await kit.execute("echo", params, ctx)
    assert result.content == "hi"


async def test_toolkit_execute_not_found():
    kit = ToolKit("test", [])
    from pydantic import BaseModel

    class Empty(BaseModel):
        pass

    ctx = ToolContext()
    result = await kit.execute("nonexistent", Empty(), ctx)
    assert "not found" in result.content
