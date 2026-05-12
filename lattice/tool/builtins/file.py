from pathlib import Path

from lattice.tool.tool import tool


@tool(description="Read the contents of a file")
async def read_file(path: str, encoding: str = "utf-8") -> str:
    try:
        return Path(path).read_text(encoding=encoding)
    except Exception as e:
        return f"Error reading file: {e}"


@tool(description="Write content to a file")
async def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(description="List files and directories at the given path")
async def list_dir(path: str = ".") -> str:
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = []
        for entry in entries:
            prefix = "d " if entry.is_dir() else "f "
            lines.append(prefix + entry.name)
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"
