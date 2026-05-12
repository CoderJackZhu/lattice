import asyncio

from lattice.tool.tool import tool


@tool(description="Execute a shell command and return stdout/stderr")
async def shell(command: str, timeout: float = 30.0) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "Error: command timed out"
    output = stdout.decode(errors="replace")
    if stderr:
        output += "\n" + stderr.decode(errors="replace")
    return output
