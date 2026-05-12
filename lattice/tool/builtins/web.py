from __future__ import annotations

from lattice.tool.tool import ToolOutput, tool


@tool
async def http_get(url: str, headers: str = "") -> ToolOutput:
    """Send an HTTP GET request and return the response body."""
    import httpx

    header_dict = {}
    if headers:
        for line in headers.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                header_dict[k.strip()] = v.strip()

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=header_dict)
        return ToolOutput(
            content=resp.text[:10000],
            metadata={"status_code": resp.status_code, "url": str(resp.url)},
        )


@tool
async def http_post(url: str, body: str = "", content_type: str = "application/json") -> ToolOutput:
    """Send an HTTP POST request and return the response body."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            content=body.encode(),
            headers={"Content-Type": content_type},
        )
        return ToolOutput(
            content=resp.text[:10000],
            metadata={"status_code": resp.status_code, "url": str(resp.url)},
        )
