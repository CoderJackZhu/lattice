from lattice.llm.types import (
    ImageContent,
    Message,
    ModelResponse,
    StreamEnd,
    StreamStart,
    TextContent,
    TextDelta,
    ThinkingContent,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolResult,
    ToolSchema,
    Usage,
)


def test_text_content():
    c = TextContent(text="hello")
    assert c.type == "text"
    assert c.text == "hello"


def test_image_content():
    c = ImageContent(url="http://example.com/img.png")
    assert c.type == "image"
    assert c.media_type == "image/png"


def test_thinking_content():
    c = ThinkingContent(text="let me think")
    assert c.type == "thinking"


def test_tool_call():
    tc = ToolCall(id="tc_1", name="shell", arguments={"command": "ls"})
    assert tc.type == "tool_call"
    assert tc.arguments == {"command": "ls"}


def test_tool_result():
    tr = ToolResult(tool_call_id="tc_1", content="output", is_error=False)
    assert tr.type == "tool_result"
    assert not tr.is_error


def test_message_with_mixed_content():
    msg = Message(role="user", content=[
        TextContent(text="Look at this:"),
        ImageContent(url="http://example.com/img.png"),
    ])
    assert msg.role == "user"
    assert len(msg.content) == 2
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "image"


def test_usage_defaults():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_read_tokens == 0
    assert u.cache_write_tokens == 0


def test_usage_with_values():
    u = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=20)
    assert u.input_tokens == 100
    assert u.cache_read_tokens == 20


def test_tool_schema():
    ts = ToolSchema(
        name="shell",
        description="Run a command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    assert ts.name == "shell"


def test_model_response():
    msg = Message(role="assistant", content=[TextContent(text="hi")])
    r = ModelResponse(message=msg, usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn", model="gpt-4o")
    assert r.stop_reason == "end_turn"
    assert r.usage.input_tokens == 10


def test_stream_events():
    start = StreamStart(model="gpt-4o")
    assert start.type == "stream_start"

    delta = TextDelta(text="hello")
    assert delta.type == "text_delta"

    end = StreamEnd()
    assert end.type == "stream_end"

    tc_start = ToolCallStart(tool_call_id="1", name="shell")
    assert tc_start.type == "tool_call_start"

    tc_end = ToolCallEnd(tool_call_id="1", name="shell", arguments={"cmd": "ls"})
    assert tc_end.type == "tool_call_end"
