"""Tests for PromptFormatter."""

from __future__ import annotations

from octomil.responses.prompt_formatter import PromptFormatter
from octomil.responses.types import (
    AssistantInput,
    ImageContent,
    ResponseToolCall,
    SpecificToolChoice,
    TextContent,
    ToolChoice,
    ToolResultInput,
    UserInput,
    system_input,
    text_input,
)


def test_formats_simple_text():
    result = PromptFormatter.format(input=[text_input("Hello")])
    assert "<|user|>\nHello\n" in result
    assert result.endswith("<|assistant|>\n")


def test_formats_system_message():
    result = PromptFormatter.format(
        input=[
            system_input("You are helpful"),
            text_input("Hi"),
        ]
    )
    assert "<|system|>\nYou are helpful\n" in result
    assert "<|user|>\nHi\n" in result


def test_formats_tool_result():
    result = PromptFormatter.format(
        input=[
            ToolResultInput(tool_call_id="call_1", content="72\u00b0F"),
        ]
    )
    assert "<|tool|>\n72\u00b0F\n" in result


def test_formats_assistant_with_tool_calls():
    result = PromptFormatter.format(
        input=[
            AssistantInput(
                tool_calls=[
                    ResponseToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments='{"city":"NYC"}',
                    )
                ]
            ),
        ]
    )
    assert "<|assistant|>\n" in result
    assert '"type": "tool_call"' in result
    assert '"name": "get_weather"' in result


def test_includes_tool_definitions():
    result = PromptFormatter.format(
        input=[text_input("What's the weather?")],
        tools=[
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                }
            }
        ],
    )
    assert "Function: get_weather" in result
    assert "Description: Get weather for a city" in result


def test_tool_instruction_uses_new_format():
    result = PromptFormatter.format(
        input=[text_input("Hello")],
        tools=[{"name": "fn", "description": "desc"}],
    )
    assert '"type": "tool_call"' in result
    assert "respond with ONLY this JSON" in result
    assert "normal text" in result


def test_skips_tools_when_none():
    result = PromptFormatter.format(
        input=[text_input("Hello")],
        tools=[
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                }
            }
        ],
        tool_choice=ToolChoice.NONE,
    )
    assert "Function: get_weather" not in result


def test_adds_required_instruction():
    result = PromptFormatter.format(
        input=[text_input("Hello")],
        tools=[
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                }
            }
        ],
        tool_choice=ToolChoice.REQUIRED,
    )
    assert "MUST use one of the available tools" in result


def test_adds_specific_tool_instruction():
    result = PromptFormatter.format(
        input=[text_input("Hello")],
        tools=[
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                }
            }
        ],
        tool_choice=SpecificToolChoice(name="get_weather"),
    )
    assert "MUST use the tool: get_weather" in result


def test_formats_image_placeholder():
    result = PromptFormatter.format(
        input=[
            UserInput(
                content=[
                    TextContent(text="What is this?"),
                    ImageContent(data="base64data", media_type="image/png"),
                ]
            ),
        ]
    )
    assert "What is this?" in result
    assert "[image]" in result


def test_formats_multi_turn():
    result = PromptFormatter.format(
        input=[
            system_input("You are a helpful assistant"),
            text_input("Hello"),
            AssistantInput(content=[TextContent(text="Hi! How can I help?")]),
            text_input("What is 2+2?"),
        ]
    )
    assert "<|system|>\nYou are a helpful assistant\n" in result
    assert "<|user|>\nHello\n" in result
    assert "<|assistant|>\nHi! How can I help?\n" in result
    assert "<|user|>\nWhat is 2+2?\n" in result
    assert result.endswith("<|assistant|>\n")
