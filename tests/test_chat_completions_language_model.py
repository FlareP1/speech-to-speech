"""Tests for speech_to_speech.LLM.chat_completions_language_model.

Covers _build_openai_messages, ChatCompletionsApiModelHandler tool loop,
compaction with dict entries, and fallback after tool-loop exhaustion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import OpenAI
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as RawToolCall

from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _build_openai_messages,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_runtime_config(chat_size=2, instructions="You are helpful."):
    from openai.types.realtime import RealtimeSessionCreateRequest

    return RuntimeConfig(
        chat=Chat(chat_size),
        session=RealtimeSessionCreateRequest(type="realtime", instructions=instructions),
    )


def _make_request(text="Hi", chat_size=2):
    cfg = _make_runtime_config(chat_size=chat_size)
    cfg.chat.add_item(make_user_message(text))
    return GenerateResponseRequest(runtime_config=cfg)


def _make_text_delta(content: str, **extra):
    return ChoiceDelta(content=content, role="assistant", **extra)


def _make_tool_call_delta(index=0, id="call_1", name="", arguments=""):
    """Create a tool_calls list for ChoiceDelta."""
    return [
        RawToolCall(
            index=index,
            id=id,
            function={"name": name, "arguments": arguments},
            type="function",
        )
    ]


def _make_stream_chunk(delta=None, tool_calls=None, finish_reason=None, usage=None):
    delta_obj = delta or _make_text_delta("")
    if tool_calls is not None:
        delta_obj.tool_calls = tool_calls
    return SimpleNamespace(
        choices=[
            Choice(
                index=0,
                delta=delta_obj,
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _make_handler(*, compact_history=False, mcp_enabled=False, mcp_server_url=None, mcp_servers=None):
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler.model_name = "test-model"
    handler.stream_batch_sentences = 1
    handler.gen_kwargs = {}
    handler.request_timeout_s = 20.0
    handler.request_timeout = 20.0
    handler.disable_thinking = True
    handler._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.mcp_client = None
    handler.tools = []
    handler.compactor = None
    handler.stream = True
    handler.enable_lang_prompt = False
    handler.max_tokens = 4096
    handler.compact_history = compact_history
    handler.suppress_tool_call_flush = True
    return handler


# ===================================================================
# _build_openai_messages
# ===================================================================


class TestBuildOpenaiMessages:
    def test_empty_chat(self):
        chat = Chat(size=5)
        assert _build_openai_messages(chat) == []

    def test_system_message(self):
        chat = Chat(size=5)
        chat.init_chat_message = make_assistant_message("sys")  # simulate init_chat_message
        chat.init_chat_message.content[0].text = "You are helpful."
        result = _build_openai_messages(chat)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message(self):
        chat = Chat(size=5)
        chat.add_item(make_user_message("Hello"))
        result = _build_openai_messages(chat)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        chat = Chat(size=5)
        chat.add_item(make_assistant_message("Hi there"))
        result = _build_openai_messages(chat)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_dict_entries_passed_through(self):
        chat = Chat(size=5)
        chat.add_item(make_user_message("Hello"))
        chat.buffer.append({"role": "assistant", "content": "dict response"})
        chat.buffer.append({"role": "tool", "tool_call_id": "call_1", "content": "result"})
        result = _build_openai_messages(chat)
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "dict response"}
        assert result[2] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}


# ===================================================================
# Tool loop with compaction and dict entries
# ===================================================================


def test_tool_loop_executes_tool_and_continues_then_compacts():
    """Simulate a full tool loop: tool call → MCP result → text response, then compaction."""
    handler = _make_handler(compact_history=True)

    mcp_client = MagicMock()
    mcp_client.execute_tool.return_value = '{"time": "12:00"}'
    handler.mcp_client = mcp_client
    handler.tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]

    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [
                _make_stream_chunk(
                    _make_text_delta(""),
                    tool_calls=_make_tool_call_delta(index=0, id="call_1", name="get_time", arguments="{}"),
                    finish_reason="tool_calls",
                )
            ]
        else:
            return [
                _make_stream_chunk(_make_text_delta("The time is 12:00."), finish_reason="stop"),
            ]

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    cfg = _make_runtime_config(chat_size=2)
    cfg.chat.add_item(make_user_message("What time is it?"))

    request = GenerateResponseRequest(runtime_config=cfg)
    outputs = list(handler.process(request))

    text_chunks = [o for o in outputs if isinstance(o, LLMResponseChunk) and o.text]
    eos = [o for o in outputs if isinstance(o, EndOfResponse)]
    assert len(text_chunks) == 1
    assert text_chunks[0].text == "The time is 12:00."
    assert len(eos) == 1

    mcp_client.execute_tool.assert_called_once_with("get_time", {})

    dict_entries = [e for e in cfg.chat.buffer if isinstance(e, dict)]
    assert len(dict_entries) >= 2


def test_tool_loop_exhaustion_yields_fallback():
    """When all 5 iterations return tool_calls with no text, yield fallback."""
    handler = _make_handler(compact_history=False)

    mcp_client = MagicMock()
    mcp_client.execute_tool.return_value = "error"
    handler.mcp_client = mcp_client
    handler.tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]

    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        return [
            _make_stream_chunk(
                _make_text_delta(""),
                tool_calls=_make_tool_call_delta(index=0, id=f"call_{call_count}", name="noop", arguments="{}"),
                finish_reason="tool_calls",
            )
        ]

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    cfg = _make_runtime_config(chat_size=2)
    cfg.chat.add_item(make_user_message("Do something"))

    request = GenerateResponseRequest(runtime_config=cfg)
    outputs = list(handler.process(request))

    text_chunks = [o for o in outputs if isinstance(o, LLMResponseChunk) and o.text]
    assert len(text_chunks) == 1
    assert "I wasn't able to find the information" in text_chunks[0].text


def test_compaction_with_dict_entries_does_not_crash():
    """Compaction should handle dict entries in buffer without AttributeError."""
    handler = _make_handler(compact_history=True)

    captured = []

    def stub_compactor(snapshot):
        captured.append(snapshot)
        from speech_to_speech.LLM.chat import CompactionResult
        return CompactionResult(user_summary="U", assistant_summary="A")

    handler.compactor = stub_compactor

    cfg = _make_runtime_config(chat_size=2)
    cfg.chat.add_item(make_user_message("u0"))
    cfg.chat.add_item(make_assistant_message("a0"))

    cfg.chat.buffer.append({"role": "assistant", "content": "tool assistant", "tool_calls": []})
    cfg.chat.buffer.append({"role": "tool", "tool_call_id": "call_1", "content": "result"})

    cfg.chat.add_item(make_user_message("u1"))
    cfg.chat.add_item(make_assistant_message("a1"))

    cfg.chat.buffer.append({"role": "assistant", "content": "more text"})
    cfg.chat.buffer.append({"role": "tool", "tool_call_id": "call_2", "content": "output"})

    cfg.chat.add_item(make_user_message("u2"))
    cfg.chat.add_item(make_assistant_message("a2"))
    cfg.chat.add_item(make_user_message("u3"))

    cfg.chat.trim_if_needed(handler.compactor)

    assert len(captured) == 1
    assert len(cfg.chat.buffer) > 0
