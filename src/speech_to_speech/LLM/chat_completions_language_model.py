"""LLM backend using OpenAI Chat Completions API (llama.cpp compatible).

Uses ``client.chat.completions.create()`` instead of the Responses API.
Supports tool calling with MCP tool execution in a multi-turn loop.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.mcp_client import MCPClient
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_openai_messages(chat: Chat) -> list[dict]:
    """Convert Chat items to OpenAI Chat Completions message format.

    Handles:
    - ``init_chat_message`` (system) -> ``{"role": "system", "content": ...}``
    - ``RealtimeConversationItemUserMessage`` -> ``{"role": "user", "content": ...}``
    - ``RealtimeConversationItemAssistantMessage`` -> ``{"role": "assistant", "content": ...}``
    - ``RealtimeConversationItemFunctionCall`` -> merged into the last assistant
      message as ``tool_calls``
    - ``RealtimeConversationItemFunctionCallOutput`` -> ``{"role": "tool", ...}``
    - Plain dicts (tool result messages appended during the tool loop) -> passed through
    """
    messages: list[dict] = []

    # System message
    if chat.init_chat_message is not None:
        text = " ".join(p.text for p in chat.init_chat_message.content if p.text)
        if text:
            messages.append({"role": "system", "content": text})

    for item in chat.buffer:
        # Plain dict (tool result message from tool loop) — pass through
        if isinstance(item, dict):
            messages.append(item)
            continue

        if isinstance(item, RealtimeConversationItemUserMessage):
            text = " ".join(p.text for p in item.content if p.type == "input_text" and p.text)
            if text:
                messages.append({"role": "user", "content": text})

        elif isinstance(item, RealtimeConversationItemAssistantMessage):
            text = " ".join(p.text for p in item.content if p.text)
            messages.append({"role": "assistant", "content": text})

        elif isinstance(item, RealtimeConversationItemFunctionCall):
            # Merge into the last assistant message as a tool_call
            if messages and messages[-1].get("role") == "assistant":
                tc = {
                    "id": item.call_id or _generate_id("call"),
                    "type": "function",
                    "function": {
                        "name": item.name or "",
                        "arguments": item.arguments or "{}",
                    },
                }
                messages[-1].setdefault("tool_calls", []).append(tc)
            else:
                logger.warning("Function call without preceding assistant message; skipping")

        elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id or "",
                    "content": item.output or "",
                }
            )

    return messages


def _make_tool_message(call_id: str, content: str) -> dict:
    """Create a tool role message for the Chat Completions API."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ChatCompletionsApiModelHandler(BaseHandler[LLMIn, LLMOut]):
    """LLM backend using OpenAI Chat Completions API (llama.cpp compatible).

    Supports streaming, sentence-batched TTS forwarding, and MCP tool
    execution in a multi-turn loop (up to 5 iterations).
    """

    def setup(
        self,
        api_base_url: Optional[str] = None,
        api_api_key: Optional[str] = None,
        api_model: Optional[str] = None,
        mcp_server_url: Optional[str] = None,
        mcp_servers: Optional[str] = None,
        mcp_enabled: bool = False,
        cancel_scope: Optional[CancelScope] = None,
        speculative_turns: Optional[SpeculativeTurnTracker] = None,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        request_timeout_s: float = 120.0,
        max_tokens: int = 4096,
        disable_thinking: bool = True,
        compact_history: bool = False,
        gen_kwargs: dict[str, Any] = {},
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.max_tokens = max(1, max_tokens)
        self.gen_kwargs = dict(gen_kwargs)
        self.mcp_client: Optional[MCPClient] = None
        self.tools: list[dict] = []

        # Timeout
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        # OpenAI client
        api_key = api_api_key or "sk-placeholder"
        base_url = api_base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Resolve model name: try auto-detect via GET /v1/models, fallback to param
        self.model_name = self._resolve_model(api_model)

        # Thinking suppression — mirrors ResponsesApiModelHandler
        self._extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if disable_thinking
            and base_url is not None
            and base_url != "https://api.openai.com/v1"
            else None
        )

        # MCP tools
        if mcp_enabled and mcp_server_url:
            server_list = [s.strip() for s in (mcp_servers or "").split(",") if s.strip()]
            if server_list:
                self.mcp_client = MCPClient(
                    base_url=mcp_server_url,
                    servers=server_list,
                )
                self.tools = self.mcp_client.connect()
                logger.info(
                    "MCP: discovered %d tool(s): %s",
                    len(self.tools),
                    [t.get("function", {}).get("name", "?") for t in self.tools],
                )
            else:
                logger.warning("mcp_enabled=True but no servers specified; tools disabled")
        elif mcp_enabled:
            logger.warning("mcp_enabled=True but mcp_server_url not set; tools disabled")

        # Compaction
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None

        self.warmup()

    def _resolve_model(self, fallback_model: Optional[str]) -> str:
        """Try to auto-detect model via GET /v1/models; fall back to param."""
        try:
            resp = self.client.models.list(timeout=self.request_timeout)
            data = resp.data
            if data:
                detected = data[0].id
                logger.info("Auto-detected model: %s", detected)
                return detected
        except Exception:
            logger.debug("Model auto-detection failed; will use fallback")

        if fallback_model:
            return fallback_model
        return "gpt-4o-mini"

    # ---- cancellation helpers (mirrors ResponsesApiModelHandler) ------------

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a generate fn that calls Chat Completions for compaction."""
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout

        def generate(system: str, user: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""

        return generate

    # ---- lifecycle ---------------------------------------------------------

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = time.time()
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hi"},
                ],
                max_tokens=1,
                timeout=self.request_timeout,
                extra_body=self._extra_body,
            )
        except Exception:
            logger.debug("Warmup call failed (non-fatal)")
        logger.info("%s: warmed up in %.3f s", self.__class__.__name__, time.time() - start)

    def on_session_end(self) -> None:
        logger.debug("%s: session state reset", self.__class__.__name__)
        if self.mcp_client:
            self.mcp_client.disconnect()
            self.mcp_client = None
            self.tools = []

    def cleanup(self) -> None:
        if self.mcp_client:
            self.mcp_client.disconnect()

    # ---- core processing ---------------------------------------------------

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a :class:`GenerateResponseRequest` and yield LLM output."""
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        active_chat = original_chat.copy()

        # Instructions / system prompt
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        if instructions:
            full_instructions = build_voice_system_prompt(instructions)
            active_chat.add_item(make_system_message(full_instructions))

        # Language
        language_code = request.language_code
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        # Tools: prefer MCP-discovered tools; fall back to response/session tools
        req_tools = self.tools if self.tools else None
        if req_tools is None and response and response.tools:
            req_tools = response.tools
        if req_tools is None:
            req_tools = runtime_config.session.tools

        # Generation counter for cancellation
        gen = self.cancel_scope.generation if self.cancel_scope else None

        yield from self._generate(
            active_chat=active_chat,
            original_chat=original_chat,
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            req_tools=req_tools,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
        )

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        runtime_config: Any,
        response: Any,
        req_tools: Optional[list],
        turn_id: str | None,
        turn_revision: int | None,
        speech_stopped_at_s: float | None,
    ) -> Iterator[LLMOut]:
        """Multi-turn generation loop with MCP tool execution."""
        cancelled = False
        full_text = ""  # Accumulated text across all iterations

        for iteration in range(5):
            messages = _build_openai_messages(active_chat)
            logger.debug("=== LLM iteration %d: sending %d messages ===", iteration, len(messages))

            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=req_tools if req_tools else None,
                    stream=True,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout,
                    extra_body=self._extra_body,
                    **self.gen_kwargs,
                )
            except httpx.ReadTimeout:
                logger.warning(
                    "Chat Completions API read timed out after %.1fs",
                    self.request_timeout_s,
                )
                if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
                    yield LLMResponseChunk(
                        text="Wow I'm a bit slow today, could you repeat that?",
                        runtime_config=runtime_config,
                        response=response,
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                        speech_stopped_at_s=speech_stopped_at_s,
                        cancel_generation=gen,
                    )
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                )
                return
            except Exception:
                logger.exception("Chat Completions API call failed on iteration %d", iteration)
                cancelled = True
                break

            # --- collect streaming deltas ---
            tool_calls_collected: dict[int, dict] = {}
            text_parts: list[str] = []
            printable_text = ""
            reasoning_text = ""
            sentence_batch: list[str] = []
            finish_reason: Optional[str] = None
            input_tokens = 0
            output_tokens = 0
            event_count = 0
            text_count = 0

            for event in stream:
                event_count += 1
                if event_count % 500 == 0:
                    logger.debug("Stream event %d | text chars so far: %d", event_count, len("".join(text_parts)))

                # Cancellation check
                if self._generation_is_stale(gen):
                    logger.info("LLM generation cancelled (interruption)")
                    cancelled = True
                    break

                choices = getattr(event, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.delta

                # Usage (some providers emit it on the final chunk)
                usage = getattr(event, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0

                # Reasoning content — accumulate and update on single line
                reasoning_content = delta.model_extra.get("reasoning_content")
                if reasoning_content:
                    reasoning_text += reasoning_content
                    sys.stderr.write(f"{reasoning_content}")
                    sys.stderr.flush()
                    continue

                # Text delta — spoken content only
                if delta.content:
                    text_count += 1
                    new_text = remove_unspeechable(delta.content)
                    text_parts.append(new_text)
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            sentence_batch.append(s)
                            if len(sentence_batch) >= self.stream_batch_sentences:
                                if self._generation_is_stale(gen):
                                    cancelled = True
                                    break
                                if not self._turn_output_allowed(turn_id, turn_revision):
                                    logger.info("LLM generation cancelled (stale speculative turn)")
                                    cancelled = True
                                    break
                                logger.debug("STREAMING CHUNK: %s", " ".join(sentence_batch))
                                yield LLMResponseChunk(
                                    text=" ".join(sentence_batch),
                                    language_code=language_code,
                                    runtime_config=runtime_config,
                                    response=response,
                                    turn_id=turn_id,
                                    turn_revision=turn_revision,
                                    speech_stopped_at_s=speech_stopped_at_s,
                                    cancel_generation=gen,
                                )
                                sentence_batch = []
                        if cancelled:
                            break
                        printable_text = sentences[-1]

                # Tool call delta
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_collected:
                            tool_calls_collected[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = tool_calls_collected[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            try:
                stream.close()
            except Exception:
                pass

            if reasoning_text.strip():
                sys.stderr.write("\n")
            logger.debug("Stream ended: %d events | %d text deltas | total text chars: %d | finish_reason: %s",
                         event_count, text_count, len("".join(text_parts)), finish_reason)

            if cancelled:
                break

            # --- handle finish_reason ---
            if finish_reason == "length":
                logger.warning(
                    "LLM hit max_tokens (%d) on iteration %d; response may be truncated",
                    self.max_tokens,
                    iteration,
                )
                full_text = "".join(text_parts).strip()
                if not full_text:
                    full_text = "I'm sorry, I lost my train of thought. Could you repeat that?"
                yield LLMResponseChunk(
                    text=full_text,
                    language_code=language_code,
                    runtime_config=runtime_config,
                    response=response,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                    cancel_generation=gen,
                )
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                )
                return

            if finish_reason == "tool_calls":
                if not tool_calls_collected:
                    logger.warning("finish_reason=tool_calls but no tool calls collected")
                    break

                if self.mcp_client is None:
                    logger.warning("LLM requested tool calls but MCP is not enabled; breaking out of tool loop")
                    full_text = "".join(text_parts).strip()
                    break

                # Merge assistant message with tool_calls into active_chat
                assistant_text = "".join(text_parts)
                tc_list: list[dict] = []
                for tc in tool_calls_collected.values():
                    tc_list.append(
                        {
                            "id": tc["id"] if tc.get("id", "").startswith("call_") else _generate_id("call"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"] or "{}",
                            },
                        }
                    )
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": assistant_text if assistant_text else None,
                    "tool_calls": tc_list,
                }
                active_chat.buffer.append(assistant_msg)

                # Record assistant message in original chat for persistent history
                original_chat.buffer.append(assistant_msg)

                # Execute each tool call
                for tc in tool_calls_collected.values():
                    tc_id = tc["id"] if tc.get("id", "").startswith("call_") else _generate_id("call")
                    logger.info("Tool call: %s(%s)", tc["name"], tc["arguments"])
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                        logger.warning(
                            "Invalid JSON arguments for tool %s: %s",
                            tc["name"],
                            tc["arguments"],
                        )

                    result = self.mcp_client.execute_tool(tc["name"], args)
                    logger.info("Tool result (%s): %s", tc["name"], result[:200])

                    # Append tool result to active_chat for next LLM prompt
                    tool_msg = _make_tool_message(tc_id, result)
                    active_chat.buffer.append(tool_msg)

                    # Record tool result in original chat for persistent history
                    original_chat.buffer.append(tool_msg)

                # Re-prompt LLM with tool results
                continue

            # LLM returned text — flush remaining and break
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                remaining = " ".join(sentence_batch)
                if remaining:
                    yield LLMResponseChunk(
                        text=remaining,
                        language_code=language_code,
                        runtime_config=runtime_config,
                        response=response,
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                        speech_stopped_at_s=speech_stopped_at_s,
                        cancel_generation=gen,
                    )

            full_text = "".join(text_parts).strip()
            if full_text:
                logger.debug("=== LLM iteration %d final output: %s ===", iteration, full_text)
            break

        # --- post-generation bookkeeping ---
        if not cancelled and full_text:
            original_chat.add_item(make_assistant_message(full_text))
        elif not cancelled and iteration >= 4:
            # Tool loop exhausted without producing text — give the user something
            logger.warning("Tool loop exhausted without text response; yielding fallback")
            full_text = "I wasn't able to find the information you were looking for. Could you try rephrasing your question?"
            yield LLMResponseChunk(
                text=full_text,
                language_code=language_code,
                runtime_config=runtime_config,
                response=response,
                turn_id=turn_id,
                turn_revision=turn_revision,
                speech_stopped_at_s=speech_stopped_at_s,
                cancel_generation=gen,
            )
            original_chat.add_item(make_assistant_message(full_text))

        original_chat.strip_images()
        original_chat.trim_if_needed(self.compactor)

        if input_tokens or output_tokens:
            yield TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                turn_id=turn_id,
                turn_revision=turn_revision,
            )

        yield EndOfResponse(
            turn_id=turn_id,
            turn_revision=turn_revision,
            cancel_generation=gen,
        )

    # ---- timing -----------------------------------------------------------

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
