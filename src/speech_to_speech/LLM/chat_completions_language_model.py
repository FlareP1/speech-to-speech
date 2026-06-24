"""LLM backend using OpenAI Chat Completions API (llama.cpp compatible).

Uses ``client.chat.completions.create()`` instead of the Responses API.
Supports tool calling with MCP tool execution in a multi-turn loop.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx
import tiktoken
from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall

from nltk import sent_tokenize

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    ToolCall,
    Usage,
    _GenState,
    _Turn,
)
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
    build_active_chat,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.LLM.mcp_client import MCPClient
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

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


class ChatCompletionsApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM backend using OpenAI Chat Completions API (llama.cpp compatible).

    Extends BaseOpenAICompatibleHandler for shared lifecycle. Overrides process()
    for MCP tool loop with tiktoken context logging.
    """

    def setup(
        self,
        model_name: str = "Qwen3.6-27B",
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = True,
        reasoning_effort: str | None = None,
        mcp_server_url: str | None = None,
        mcp_servers: str | None = None,
        mcp_enabled: bool = False,
        suppress_tool_call_flush: bool = True,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        request_timeout_s: float = 120.0,
        max_tokens: int = 4096,
        compact_history: bool = True,
        init_chat_prompt_voice_lead: str = "",
        init_chat_prompt_voice_lead_file: str | None = None,
        init_chat_prompt_voice_tail: str = "",
        init_chat_prompt_voice_tail_file: str | None = None,
        gen_kwargs: dict[str, Any] = {},
        **_kwargs: Any,
    ) -> None:
        super().setup(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            disable_thinking=disable_thinking,
            reasoning_effort=reasoning_effort,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            stream_batch_sentences=stream_batch_sentences,
            enable_lang_prompt=enable_lang_prompt,
            request_timeout_s=request_timeout_s,
            compact_history=compact_history,
            init_chat_prompt_voice_lead=init_chat_prompt_voice_lead,
            init_chat_prompt_voice_lead_file=init_chat_prompt_voice_lead_file,
            init_chat_prompt_voice_tail=init_chat_prompt_voice_tail,
            init_chat_prompt_voice_tail_file=init_chat_prompt_voice_tail_file,
            gen_kwargs=gen_kwargs,
        )
        self.stream = True  # Always stream; _iter_response_events not supported
        self.max_tokens = max(1, max_tokens)
        self.suppress_tool_call_flush = suppress_tool_call_flush
        self.mcp_client: MCPClient | None = None
        self.tools: list[dict] = []
      # Deferred commit: hold messages until next turn, then insert at correct position
        self._pending_messages: list | None = None
        self._pending_turn_id: str | None = None
        self._pending_turn_revision: int | None = None
        self._pending_insert_pos: int | None = None
        self._pending_prev_baseline: int | None = None
        # Baseline tracking for reopen detection
        self._turn_baseline: int = 0
        self._prev_baseline: int = 0

        if mcp_enabled and mcp_server_url:
            server_list = [s.strip() for s in (mcp_servers or "").split(",") if s.strip()]
            if server_list:
                self.mcp_client = MCPClient(base_url=mcp_server_url, servers=server_list)
                self.tools = self.mcp_client.connect()
                logger.info("MCP: discovered %d tool(s): %s", len(self.tools), [t.get("function", {}).get("name", "?") for t in self.tools])
            else:
                logger.warning("mcp_enabled=True but no servers specified; tools disabled")
        elif mcp_enabled:
            logger.warning("mcp_enabled=True but mcp_server_url not set; tools disabled")

    def _get_max_context(self) -> int:
        """Try to get the model's max context length via GET /v1/models."""
        try:
            resp = self.client.models.list(timeout=self.request_timeout)
            data = resp.data
            if data:
                m = data[0]
                ctx = getattr(m, "max_context_length", None) or getattr(m, "context_length", None)
                if ctx:
                    return int(ctx)
                extra = getattr(m, "model_extra", None)
                if extra:
                    meta = extra.get("meta", {})
                    n_ctx = meta.get("n_ctx") or meta.get("n_ctx_train")
                    if n_ctx:
                        return int(n_ctx)
        except Exception:
            pass
        return 0

    def _count_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a messages list using tiktoken."""
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = len(messages) * 3

        for msg in messages:
            tokens += len(enc.encode(str(msg.get("role", ""))))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") in ("input_text", "output_text")
                )
            tokens += len(enc.encode(str(content)))
            for tc in msg.get("tool_calls", []):
                tokens += len(enc.encode(json.dumps(tc)))
            if msg.get("tool_call_id"):
                tokens += len(enc.encode(msg["tool_call_id"]))

        if self.tools:
            tokens += len(enc.encode(json.dumps(self.tools)))

        return max(1, tokens)

    def _resolve_model(self, fallback_model: str | None) -> str:
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

    def warmup(self) -> None:
        self.model_name = self._resolve_model(self.model_name)
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

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout
        extra_body = self._extra_body

        def generate(system: str, user: str) -> str:
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 1024,
                "timeout": timeout,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        return generate

    # ── streaming override: conditional tool-call flush ───────────────────────

    def _consume_streaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            yield self._chunk(turn, text=" ".join(batch))

        for event in events:
            if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                # Flush pending text before tool call only when suppress_tool_call_flush is False
                if not self.suppress_tool_call_flush:
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                        printable_text = ""
                    if sentence_batch:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield from _flush(sentence_batch)
                        sentence_batch = []
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                if not turn.wants_audio:
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(s)
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield from _flush(sentence_batch)
                            sentence_batch = []
                        if cancelled:
                            break
                    printable_text = sentences[-1]

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._generation_is_stale(turn.gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug("Clean text: %s", state.clean_text)
                    yield from _flush(sentence_batch)
            logger.info("Tools: %s", state.tools)

    def _serialize(self, active_chat: Chat) -> Any:
        return _build_openai_messages(active_chat)

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        tools = self.tools if self.tools else (req_tools if req_tools else None)
        optional_kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if tools is not None:
            optional_kwargs["tools"] = tools
        if req_tool_choice is not None:
            optional_kwargs["tool_choice"] = req_tool_choice
        return optional_kwargs

    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=api_input,
            stream=True,
            timeout=self.request_timeout,
            extra_body=self._extra_body,
            **optional_kwargs,
            **self.gen_kwargs,
        )

    def _iter_stream_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        tool_calls_collected: dict[int, dict] = {}
        for event in api_response:
            choices = getattr(event, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = choice.delta

            usage = getattr(event, "usage", None)
            if usage:
                yield Usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )

            if delta.content:
                yield TextDelta(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_collected:
                        tool_calls_collected[idx] = {"id": "", "name": "", "arguments": ""}
                    entry = tool_calls_collected[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

            if getattr(choice, "finish_reason", None) == "tool_calls":
                for tc in tool_calls_collected.values():
                    item = ResponseFunctionToolCall(
                        id=_generate_id("fc"),
                        call_id=tc["id"] if tc["id"].startswith("call_") else _generate_id("call"),
                        type="function_call",
                        name=tc["name"],
                        arguments=tc["arguments"] or "{}",
                    )
                    yield ToolCall(item=item)

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        raise NotImplementedError("Non-streaming not supported")

    # ── process override: MCP tool loop ──────────────────────────────────────

    def _try_commit_pending(self, new_turn_id: str, new_turn_revision: int, original_chat: Chat) -> None:
        """Commit pending messages from previous process() if turn advanced (not reopened).

        Uses baseline tracking to insert at correct position (before user message)
        and to truncate previous revision's messages on reopen.
        """
        if self._pending_messages is None:
            return

        # If same turn_id but higher revision, the turn was reopened - discard pending and old user message
        if self._pending_turn_id == new_turn_id and new_turn_revision > self._pending_turn_revision:
            logger.debug("Turn %s reopened (rev %d -> %d), discarding rev %d pending messages and old user message",
                         new_turn_id, self._pending_turn_revision, new_turn_revision,
                         self._pending_turn_revision)
            # Save new messages added since previous baseline (the new user message for this revision)
            new_msgs = original_chat.buffer[self._prev_baseline:]
            # Reconstruct buffer: history (before old turn) + new messages only
            prev_bl = self._pending_prev_baseline if self._pending_prev_baseline is not None else 0
            original_chat.buffer[:] = original_chat.buffer[:prev_bl] + new_msgs
            self._prev_baseline = prev_bl
            self._turn_baseline = len(original_chat.buffer)
            self._clear_pending()
            return

        # Insert pending at saved position (before user message from this turn)
        insert_pos = self._pending_insert_pos if self._pending_insert_pos is not None else len(original_chat.buffer)
        original_chat.buffer[insert_pos:insert_pos] = self._pending_messages
        original_chat.strip_images()
        original_chat.trim_if_needed(self.compactor)
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_messages = None
        self._pending_turn_id = None
        self._pending_turn_revision = None
        self._pending_insert_pos = None
        self._pending_prev_baseline = None

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
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
        # Save previous baseline before updating, for reopen truncation
        self._prev_baseline = self._turn_baseline
        self._turn_baseline = len(original_chat.buffer)
        self._try_commit_pending(turn_id, turn_revision, original_chat)
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()

        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)
        gen = self.cancel_scope.generation if self.cancel_scope else None

        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
        )

        cancelled = False
        error_message: str | None = None
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(5):
            messages = _build_openai_messages(active_chat)
            est_tokens = self._count_tokens(messages)
            max_ctx = self._get_max_context()
            pct = (est_tokens / max_ctx * 100) if max_ctx else 0
            logger.info("=== LLM iteration %d: %d messages, buffer=%d items, est_tokens=%d/%d (%.1f%%) ===",
                         iteration, len(messages), len(active_chat.buffer), est_tokens, max_ctx, pct)
            # DEBUG: log each message sent to LLM
            for i, m in enumerate(messages):
                role = m.get("role", "?")
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 100:
                    content = content[:100] + "..."
                tc_info = f" tool_calls={len(m.get('tool_calls', []))}" if m.get("tool_calls") else ""
                logger.info("  msg[%d] %s: %s%s", i, role, content, tc_info)

            if self._generation_is_stale(gen) or not self._turn_is_latest(turn_id, turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            api_response: Any = None
            state = _GenState()
            try:
                api_input = self._serialize(active_chat)
                api_response = self._request(api_input, optional_kwargs)
                events = self._iter_events(api_response)
                yield from self._consume_streaming(events, state, turn)
            except httpx.ReadTimeout:
                logger.warning("Chat Completions API read timed out after %.1fs", self.request_timeout_s)
                if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
                    yield self._chunk(turn, text="Wow I'm a bit slow today, could you repeat that?")
                error_message = f"API read timeout after {self.request_timeout_s:.1f}s"
                cancelled = True
            except Exception as exc:
                logger.exception("Chat Completions API call failed on iteration %d", iteration)
                error_message = f"Language model generation failed: {exc}"
                cancelled = True
            finally:
                if api_response is not None and hasattr(api_response, "close"):
                    try:
                        api_response.close()
                    except Exception:
                        pass

            if cancelled:
                break

            total_input_tokens = max(total_input_tokens, state.input_tokens)
            total_output_tokens += state.output_tokens

            if not state.tools:
                break

            if self.mcp_client is None:
                logger.warning("LLM requested tool calls but MCP is not enabled")
                break

            tc_list: list[dict] = []
            for tc in state.tools:
                tc_list.append({
                    "id": tc.call_id or _generate_id("call"),
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments or "{}",
                    },
                })
            assistant_msg: dict = {
                "role": "assistant",
                "content": state.clean_text if state.clean_text.strip() else None,
                "tool_calls": tc_list,
            }
            active_chat.buffer.append(assistant_msg)

            for tc in state.tools:
                tc_id = tc.call_id or _generate_id("call")
                logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                    logger.warning("Invalid JSON arguments for tool %s: %s", tc.name, tc.arguments)
                result = self.mcp_client.execute_tool(tc.name, args)
                logger.info("Tool result (%s): %s", tc.name, result[:200])
                tool_msg = _make_tool_message(tc_id, result)
                active_chat.buffer.append(tool_msg)

        if not cancelled and not is_out_of_band(response):
            # Add assistant's final text response to active_chat.buffer if not already there
            if state.clean_text.strip():
                from speech_to_speech.LLM.chat import make_assistant_message
                active_chat.add_item(make_assistant_message(state.clean_text))

            # Collect all buffer entries that aren't in original_chat
            pending = [entry for entry in active_chat.buffer if entry not in original_chat.buffer]
            self._pending_messages = pending
            self._pending_insert_pos = len(original_chat.buffer)
            self._pending_turn_id = turn_id
            self._pending_turn_revision = turn_revision
            self._pending_prev_baseline = self._prev_baseline
            logger.debug("Deferred: saved %d pending messages at insert_pos=%d for turn=%s rev=%s",
                         len(pending), self._pending_insert_pos, turn_id, turn_revision)

        if not cancelled and iteration >= 4 and not state.clean_text.strip():
            fallback = "I wasn't able to find the information you were looking for. Could you try rephrasing your question?"
            yield self._chunk(turn, text=fallback)
            if not is_out_of_band(response):
                from speech_to_speech.LLM.chat import make_assistant_message
                # Add to pending for deferred commit
                if self._pending_messages is not None:
                    self._pending_messages.append(make_assistant_message(fallback))
                else:
                    # If pending was already committed (shouldn't happen normally), commit directly
                    original_chat.add_item(make_assistant_message(fallback))

        if total_input_tokens or total_output_tokens:
            yield TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                turn_id=turn_id,
                turn_revision=turn_revision,
            )

        yield EndOfResponse(
            turn_id=turn_id,
            turn_revision=turn_revision,
            cancel_generation=gen,
            error=error_message,
        )

    def on_session_end(self) -> None:
        logger.debug("%s: session state reset", self.__class__.__name__)
        # Discard any leftover pending messages on session end
        self._pending_messages = None
        self._pending_turn_id = None
        if self.mcp_client:
            self.mcp_client.disconnect()
            self.mcp_client = None
            self.tools = []

    def cleanup(self) -> None:
        if self.mcp_client:
            self.mcp_client.disconnect()
