# MergePlan.md: Option A — Migrate to BaseOpenAICompatibleHandler

## Date: 2026-06-23
## Status: Ready to execute

---

## Goal

Replace our standalone `ChatCompletionsApiModelHandler` (633 lines) with one that extends `BaseOpenAICompatibleHandler` (upstream base class, 489 lines of shared lifecycle). The MCP tool loop overrides `process()` to wrap base class generation in a multi-turn loop. Target: ~250 lines.

**Current state:**
- `main` branch: Aligned with upstream v0.2.10, includes `BaseOpenAICompatibleHandler`
- `mcp-tool-call` branch: Standalone handler with MCP loop, tiktoken, dict handling (633 lines)
- 547 tests passing, pipeline verified working

---

## Safety Net

**Before any changes — create rollback points:**

```bash
git branch backup/pre-baseclass-migration-2
git tag backup/pre-baseclass-migration-2
```

Rollback command (if needed):
```bash
git reset --hard backup/pre-baseclass-migration-2
```

---

## Analysis: What We Keep vs What We Drop

### Keep from our handler (unique value, ~200 lines)

| Method/Feature | Lines | Purpose |
|----------------|-------|---------|
| `_to_chat_tools()` / `_to_chat_tool_choice()` | ~50 | Tool format conversion (identical to upstream) |
| `_to_chat_content_part()` / `_chat_messages()` | ~50 | Chat serialization (identical to upstream) |
| `_tool_calls_from_accum()` | ~20 | Tool call event builder (identical to upstream) |
| `_serialize()`, `_build_optional_kwargs()`, `_request()`, `_iter_stream_events()`, `_iter_response_events()` | ~80 | Base class hooks (identical to upstream) |
| MCP imports + `MCPClient` integration | ~30 | MCP tool execution |
| Tiktoken counting + context % logging | ~50 | Debug logging |
| `_get_max_context()` | ~20 | llama.cpp context detection |
| `_build_compaction_generate_fn()` | ~15 | Compaction generate function |
| `warmup()` | ~10 | Warmup call |

### Drop from our handler (provided by base class, ~400 lines)

| Method/Feature | Lines | Replaced by |
|----------------|-------|-------------|
| `setup()` | ~70 | Base class `setup()` |
| `_turn_is_latest()`, `_turn_output_allowed()`, `_generation_is_stale()` | ~15 | Inherited |
| `_chunk()` | ~15 | Inherited |
| `_consume_streaming()` / `_consume_nonstreaming()` | ~100 | Inherited |
| `_apply_config()` | ~10 | Inherited |
| `timing_log_level` / `should_log_timing()` | ~5 | Inherited |
| `process()` (non-MCP parts) | ~80 | Inherited, overridden with MCP loop |
| Manual `LLMResponseChunk`, `EndOfResponse`, `TokenUsage` handling | ~100 | Base class orchestration |
| `_record_tool_call()`, `_GenState`, `_Turn` | ~50 | Inherited |
| `on_session_end()`, `cleanup()` | ~10 | Inherited / simplified |

---

## Step 1: Create Backup

```bash
git branch backup/pre-baseclass-migration-2
git tag backup/pre-baseclass-migration-2
```

## Step 2: Rewrite `chat_completions_language_model.py`

### New class structure

```python
class ChatCompletionsApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler for OpenAI-compatible /v1/chat/completions servers.
    
    Extends BaseOpenAICompatibleHandler for shared lifecycle (cancellation,
    speculative turns, sentence batching, text-only mode, out-of-band responses).
    Overrides process() to add MCP tool loop with tiktoken context logging.
    """
```

### Methods to implement (override base class hooks)

1. **`warmup()`** — Keep as-is (already matches base class signature)

2. **`_build_compaction_generate_fn()`** — Keep as-is, adjust to use `self._extra_body`

3. **`_serialize(active_chat: Chat)`** — Keep as-is, calls `self._chat_messages()`

4. **`_build_optional_kwargs(req_tools, req_tool_choice)`** — Keep as-is

5. **`_request(api_input, optional_kwargs)`** — Keep as-is

6. **`_iter_stream_events(api_response)`** — Keep as-is

7. **`_iter_response_events(api_response)`** — Keep as-is

8. **`process(request: LLMIn)`** — **Override with MCP loop** (see below)

### Helper methods (remain as-is)

- `_to_chat_tools()` / `_to_chat_tool_choice()` — module-level functions
- `_to_chat_content_part()` / `_chat_messages()` — class/static methods
- `_tool_calls_from_accum()` — static method
- `_get_max_context()` — instance method
- `_count_tokens()` — instance method

### New `process()` implementation

The base class `process()` handles: turn gating, chat setup, language resolution, config application, and calls `_generate()`. Our override needs to do the same setup, then loop for MCP tools:

```python
def process(self, request: LLMIn) -> Iterator[LLMOut]:
    # --- Turn gating (from base class) ---
    if not self._turn_is_latest(turn_id, turn_revision):
        yield EndOfResponse(...)
        return

    # --- Chat setup (from base class) ---
    original_chat = runtime_config.chat
    if is_out_of_band(response):
        active_chat = build_active_chat(original_chat, response)
    else:
        active_chat = original_chat.copy()
    
    # Language, instructions, tools, optional_kwargs (from base class)
    self._apply_config(active_chat, instructions, wants_audio)
    optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)
    gen = self.cancel_scope.generation if self.cancel_scope else None
    
    # --- MCP tool loop ---
    state = _GenState()  # Accumulate across iterations
    for iteration in range(5):
        # Context % logging (our addition)
        messages = self._chat_messages(active_chat)
        est_tokens = self._count_tokens(messages)
        max_ctx = self._get_max_context()
        pct = (est_tokens / max_ctx * 100) if max_ctx else 0
        logger.info("iteration %d: %d messages, est_tokens=%d/%d (%.1f%%)", ...)

        # Check stale before each iteration
        if self._generation_is_stale(gen) or not self._turn_is_latest(...):
            logger.info("LLM generation cancelled (interruption)")
            break

        # Call API
        api_input = self._serialize(active_chat)
        api_response = self._request(api_input, optional_kwargs)
        events = self._iter_events(api_response)
        
        # Consume events
        collected_text = ""
        tool_calls = []
        for event in events:
            if isinstance(event, TextDelta):
                text = remove_unspeechable(event.text) if wants_audio else event.text
                collected_text += text
                if not self._generation_is_stale(gen) and self._turn_output_allowed(...):
                    yield self._chunk(turn, text=text)
            elif isinstance(event, ToolCall):
                tool_calls.append(event.item)
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens

        # Check if we need another iteration
        if not tool_calls:
            break  # Normal text response
        if self.mcp_client is None:
            logger.warning("Tool calls requested but MCP not enabled")
            break

        # Execute tools via MCP, append to active_chat
        assistant_text = collected_text.strip()
        tc_list = []
        for tc in tool_calls:
            tc_id = tc.call_id or _generate_id("call")
            tc_list.append({"id": tc_id, "name": tc.name, "arguments": tc.arguments})
        
        # Append assistant message with tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": assistant_text if assistant_text else None,
            "tool_calls": tc_list,
        }
        active_chat.buffer.append(assistant_msg)

        # Execute each tool
        for tc in tool_calls:
            tc_id = tc.call_id or _generate_id("call")
            logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
            try:
                args = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                args = {}
            result = self.mcp_client.execute_tool(tc.name, args)
            
            # Append tool result
            active_chat.buffer.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

    # --- Post-loop: write history, emit usage, EndOfResponse (from base class) ---
    if not is_out_of_band(response):
        for item in state.pending:
            original_chat.add_item(item)
        original_chat.strip_images()
        original_chat.trim_if_needed(self.compactor)
    
    if state.input_tokens or state.output_tokens:
        yield TokenUsage(...)
    
    yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, cancel_generation=gen)
```

### Key design decisions

1. **Suppress `EndOfResponse` on intermediate iterations** — Only emit after the loop completes
2. **History write-back after loop** — Manually replicate base class logic
3. **Context % logging** — Call `self._chat_messages()` + `self._count_tokens()` before each iteration
4. **Dict handling** — `active_chat.buffer.append(dict)` during tool loop; `_chat_messages()` handles dicts via `chat.to_transformers_chat()` (already fixed in `chat.py`)

## Step 3: Update `chat_completions_language_model_arguments.py`

Add missing fields our handler needs:

```python
responses_api_request_timeout_s: float = field(
    default=120.0,
    metadata={"help": "Request timeout in seconds for API calls (default: 120.0)."},
)
```

Note: `max_tokens` is not in the argument class — it goes in `gen_kwargs` or uses the handler default.

## Step 4: Update imports

```python
from speech_to_speech.LLM.base_openai_compatible_language_model import (
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    ToolCall,
    Usage,
    _GenState,
    _Turn,
)
```

Remove unused imports:
- `httpx` (base class handles timeouts)
- `nltk.sent_tokenize` (base class handles sentence batching)
- `remove_unspeechable`, `resolve_auto_language` (base class handles these)
- `build_voice_system_prompt`, `build_text_system_prompt` (base class handles these)
- `CancelScope`, `SpeculativeTurnTracker` (base class handles these)

Keep:
- `tiktoken` (our addition)
- `MCPClient` (our addition)
- `_generate_id` (used in tool loop)
- `is_out_of_band`, `build_active_chat` (from chat.py, used in process override)

## Step 5: Run Tests

```bash
uv run pytest tests/ -x -q
```

Expected: 547 tests passing (our handler tests may need updating for new class structure).

## Step 6: Manual Test

Run the pipeline with MCP tools to verify tool loop works:

```bash
python -m speech_to_speech.s2s_pipeline --mode local --stt parakeet-tdt --llm_backend chat-completions --tts qwen3 --responses_api_base_url "http://127.0.0.1:8080/v1" --responses_api_api_key "none" --mcp_server_url "http://127.0.0.1:8008" --mcp_servers "time,ddg-search" --mcp_enabled --enable_live_transcription --init_chat_prompt "Your name is Myra" --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-Base --qwen3_tts_language english --qwen3_tts_instruct "soft_female_voice" --qwen3_tts_ref_audio ..\MYRA_referenceShort.wav --qwen3_tts_ref_text "Hi there" --min_silence_ms 500
```

Test scenarios:
1. Simple Q&A (no tools) — verify text response works
2. Time query — verify `get_current_time` tool call
3. Weather query — verify `search` tool call
4. Interruption — verify cancellation works

## Step 7: Commit & Push

```bash
git add src/speech_to_speech/LLM/chat_completions_language_model.py
git add src/speech_to_speech/arguments_classes/chat_completions_language_model_arguments.py
git commit -m "Migrate ChatCompletionsApiModelHandler to BaseOpenAICompatibleHandler

- Extend BaseOpenAICompatibleHandler for shared lifecycle (cancellation,
  speculative turns, sentence batching, text-only mode, out-of-band responses)
- Override process() with MCP tool loop (up to 5 iterations)
- Keep tiktoken context % logging and _get_max_context() for llama.cpp
- Remove ~400 lines of duplicated lifecycle logic
- Handler reduced from 633 to ~250 lines"
git push origin mcp-tool-call
```

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| MCP loop breaks after migration | Medium | Backup branch, tested baseline before migration |
| Base class `_GenState` / `_Turn` internals change | Low | Only used in `process()` override, stable interface |
| Context % logging lost | Low | Keep as helper methods, call from new `process()` |
| Dict handling breaks | Low | Already in `chat.py`, independent of handler |
| Argument class field mismatch | Low | `rename_args()` strips prefix, base class `setup()` handles standard fields |
| `request_timeout_s` default changes (20.0 → 120.0) | Low | Add to argument class with our default |

---

## Rollback Plan

If migration results in broken code:

```bash
git reset --hard backup/pre-baseclass-migration-2
```

This restores the branch to the working standalone handler.

---

## Files Changed

| File | Change |
|------|--------|
| `chat_completions_language_model.py` | Rewrite: extend base class, override `process()` with MCP loop |
| `chat_completions_language_model_arguments.py` | Add `responses_api_request_timeout_s` field |
| `tests/test_chat_completions_language_model.py` | Update tests for new class structure |

---

## Expected Result

- Handler: 633 lines → ~250 lines (60% reduction)
- Duplicated lifecycle code: 400 lines → 0 lines
- MCP tool loop: preserved, ~100 lines
- Tiktoken + context logging: preserved, ~50 lines
- Benefits: text-only mode, out-of-band responses, timeout handling, `reasoning_effort` support, future-proofed against upstream changes
