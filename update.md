# Update Summary: MCP Tool Support & Chat Completions Backend

## Branch: `mcp-tool-call`

This branch adds a new `chat-completions` LLM backend with MCP (Model Context Protocol) tool calling support, along with pipeline fixes and dependency hardening. As of June 2026, 36 commits from upstream `huggingface/speech-to-speech` (v0.2.10) have been merged.

---

## Upstream Merge (v0.2.10)

36 commits merged from `huggingface/speech-to-speech` upstream:

### BaseOpenAICompatibleHandler
New shared base class for OpenAI-compatible LLM backends (`BaseOpenAICompatibleHandler`). Extracts common lifecycle logic (speculative-turn gating, cancellation, sentence batching, text-only vs audio handling, history write-back, token usage, out-of-band responses, error termination) so both the Responses API and Chat Completions handlers share the same orchestration code.

### VAD Improvements
- Speech continuation hysteresis (192ms threshold)
- Noise floor applied to early speech start and stitch gaps
- Adjacent short VAD segment stitching
- Low-latency timing profile defaults
- Unanswered speculative turn reopening past grace window

### Out-of-Band Responses
- `build_active_chat()` / `add_supported_item()` in `chat.py`
- Text-only and `conversation=none` response support
- Assistant transcript emission for fresh responses when discard guard is stuck

### Tool Calling Fixes
- Tool follow-up deferral until outputs complete
- Tool-only realtime response completion fix
- Voice prompt lead-ins shortened and improved
- Usage attribution preserved across response boundaries

### Paraformer
- Progressive transcription events fix
- Optional dependency import guard

### Speculative Turns
- 334+ new tests for speculative turn handling
- Unanswered speech turn merging

### Version
- Bumped to 0.2.10

---

## Functional Changes

### 1. New `chat-completions` LLM Backend

A new LLM backend using the OpenAI Chat Completions API (`client.chat.completions.create()`) instead of the Responses API. This is designed for use with llama.cpp-compatible servers (Ollama, LocalAI, etc.) that expose a `/v1/chat/completions` endpoint.

**Key features:**
- Streaming text deltas with sentence-batched TTS forwarding
- MCP tool calling in a multi-turn loop (up to 5 iterations)
- Thinking suppression via `extra_body` for local models
- `reasoning_effort` support (upstream addition)
- Model auto-detection via `GET /v1/models`
- Reasoning content logging to stderr
- Fallback response when tool loop exhausts without producing text
- Compaction support (same as responses API)
- Speculative turn cancellation (same as responses API)
- Passthrough of arbitrary `gen_kwargs` to the API call
- Accurate token counting with tiktoken and context % logging

**Note:** The handler is currently standalone (not extending `BaseOpenAICompatibleHandler`). Migration to the base class is deferred to a follow-up PR.

### 2. MCP Tool Calling

A new `MCPClient` class communicates with [mcp-proxy](https://github.com/chrishayuk/mcp-proxy) over plain HTTP POST (JSON-RPC 2.0). No extra `mcp` package dependency required.

**Flow:**
1. At startup, `MCPClient` connects to each configured server, initializes a session, and discovers available tools
2. Tools are converted to OpenAI function-calling format and passed to the LLM
3. When the LLM returns `finish_reason: tool_calls`, the handler executes each tool via `mcp-proxy`, appends results, and re-prompts
4. Loop continues until the LLM returns text or exhausts 5 iterations

**Default tools** (with `--mcp_servers time,ddg-search`):
- `get_current_time` — returns current time
- `convert_time` — converts between time zones
- `search` — DuckDuckGo web search
- `fetch_content` — fetches web page content

### 3. Pipeline Fixes

- **init_chat_prompt routing:** `s2s_pipeline.py` now correctly routes `chat_size` and `init_chat_prompt` to the `chat_completions_language_model_handler_kwargs` when `--llm_backend chat-completions` is used.
- **Tool loop fallback:** When the tool loop exhausts all 5 iterations without producing spoken text, the handler yields a fallback message: *"I wasn't able to find the information you were looking for. Could you try rephrasing your question?"*

### 4. Dependency Fixes

- `pyproject.toml` now pins `torch`, `torchaudio`, and `torchvision` to `+cu126` versions on Windows/Linux
- Added `[[tool.uv.index]]` pointing to the PyTorch CUDA index, so `uv sync` installs CUDA wheels instead of CPU-only builds
- Added `tiktoken>=0.3.3` for accurate token counting

---

## CLI Options

The `chat-completions` backend shares connection flags with the `responses-api` backend (via `ChatCompletionsLanguageModelHandlerArguments` subclassing).

| Option | Default | Description |
|--------|---------|-------------|
| `--llm_backend chat-completions` | `responses-api` | Select the chat completions backend |
| `--responses_api_base_url` | `None` | Base URL for the OpenAI-compatible API (used by both backends) |
| `--responses_api_api_key` | `None` | API key (used by both backends) |
| `--responses_api_disable_thinking` | `True` | Suppress `<thinking>` blocks (used by both backends) |
| `--responses_api_reasoning_effort` | `None` | Provider-specific reasoning level (chat-completions only) |
| `--model_name` | `Qwen3.6-27B` | Model name (auto-detected from `/v1/models` if available) |
| `--mcp_server_url` | `http://127.0.0.1:8008` | Base URL for the mcp-proxy server |
| `--mcp_servers` | `time,ddg-search` | Comma-separated list of MCP server names |
| `--mcp_enabled` | `False` | Enable MCP tool calling |

Inherited from `LanguageModelBaseArguments`:
- `--chat_size` — conversation turns to keep in context (default: 30)
- `--compact_history` — enable conversation history compaction (default: True)
- `--stream_batch_sentences` — sentences per batch (default: 3)
- `--enable_lang_prompt` — append language instruction (default: False)

---

## New Files

| File | Description |
|------|-------------|
| `src/speech_to_speech/LLM/chat_completions_language_model.py` | Chat completions LLM handler with MCP tool loop (our version) |
| `src/speech_to_speech/LLM/mcp_client.py` | Synchronous MCP client (JSON-RPC 2.0 over HTTP) |
| `src/speech_to_speech/LLM/base_openai_compatible_language_model.py` | Shared base class (upstream — not yet used by our handler) |
| `src/speech_to_speech/LLM/text_prompt.py` | Text-only system prompt builder (upstream) |
| `src/speech_to_speech/arguments_classes/chat_completions_language_model_arguments.py` | CLI arguments (extends ResponsesApi, adds MCP fields) |
| `Install.bat` | Windows setup script: creates venv, runs `uv sync`, installs CUDA torch |
| `Run Speech Detection.bat` | Updated to use `uv` venv and `speech-to-speech` entry point |

---

## Modified Files

| File | Changes |
|------|---------|
| `pyproject.toml` | CUDA torch pins, PyTorch CUDA index, tiktoken, version 0.2.10 |
| `src/speech_to_speech/LLM/chat.py` | Dict handling in compaction + upstream `build_active_chat()` |
| `src/speech_to_speech/arguments_classes/module_arguments.py` | Added `chat-completions` as valid `llm_backend` |
| `src/speech_to_speech/s2s_pipeline.py` | Upstream pipeline with `chat-completions` dispatcher |
| `src/speech_to_speech/VAD/vad_handler.py` | VAD improvements from upstream |
| `src/speech_to_speech/STT/paraformer_handler.py` | Paraformer progressive events fix |
| `tests/test_chat.py` | Dict handling tests + upstream `TestBuildActiveChat` |
| `tests/test_chat_completions_language_model.py` | 8 tests for handler (tool loop, fallback, compaction, messages) |

---

## Example Usage

```bash
speech-to-speech --mode local \
  --stt parakeet-tdt \
  --llm_backend chat-completions \
  --tts qwen3 \
  --responses_api_base_url http://127.0.0.1:8080/v1 \
  --responses_api_api_key none \
  --mcp_server_url http://127.0.0.1:8008 \
  --mcp_servers time,ddg-search \
  --mcp_enabled \
  --enable_live_transcription \
  --init_chat_prompt "Your name is Myra" \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_tts_language english \
  --qwen3_tts_instruct "soft_female_voice" \
  --qwen3_tts_ref_audio MYRA_referenceShort.wav \
  --qwen3_tts_ref_text "Hi there"
```

**Note:** The connection flags use the `--responses_api_*` prefix (shared naming convention). The handler still calls the Chat Completions API (`/v1/chat/completions`) for tool calls.

---

## Bug Fixes & Improvements

### Compaction Dict Handling
The chat-completions backend stores plain `dict` objects (assistant messages with tool_calls, tool result messages) in the chat buffer alongside `RealtimeConversationItem*` objects. Fixed 3 places that accessed `.id` without checking for dicts:
- `_snapshot_for_compaction`: skip dicts when collecting `marker_ids`
- `_to_responses_api_chat_locked`: pass dicts through unchanged before `.id` assertion
- `_apply_compaction`: skip dicts when computing `drop_ids` and `remaining`

### Compaction Thinking Suppression
The compaction LLM call didn't pass `extra_body` (enable_thinking=False), causing the model to output `<thinking>` blocks or empty content, which caused `_extract_json` to fail. Fixed by passing `extra_body` to the compaction generate function.

### Max Context Detection for llama.cpp
llama.cpp models don't expose `max_context_length` or `context_length` fields. Added `_get_max_context()` that checks both standard OpenAI fields and `model_extra['meta']['n_ctx']`.

### Accurate Token Counting
Replaced 4-char-per-token heuristic with `tiktoken`-based counting that uses the correct encoding per model name (with cl100k_base fallback). Counts tool definitions and tool call metadata.

### Tests
Added 11 new tests: 8 for chat-completions backend (tool loop, fallback, compaction, message building) and 3 for dict handling in compaction/serialization. Upstream added 334+ tests for speculative turns, VAD, and realtime service.

### Test Count
**547 tests passing** (upstream: ~446, our additions: 11, shared: ~90)

---

## Deferred Work (Follow-up PRs)

### Migrate to BaseOpenAICompatibleHandler
Refactor `ChatCompletionsApiModelHandler` to extend `BaseOpenAICompatibleHandler`. Benefits: better cancellation, text-only mode, timeout handling, out-of-band responses. MCP tool loop would need to be adapted (either wrapped around base class `process()` or moved to pipeline level).

### Move Token Counting to Utilities
Move `_count_tokens()` and `_get_max_context()` to `utils/utils.py` for reuse across backends.

---

## Commits

```
36c147c Restore custom AGENTS.md with project-specific rules
07475fe Merge upstream/main: VAD improvements, out-of-band responses, base class, spec turns
365fd29 Update branch notes: compaction fixes, token counting, tests
2b17548 Add tiktoken as runtime dependency
9152148 Fix compaction crashes and add accurate token counting
72fb521 Add update.md summarizing mcp-tool-call branch changes
79fe50a Fix init_chat_prompt for chat-completions backend, add tool loop fallback, pin CUDA torch
4f90236 Align chat completions backend with responses API
d7f84c1 Add reasoning content logging and debug output to chat completions backend
5fb960c Merge branch 'mcp-tool-call' of https://github.com/FlareP1/speech-to-speech into mcp-tool-call
1e0712e Revert pre-parse to use _use_responses_api boolean
a7e841b Update bat files for uv venv and MCP support
553a21b Update bat files for uv venv and MCP support
91dd93b Add chat-completions LLM backend with MCP tool support
```
