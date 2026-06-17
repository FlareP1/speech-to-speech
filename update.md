# Update Summary: MCP Tool Support & Chat Completions Backend

## Branch: `mcp-tool-call`

This branch adds a new `chat-completions` LLM backend with MCP (Model Context Protocol) tool calling support, along with pipeline fixes and dependency hardening.

---

## Functional Changes

### 1. New `chat-completions` LLM Backend

A new LLM backend using the OpenAI Chat Completions API (`client.chat.completions.create()`) instead of the Responses API. This is designed for use with llama.cpp-compatible servers (Ollama, LocalAI, etc.) that expose a `/v1/chat/completions` endpoint.

**Key features:**
- Streaming text deltas with sentence-batched TTS forwarding
- MCP tool calling in a multi-turn loop (up to 5 iterations)
- Thinking suppression via `extra_body` for local models
- Model auto-detection via `GET /v1/models`
- Reasoning content logging to stderr
- Fallback response when tool loop exhausts without producing text
- Compaction support (same as responses API)
- Speculative turn cancellation (same as responses API)
- Passthrough of arbitrary `gen_kwargs` to the API call

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

---

## New CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--llm_backend chat-completions` | `responses-api` | Select the chat completions backend (new value alongside `transformers`, `mlx-lm`, `responses-api`) |
| `--chat_completions_api_base_url` | `http://127.0.0.1:8080/v1` | Base URL for the OpenAI-compatible chat completions API |
| `--chat_completions_api_api_key` | `none` | API key for the chat completions endpoint |
| `--chat_completions_api_model` | `Qwen3.6-27B` | Model name (auto-detected from `/v1/models` if available) |
| `--mcp_server_url` | `http://127.0.0.1:8008` | Base URL for the mcp-proxy server |
| `--mcp_servers` | `time,ddg-search` | Comma-separated list of MCP server names to connect to |
| `--mcp_enabled` | `False` | Enable MCP tool calling with the chat completions backend |
| `--chat_completions_max_tokens` | `4096` | Maximum tokens to generate per response |
| `--chat_completions_request_timeout_s` | `120.0` | Request timeout in seconds for API calls |

Inherited from `LanguageModelBaseArguments` (also available for this backend):
- `--chat_size` — number of conversation turns to keep in context (default: 10)
- `--compact_history` — enable conversation history compaction
- `--disable_thinking` — suppress `<thinking>` blocks (default: `True`)
- `--speculative_turns` — speculative turn tracking settings

---

## New Files

| File | Description |
|------|-------------|
| `src/speech_to_speech/LLM/chat_completions_language_model.py` | Chat completions LLM handler with MCP tool loop |
| `src/speech_to_speech/LLM/mcp_client.py` | Synchronous MCP client (JSON-RPC 2.0 over HTTP) |
| `src/speech_to_speech/arguments_classes/chat_completions_language_model_arguments.py` | CLI argument definitions for the chat completions backend |
| `Install.bat` | Windows setup script: creates venv, runs `uv sync`, installs CUDA torch |
| `Run Speech Detection.bat` | Updated to use `uv` venv and `speech-to-speech` entry point |

---

## Modified Files

| File | Changes |
|------|---------|
| `pyproject.toml` | Pinned CUDA torch versions, added PyTorch CUDA index for `uv sync` |
| `src/speech_to_speech/arguments_classes/module_arguments.py` | Added `chat-completions` as valid `llm_backend` option |
| `src/speech_to_speech/s2s_pipeline.py` | Registered chat completions args, routed kwargs correctly, updated backend dispatcher |

---

## Example Usage

```bash
speech-to-speech --mode local \
  --stt parakeet-tdt \
  --llm_backend chat-completions \
  --tts qwen3 \
  --chat_completions_api_base_url http://127.0.0.1:8080/v1 \
  --chat_completions_api_api_key none \
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
Added 11 new tests: 8 for chat-completions backend (tool loop, fallback, compaction, message building) and 3 for dict handling in compaction/serialization.

## Commits

```
2b17548 Add tiktoken as runtime dependency
9152148 Fix compaction crashes and add accurate token counting
79fe50a Fix init_chat_prompt for chat-completions backend, add tool loop fallback, pin CUDA torch
4f90236 Align chat completions backend with responses API
d7f84c1 Add reasoning content logging and debug output to chat completions backend
5fb960c Merge branch 'mcp-tool-call' of https://github.com/FlareP1/speech-to-speech into mcp-tool-call
1e0712e Revert pre-parse to use _use_responses_api boolean
a7e841b Update bat files for uv venv and MCP support
553a21b Update bat files for uv venv and MCP support
91dd93b Add chat-completions LLM backend with MCP tool support
```
