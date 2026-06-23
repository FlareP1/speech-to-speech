from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(ResponsesApiLanguageModelHandlerArguments):
    """Arguments for the ``chat-completions`` LLM backend.

    Inherits the OpenAI-compatible connection fields from the Responses-API
    arguments (``responses_api_base_url`` / ``responses_api_api_key`` /
    ``responses_api_stream`` / ``responses_api_disable_thinking``) so the same
    CLI flags and launcher env vars drive both backends, and adds the
    Chat-Completions-only ``reasoning_effort`` knob.
    """

    responses_api_reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provider-specific reasoning level sent as extra_body={'reasoning_effort': <value>} on the "
            "Chat Completions request. Use to disable reasoning on providers where "
            "chat_template_kwargs.enable_thinking is ignored (e.g. 'none' / 'low'). When unset, falls back to "
            "the disable_thinking behaviour (chat_template_kwargs.enable_thinking=false). Default is None."
        },
    )
    mcp_server_url: Optional[str] = field(
        default="http://127.0.0.1:8008",
        metadata={"help": "Base URL for the mcp-proxy server."},
    )
    mcp_servers: Optional[str] = field(
        default="time,ddg-search",
        metadata={"help": "Comma-separated list of MCP server names to connect to."},
    )
    mcp_enabled: bool = field(
        default=False,
        metadata={"help": "Enable MCP tool calling with the chat completions backend."},
    )
    responses_api_request_timeout_s: float = field(
        default=20.0,
        metadata={
            "help": "Request timeout in seconds for API calls. Default is 20.0."
        },
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens to generate per request. Default is 4096."},
    )
    suppress_tool_call_flush: bool = field(
        default=True,
        metadata={
            "help": "When True, suppresses streaming text output before tool calls, holding text "
            "until the final response. Use True for smooth playback with MCP tools (old behavior). "
            "Use False to speak preamble text before tools execute (new behavior). Default is True."
        },
    )
