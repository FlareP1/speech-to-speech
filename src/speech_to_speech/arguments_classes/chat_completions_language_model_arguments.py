from dataclasses import dataclass, field

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(LanguageModelBaseArguments):
    chat_completions_api_base_url: str = field(
        default="http://127.0.0.1:8080/v1",
        metadata={
            "help": "Base URL for the OpenAI-compatible chat completions API endpoint. Default is 'http://127.0.0.1:8080/v1'."
        },
    )
    chat_completions_api_api_key: str = field(
        default="none",
        metadata={"help": "API key used to authenticate access to the chat completions API. Default is 'none'."},
    )
    chat_completions_api_model: str = field(
        default="Qwen3.6-27B",
        metadata={"help": "The model to use with the chat completions API. Default is 'Qwen3.6-27B'."},
    )
    mcp_server_url: str = field(
        default="http://127.0.0.1:8008",
        metadata={"help": "Base URL for the MCP server. Default is 'http://127.0.0.1:8008'."},
    )
    mcp_servers: str = field(
        default="time,ddg-search",
        metadata={"help": "Comma-separated list of MCP servers to register. Default is 'time,ddg-search'."},
    )
    mcp_enabled: bool = field(
        default=False,
        metadata={"help": "Enable MCP tool use with the chat completions backend. Default is False."},
    )
    chat_completions_max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens to generate per response. Default is 4096."},
    )
    chat_completions_request_timeout_s: float = field(
        default=120.0,
        metadata={"help": "Request timeout in seconds for chat completions API calls. Default is 120."},
    )
