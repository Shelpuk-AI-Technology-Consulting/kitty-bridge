"""Shared types used across kitty sub-packages."""

from enum import Enum


class BridgeProtocol(str, Enum):
    """Wire protocol spoken between the coding agent and the local bridge."""

    RESPONSES_API = "responses"
    MESSAGES_API = "messages"
    GEMINI_API = "gemini"
    CHAT_COMPLETIONS_API = "chat_completions"
