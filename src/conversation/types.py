"""Type definitions for conversation module."""

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_api_format(self) -> dict:
        """Convert to LiteLLM/OpenAI API format."""
        return {"role": self.role, "content": self.content}
