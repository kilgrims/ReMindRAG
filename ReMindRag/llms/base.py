from abc import ABC, abstractmethod
from typing import List, Dict

class AgentBase(ABC):
    @abstractmethod
    def generate_response(self, system_prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Abstract method to generate a response based on a system prompt and chat history.

        Args:
            system_prompt (str): A string containing the system's instructions or context.
            chat_history (list[dict[str, str]]): A list representing the chat history, where in openai forms.

        Returns:
            str: The generated response as a string.
        """
        pass
    