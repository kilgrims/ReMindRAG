from .base import AgentBase
from openai import OpenAI
import time
from typing import Optional, List, Dict, Any
from openai import APIConnectionError, APIError, RateLimitError
import requests.exceptions
import urllib3.exceptions 
import socket

class OpenaiAgent(AgentBase):
    def __init__(self, base_url: str, api_key: str, llm_model_name: str, time_out: int = 120, max_retries: int = 3, retry_delay: float = 1.0):
        self.base_url = base_url
        self.llm_model_name = llm_model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=time_out
        )
        
    def generate_response(self, system_prompt: Optional[str], chat_history: List[Dict[str, Any]]) -> str:
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + chat_history
        else:
            messages = chat_history

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    seed=123,
                    temperature=0
                )
                return response.choices[0].message.content
                
            except (APIConnectionError, APIError, RateLimitError,
                   requests.exceptions.ConnectionError,
                   requests.exceptions.RequestException,
                   urllib3.exceptions.ProtocolError,
                   urllib3.exceptions.HTTPError,
                   ConnectionResetError,
                   socket.timeout,
                   TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise Exception(f"Failed after {self.max_retries} retries. Last error: {str(last_error)}") from last_error