"""
agents/llm_client.py

LLM abstraction layer for the Physics Discovery Agent.
Only uses Groq by default.
"""

import os
import time
from abc import ABC, abstractmethod

# Attempt to load .env natively or via python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate a text response given a prompt and optional system message."""
        pass


class GroqClient(LLMClient):
    """Groq client — provides extremely fast, FREE Llama 3 models. Uses GROQ_API_KEY."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GROQ_API_KEY environment variable (e.g. in .env file)."
            )

        # Groq uses the exact same API signature as OpenAI
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self._model_name = model_name

    def generate(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Simple retry logic for Groq
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print(f"  [Groq] Rate limited. Waiting {5 * (attempt + 1)}s...")
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

    def __repr__(self):
        return f"GroqClient({self._model_name})"


def get_llm_client(prefer: str = "groq") -> LLMClient:
    """
    Returns the Groq client since Gemini was removed.

    Returns:
        LLMClient instance

    Raises:
        RuntimeError if initialization fails.
    """
    try:
        return GroqClient()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize GroqClient. Ensure GROQ_API_KEY is set in .env.\nError: {e}"
        )
