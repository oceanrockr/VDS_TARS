"""
T.A.R.S. Ollama Service
Integration with Ollama for LLM inference and token streaming
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional
import json

import httpx
from ..core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Service for interacting with Ollama LLM API.

    Features:
    - Async token streaming
    - Connection pooling
    - Retry logic with exponential backoff
    - Performance metrics
    """

    def __init__(self):
        """Initialize Ollama service"""
        self.base_url = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
        self.timeout = httpx.Timeout(120.0, connect=10.0)

        # Create async HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

        logger.info(f"OllamaService initialized: {self.base_url} (model={self.model})")

    async def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming response from Ollama.

        Args:
            prompt: User prompt
            model: Model to use (defaults to configured model)
            temperature: Temperature override
            max_tokens: Max tokens override
            system_prompt: Optional system prompt

        Yields:
            Dictionary containing token and metadata
        """
        model = model or self.model
        temperature = temperature if temperature is not None else settings.MODEL_TEMPERATURE
        max_tokens = max_tokens or settings.MODEL_MAX_TOKENS

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": settings.MODEL_TOP_P,
                "top_k": settings.MODEL_TOP_K,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        token_count = 0
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                async with self.client.stream(
                    "POST",
                    "/api/generate",
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)

                            # Extract token from response
                            if "response" in chunk:
                                token = chunk["response"]
                                token_count += 1

                                yield {
                                    "token": token,
                                    "done": chunk.get("done", False),
                                    "token_count": token_count,
                                    "model": model,
                                }

                                # Check if generation is complete
                                if chunk.get("done", False):
                                    elapsed_time = time.time() - start_time
                                    logger.info(
                                        f"Generation complete: {token_count} tokens in "
                                        f"{elapsed_time:.2f}s "
                                        f"({token_count / elapsed_time:.1f} tokens/s)"
                                    )
                                    return

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Ollama response: {e}")
                            continue

                # If we reach here, stream completed successfully
                return

            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error: {e.response.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise

            except httpx.RequestError as e:
                logger.error(f"Ollama request error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in Ollama stream: {e}")
                raise

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a non-streaming response from Ollama.

        Args:
            prompt: User prompt
            model: Model to use
            temperature: Temperature override
            max_tokens: Max tokens override
            system_prompt: Optional system prompt

        Returns:
            Dictionary with response and metadata
        """
        model = model or self.model
        temperature = temperature if temperature is not None else settings.MODEL_TEMPERATURE
        max_tokens = max_tokens or settings.MODEL_MAX_TOKENS

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": settings.MODEL_TOP_P,
                "top_k": settings.MODEL_TOP_K,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()

            return {
                "response": result.get("response", ""),
                "model": model,
                "done": result.get("done", False),
            }

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def list_models(self) -> list[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            return models

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()
        logger.info("OllamaService closed")


# Global Ollama service instance
ollama_service = OllamaService()
