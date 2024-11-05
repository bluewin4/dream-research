from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

class LLMClient:
    """Asynchronous OpenAI client wrapper with retry logic"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize the LLM client
        
        Args:
            api_key: Optional API key. If not provided, will use OPENAI_API_KEY env variable
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_model = "gpt-4"
        
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response asynchronously with retry logic
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to set context/personality
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to gpt-4)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                messages: List[Dict[str, str]] = []
                
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    
                messages.append({"role": "user", "content": prompt})
                
                response = await self.client.chat.completions.create(
                    model=model or self.default_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    print(f"Final error in generate after {attempt} attempts: {str(e)}")
                    return f"Error: {str(e)}"
                else:
                    print(f"Attempt {attempt} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> List[str]:
        """Generate multiple responses asynchronously
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt applied to all generations
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of generated responses
        """
        tasks = [
            self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)