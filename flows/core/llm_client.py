from typing import List, Optional
from openai import AsyncOpenAI
import asyncio
import json
from pathlib import Path
from functools import lru_cache

class LLMClient:
    def __init__(self, api_key: str, cache_size: int = 1000, cache_file: str = "llm_cache.json"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        # Apply caching decorator to internal generate method
        self._generate_cached = lru_cache(maxsize=cache_size)(self._generate)
        
    def _load_cache(self) -> dict:
        """Load cached responses from file"""
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text())
        return {}
        
    def _save_cache(self):
        """Save cache to file"""
        self.cache_file.write_text(json.dumps(self.cache, indent=2))
        
    def _get_cache_key(self, prompt: str, system_prompt: str, temperature: float) -> str:
        """Generate cache key with temperature bucketing"""
        temp_bucket = round(temperature * 2) / 2  # Rounds to nearest 0.5
        return f"{prompt}:{system_prompt}:{temp_bucket}"
        
    def _generate(self, prompt: str, system_prompt: str = None, 
                 temperature: float = 0.7, max_tokens: int = 100) -> str:
        """Internal generation method"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    def generate(self, prompt: str, system_prompt: str = None,
                temperature: float = 0.7, max_tokens: int = 100) -> str:
        """Generate with caching"""
        cache_key = self._get_cache_key(prompt, system_prompt, temperature)
        
        # Check persistent cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Try memory cache next
        try:
            response = self._generate_cached(prompt, system_prompt, temperature, max_tokens)
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise
            
        # Save to persistent cache
        self.cache[cache_key] = response
        self._save_cache()
        
        return response
        
    async def generate_async(self, prompt: str, system_prompt: str = None,
                           temperature: float = 0.7, max_tokens: int = 100) -> str:
        """Async generation method"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    def batch_generate(self, prompts: List[str], system_prompt: str = None,
                      batch_size: int = 5) -> List[str]:
        """Generate multiple responses in batches"""
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            # Process batch using async
            batch_responses = asyncio.run(self._process_batch(batch, system_prompt))
            responses.extend(batch_responses)
        return responses
        
    async def _process_batch(self, prompts: List[str], 
                           system_prompt: str = None) -> List[str]:
        """Process a batch of prompts concurrently"""
        tasks = [
            self.generate_async(prompt, system_prompt)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)