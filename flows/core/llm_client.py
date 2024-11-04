from typing import List, Optional
from openai import AsyncOpenAI
import asyncio
import json
from pathlib import Path
from functools import lru_cache

class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", max_tokens: int = 150):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.cache = {}
        self.cache_file = Path("llm_cache.json")
        self.cache = self._load_cache()
        
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
        
    async def _generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Make the actual API call"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in LLM generation: {str(e)}")
            return ""
        
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = None) -> str:
        """Generate response using the LLM"""
        cache_key = self._get_cache_key(prompt, system_prompt, temperature)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # If not in cache, generate response
        response = await self._generate(prompt, system_prompt, temperature, max_tokens)
        
        # Cache the response
        self.cache[cache_key] = response
        self._save_cache()
        
        return response
        
    async def _generate_cached(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate with caching"""
        cache_key = self._create_cache_key(prompt, system_prompt, temperature, max_tokens)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        response = await self._generate(prompt, system_prompt, temperature, max_tokens)
        self.cache[cache_key] = response
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