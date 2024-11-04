from typing import List, Any, Callable, TypeVar, Dict
import asyncio
from dataclasses import dataclass
import logging

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchConfig:
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent_batches: int = 5

class BatchProcessor:
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)
        
    async def process_batches(self, 
                            items: List[T],
                            process_fn: Callable[[T], R],
                            **kwargs) -> List[R]:
        """Process items in batches using provided function"""
        results = []
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]
        
        # Process batches with concurrency limit
        async with asyncio.Semaphore(self.config.max_concurrent_batches):
            batch_tasks = [
                self._process_batch_with_retries(batch, process_fn, **kwargs)
                for batch in batches
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
                    
        return results
        
    async def _process_batch_with_retries(self,
                                        batch: List[T],
                                        process_fn: Callable,
                                        **kwargs) -> List[R]:
        """Process a single batch with retries"""
        for attempt in range(self.config.max_retries):
            try:
                return await self._process_batch(batch, process_fn, **kwargs)
            except Exception as e:
                self.logger.warning(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
        
    async def _process_batch(self,
                           batch: List[T],
                           process_fn: Callable,
                           **kwargs) -> List[R]:
        """Process a single batch"""
        tasks = [
            process_fn(item, **kwargs)
            for item in batch
        ]
        return await asyncio.gather(*tasks) 