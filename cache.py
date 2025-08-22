"""Cache management using Redis."""

import json
from typing import Any, Dict, Optional
import redis.asyncio as redis
from loguru import logger

from ai_trends.utils.exceptions import CacheError


class CacheManager:
    """Redis-based cache manager."""

    def __init__(self):
        """Initialize cache manager."""
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False

    async def initialize(self, redis_url: str) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {str(e)}")
            self.is_connected = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_connected or not self.redis_client:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        if not self.is_connected or not self.redis_client:
            return False
            
        try:
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_connected or not self.redis_client:
            return False
            
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {str(e)}")
            return False

    async def clear_all(self) -> bool:
        """Clear all cached data."""
        if not self.is_connected or not self.redis_client:
            return False
            
        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_connected or not self.redis_client:
            return {"connected": False, "error": "Not connected"}
            
        try:
            info = await self.redis_client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "Unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {str(e)}")
            return {"connected": False, "error": str(e)}

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Cache connection closed")