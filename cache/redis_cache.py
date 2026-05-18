import json
import uuid
import numpy as np
from redis.asyncio import Redis
from langchain_huggingface import HuggingFaceEmbeddings

from config import REDIS_URL


class SemanticCache:
    def __init__(self):
        self.redis = Redis.from_url(REDIS_URL)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.ttl = 3600
        self.similarity_threshold = 0.92

    async def get(self, query: str) -> str | None:
        query_embedding = await self.embeddings.aembed_query(query)
        query_vec = np.array(query_embedding)

        keys = await self.redis.keys("policy_cache:*")
        for key in keys:
            cached_data = await self.redis.get(key)
            if not cached_data:
                continue
            cached = json.loads(cached_data)
            stored_vec = np.array(cached["embedding"])

            dot_product = np.dot(query_vec, stored_vec)
            norm_a = np.linalg.norm(query_vec)
            norm_b = np.linalg.norm(stored_vec)
            if norm_a == 0 or norm_b == 0:
                continue
            similarity = dot_product / (norm_a * norm_b)

            if similarity >= self.similarity_threshold:
                return cached["result"]

        return None

    async def set(self, query: str, result: str):
        embedding = await self.embeddings.aembed_query(query)
        cache_key = f"policy_cache:{uuid.uuid4()}"
        data = json.dumps({
            "query": query,
            "embedding": embedding,
            "result": result,
        })
        await self.redis.set(cache_key, data, ex=self.ttl)

    async def invalidate_all(self):
        keys = await self.redis.keys("policy_cache:*")
        if keys:
            await self.redis.delete(*keys)
