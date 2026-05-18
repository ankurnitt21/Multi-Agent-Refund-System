import json
import time

import structlog
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

logger = structlog.get_logger(__name__)

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from cache.redis_cache import SemanticCache

POLICY_DOCUMENTS = [
    {
        "id": "general_policy",
        "text": "General return policy: all products returnable within standard window from purchase date. Proof of purchase required. Items must be in original condition unless defective. Refunds processed within 5-7 business days.",
    },
    {
        "id": "electronics_policy",
        "text": "Electronics return policy: 15 day return window from delivery. 15% restocking fee applies. Defective items exempt from restocking fees. Original packaging required. Serial number must match purchase record.",
    },
    {
        "id": "furniture_policy",
        "text": "Furniture return policy: 30 day return window. 10% restocking fee. Assembly voids return policy unless item is defective. Delivery and assembly fees non-refundable.",
    },
    {
        "id": "gold_tier_policy",
        "text": "Gold tier benefits: return window extended by 15 additional days for all categories. Restocking fees completely waived. Priority processing within 24 hours. Maximum 5 refunds per calendar year.",
    },
    {
        "id": "silver_tier_policy",
        "text": "Silver tier benefits: return window extended by 7 additional days. 50% restocking fee waiver. Standard processing within 48 hours. Maximum 3 refunds per year.",
    },
    {
        "id": "bronze_tier_policy",
        "text": "Bronze tier benefits: standard return windows apply. Full restocking fees apply. Standard processing within 72 hours. Maximum 2 refunds per calendar year.",
    },
    {
        "id": "fraud_policy",
        "text": "Fraud detection policy: customers with more than 3 refund requests in 90 days are flagged. Refund rate above 50% triggers risk review. Risk score above 0.8 results in automatic denial. Suspicious patterns reported to fraud team.",
    },
    {
        "id": "eligibility_policy",
        "text": "Eligibility rules: order must have delivered status. Purchase date within return window including tier extensions. Account must be in active standing. Customer risk score must be below threshold. Previous refunds counted against annual limits.",
    },
]


class PineconeStore:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        self.cache = SemanticCache()
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _create_index(self) -> None:
        self.pc.create_index(
            name=self.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        for _ in range(60):
            desc = self.pc.describe_index(self.index_name)
            status = desc.status
            ready = (
                status.get("ready", False)
                if isinstance(status, dict)
                else getattr(status, "ready", False)
            )
            if ready or (isinstance(status, dict) and status.get("state") == "Ready"):
                return
            time.sleep(2)
        raise TimeoutError(f"Pinecone index '{self.index_name}' not ready after 120s")

    def _ensure_index(self) -> None:
        existing = {idx.name for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            self._create_index()
            return

        desc = self.pc.describe_index(self.index_name)
        index_dim = desc.dimension
        if index_dim != EMBEDDING_DIMENSION:
            logger.warning(
                "pinecone_index_dimension_mismatch",
                index=self.index_name,
                current=index_dim,
                expected=EMBEDDING_DIMENSION,
            )
            self.pc.delete_index(self.index_name)
            for _ in range(60):
                if self.index_name not in {idx.name for idx in self.pc.list_indexes()}:
                    break
                time.sleep(2)
            self._create_index()

    async def build_policy_index(self):
        vectors = []
        for doc in POLICY_DOCUMENTS:
            embedding = await self.embeddings.aembed_query(doc["text"])
            vectors.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": {"text": doc["text"]},
            })
        self.index.upsert(vectors=vectors)

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        cached = await self.cache.get(query)
        if cached:
            return json.loads(cached)

        embedding = await self.embeddings.aembed_query(query)
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)

        output = []
        for match in results.matches:
            output.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
            })

        await self.cache.set(query, json.dumps(output))
        return output
