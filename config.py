import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/refund_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "warehouse-refund-policies")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "warehouse-refund-system")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.smith.langchain.com/otel")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "warehouse-refund-system")
