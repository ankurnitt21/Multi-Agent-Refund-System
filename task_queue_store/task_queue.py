"""
Deprecated: Redis hash+set queue replaced by Kafka event bus.

Use task_queue_store.kafka_events.KafkaRefundEventBus instead.
"""
from task_queue_store.kafka_events import KafkaRefundEventBus

# Backward-compatible alias
PersistentTaskQueue = KafkaRefundEventBus

__all__ = ["KafkaRefundEventBus", "PersistentTaskQueue"]
