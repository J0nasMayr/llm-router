# src/queue/redis_client.py
import json
import time
import uuid

import redis


class RedisClient:
    """Client for managing Redis task queue"""

    def __init__(self, host="localhost", port=6379, password=None, db=0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,  # Automatically decode responses to strings
        )
        # Test connection
        self.redis.ping()
        print(f"Successfully connected to Redis at {host}:{port}")

        self.result_prefix = "result:"
        self.telemetry_prefix = "telemetry:node:"
        self.placement_prefix = "placement:node:"

    def enqueue_task(self, task_data, queue_name):
        """Add task to queue and return task ID"""
        # uuid suffix keeps IDs unique even for concurrent same-millisecond tasks
        task_id = (
            f"task:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
            f":{task_data.get('query_id', 'unknown')}"
        )
        task_data["id"] = task_id
        task_json = json.dumps(task_data)

        # Store task data and add to queue using Redis
        self.redis.set(task_id, task_json)
        self.redis.lpush(queue_name, task_id)

        return task_id

    def get_next_task(self, queue_name, timeout=1):
        """Get next task from queue, waiting up to timeout seconds.

        queue_name may be a single queue or a list of queues; with a list,
        BRPOP pops from the first non-empty queue (earlier entries win).
        """
        # Use BRPOP to wait for a task with timeout
        result = self.redis.brpop(queue_name, timeout)
        if not result:
            return None

        # Get task data from its ID
        _, task_id = result
        task_json = self.redis.get(task_id)

        if not task_json:
            return None

        return json.loads(task_json)

    def set_result(self, task_id, result_data):
        """Store result for a completed task"""
        result_id = f"{self.result_prefix}{task_id}"
        result_json = json.dumps(result_data)

        self.redis.set(result_id, result_json)

    def get_result(self, task_id):
        """Get result for a task, or None if it is not finished"""
        result_json = self.redis.get(f"{self.result_prefix}{task_id}")
        return json.loads(result_json) if result_json else None

    def get_queue_length(self, queue_name):
        """Return the number of pending tasks in the given queue."""
        try:
            return int(self.redis.llen(queue_name))
        except Exception:
            return 0

    def set_node_telemetry(self, node_id, payload, ttl_seconds=15):
        """Publish a node's telemetry snapshot under telemetry:node:<node_id> with TTL."""
        key = f"{self.telemetry_prefix}{node_id}"
        body = json.dumps(payload)
        self.redis.set(key, body, ex=ttl_seconds)

    def set_node_placement(self, node_id, model_ids, ttl_seconds=None):
        """Publish the models a node should keep loaded (its target placement,
        most important first). The allocation controller renews it with a TTL,
        so a plan expires once the controller stops renewing it."""
        key = f"{self.placement_prefix}{node_id}"
        body = json.dumps(list(model_ids))
        self.redis.set(key, body, ex=ttl_seconds)

    def get_node_placement(self, node_id):
        """The node's target placement, or None if no plan names the node."""
        key = f"{self.placement_prefix}{node_id}"
        raw = self.redis.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def get_all_node_telemetry(self):
        """Return list of telemetry payloads for every active node (TTL'd entries auto-expire)."""
        results = []
        try:
            for key in self.redis.scan_iter(match=f"{self.telemetry_prefix}*"):
                raw = self.redis.get(key)
                if raw:
                    try:
                        results.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return results
