# src/queue/redis_client.py
import asyncio
import json
import time

import redis


class RedisClient:
    """Client for managing Redis task queue"""

    def __init__(self, host="localhost", port=6379, password=None, db=0):
        try:
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
        except Exception as e:
            print(f"Failed to connect to Redis at {host}:{port}: {e}")
            # Create a dummy in-memory store for testing
            self._dummy_store = {}
            self._dummy_queue = []

        self.default_queue = "llm_inference_tasks"
        self.result_prefix = "result:"

    def _is_connected(self):
        """Check if Redis connection is active"""
        try:
            return hasattr(self, "redis") and self.redis.ping()
        except Exception:
            return False

    def enqueue_task(self, task_data, queue_name=None):
        """Add task to queue and return task ID"""
        task_id = (
            f"task:{int(time.time() * 1000)}:{task_data.get('query_id', 'unknown')}"
        )
        task_data["id"] = task_id
        task_json = json.dumps(task_data)

        # Use the provided queue or fall back to default
        target_queue = queue_name or self.default_queue

        if self._is_connected():
            # Store task data and add to queue using Redis
            self.redis.set(task_id, task_json)
            self.redis.lpush(target_queue, task_id)
        else:
            # Use in-memory storage for testing
            print("In memory")
            self._dummy_store[task_id] = task_json
            self._dummy_queue.append(task_id)

        return task_id

    def get_next_task(self, queue_name=None, timeout=1):
        """Get next task from queue, waiting up to timeout seconds"""
        target_queue = queue_name or self.default_queue

        if self._is_connected():
            # Use BRPOP to wait for a task with timeout
            result = self.redis.brpop(target_queue, timeout)
            if not result:
                return None

            # Get task data from its ID
            _, task_id = result
            task_json = self.redis.get(task_id)

            if not task_json:
                return None

            return json.loads(task_json)
        else:
            # Use in-memory storage for testing
            if not self._dummy_queue:
                return None

            task_id = self._dummy_queue.pop()
            task_json = self._dummy_store.get(task_id)

            if not task_json:
                return None

            return json.loads(task_json)

    def set_result(self, task_id, result_data):
        """Store result for a completed task"""
        result_id = f"{self.result_prefix}{task_id}"
        result_json = json.dumps(result_data)

        if self._is_connected():
            self.redis.set(result_id, result_json)
        else:
            self._dummy_store[result_id] = result_json

    def get_result(self, task_id, timeout=None):
        """Get result for a task, optionally waiting for completion"""
        result_id = f"{self.result_prefix}{task_id}"

        # If no timeout, just get current result
        if timeout is None:
            if self._is_connected():
                result_json = self.redis.get(result_id)
            else:
                result_json = self._dummy_store.get(result_id)

            if not result_json:
                return None
            return json.loads(result_json)

        # With timeout, poll until timeout expires
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_connected():
                result_json = self.redis.get(result_id)
            else:
                result_json = self._dummy_store.get(result_id)

            if result_json:
                return json.loads(result_json)

            time.sleep(0.1)

        return None

    async def wait_for_result(self, task_id, timeout=60):
        """Async wait for result with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result(task_id)
            if result:
                return result

            # Sleep briefly to avoid hammering Redis
            await asyncio.sleep(0.1)

        # Timeout expired
        raise TimeoutError(f"Timeout waiting for result of task {task_id}")
