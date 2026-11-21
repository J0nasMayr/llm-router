import json
import logging
import time
import uuid

import redis

from config.manager import load_config

logger = logging.getLogger(__name__)


def node_queue_name(node_id):
    """The task queue of one worker node."""
    return f"llm_tasks_node_{node_id}"


# Event logs for the experiments: one entry per model load (workers) and one
# per plan (placement controller).
LOAD_EVENTS = "loads"
PLAN_EVENTS = "plans"


def tier_queue_name(tier):
    """The task queue every worker of a tier also listens on: the
    orchestrator's fallback when no node of the model's tier meets the
    deadline, or before any telemetry has arrived."""
    return f"llm_inference_tasks_{tier}"


class RedisClient:
    """Client for managing Redis task queue"""

    result_prefix = "result:"
    telemetry_prefix = "telemetry:node:"
    placement_prefix = "placement:node:"
    events_prefix = "events:"

    def __init__(self, host, port, password=None, db=0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,  # Automatically decode responses to strings
        )
        # Test connection
        self.redis.ping()
        logger.info(f"Connected to Redis at {host}:{port}")

    @classmethod
    def from_config(cls, host=None, port=None, password=None):
        """A client for the redis block of config/infrastructure.yaml;
        arguments that are not None override it."""
        cfg = load_config("infrastructure")["redis"]
        return cls(
            host=host or cfg["host"],
            port=port or int(cfg["port"]),
            password=password or cfg.get("password"),
        )

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
        most important first). The placement controller renews it with a TTL,
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

    def log_event(self, stream, event):
        """Append an event (a dict) to the list events:<stream>."""
        self.redis.rpush(f"{self.events_prefix}{stream}", json.dumps(event))

    def get_events(self, stream):
        """The events of events:<stream>, oldest first."""
        raw_events = self.redis.lrange(f"{self.events_prefix}{stream}", 0, -1)
        return [json.loads(raw) for raw in raw_events]

    def clear_events(self, stream):
        """Delete the events of events:<stream>, e.g. before a run."""
        self.redis.delete(f"{self.events_prefix}{stream}")

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
