"""Model placement (step 3, P2): the placement program, the controller that
plans and publishes it, nodes following their target placement, shadow
demand, and the prior from GreenServ's offline data.

Run from the repository root: ./venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.placement_prior import model_shares, read_tables
from src.bandit.linucb import LinUCB
from src.services.allocation_controller import AllocationController, solve_placement
from src.services.orchestrator import Orchestrator
from tests.test_load_time import make_orchestrator, node
from tests.worker_fixture import MODELS, FakeWorkerTestCase


def plan(demand, nodes=("e1",), budget=20.0, current=(), horizon=300.0, latency=1.0):
    """Solve for MODELS (small 2 GB, mid 4 GB, big 16 GB; loads take 1, 2 and
    8 s) on identical edge nodes."""
    models = list(MODELS)
    load = {"small": 1.0, "mid": 2.0, "big": 8.0}
    size = {"small": 2.0, "mid": 4.0, "big": 16.0}
    pairs = [(m, n) for m in models for n in nodes]
    return solve_placement(
        models,
        list(nodes),
        demand,
        latency={p: latency for p in pairs},
        load_time={(m, n): load[m] for m, n in pairs},
        size_gb={(m, n): size[m] for m, n in pairs},
        budget_gb={n: budget for n in nodes},
        eligible={p: True for p in pairs},
        current={p: p in current for p in pairs},
        planning_horizon_s=horizon,
    )


class SolvePlacementTest(unittest.TestCase):
    def test_fills_the_memory_budget_by_demand(self):
        placement, weights = plan({"big": 0.05, "mid": 0.02, "small": 0.01})
        # 16 + 4 GB fill the 20 GB budget; small no longer fits.
        self.assertEqual(placement["e1"], ["big", "mid"])
        self.assertEqual(weights["big"], [("e1", 1.0)])
        self.assertNotIn("small", weights)

    def test_loads_a_model_once_a_request_is_expected_within_the_horizon(self):
        # One request every 200 s pays off within 300 s; one every 500 s not.
        self.assertEqual(plan({"mid": 1 / 200})[0]["e1"], ["mid"])
        self.assertEqual(plan({"mid": 1 / 500})[0]["e1"], [])

    def test_keeps_the_current_placement_unless_a_change_pays_off(self):
        # Room for one 16 GB model; big2 is a copy of big with its own demand.
        models = ["big", "big2"]
        pairs = [(m, "e1") for m in models]

        def solve(demand):
            return solve_placement(
                models, ["e1"], demand,
                latency={p: 1.0 for p in pairs},
                load_time={p: 30.0 for p in pairs},
                size_gb={p: 16.0 for p in pairs},
                budget_gb={"e1": 20.0},
                eligible={p: True for p in pairs},
                current={("big", "e1"): True, ("big2", "e1"): False},
            )[0]["e1"]

        self.assertEqual(solve({"big": 0.010, "big2": 0.012}), ["big"])
        self.assertEqual(solve({"big": 0.010, "big2": 0.050}), ["big2"])

    def test_adds_a_replica_when_one_node_is_not_enough(self):
        placement, weights = plan({"small": 1.5}, nodes=("e1", "e2"))
        self.assertIn("small", placement["e1"])
        self.assertIn("small", placement["e2"])
        shares = dict(weights["small"])
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        for share in shares.values():
            self.assertLessEqual(1.5 * share, 0.85 + 1e-6)

    def test_nothing_eligible(self):
        placement, weights = solve_placement(
            ["small"], ["c1"], {"small": 1.0}, {}, {}, {}, {"c1": 20.0}, {}, {}
        )
        self.assertEqual(placement, {"c1": []})
        self.assertEqual(weights, {})


def edge_node(node_id="e1", **fields):
    telemetry = node(node_id=node_id, queue_name=f"q_{node_id}", resident_models=[])
    telemetry["memory_budget_gb"] = 20.0
    telemetry.update(fields)
    return telemetry


class ControllerTest(unittest.TestCase):
    def run_controller(self, ticks, telemetry, demand=None, **kwargs):
        """Build a controller with a fake queue client and run `ticks` ticks;
        returns (controller, queue client)."""
        queue = mock.MagicMock()
        queue.get_all_node_telemetry.return_value = telemetry

        async def scenario():
            controller = AllocationController(
                queue_client=queue,
                model_configs=MODELS,
                demand_provider=lambda: dict(demand or {}),
                **kwargs,
            )
            for _ in range(ticks):
                await controller._tick()
            return controller

        return asyncio.run(scenario()), queue

    def test_first_plan_from_the_prior_is_published(self):
        controller, queue = self.run_controller(
            1, [edge_node()], prior_demand={"big": 0.05, "mid": 0.02}
        )
        # Published with an expiry: 3 ticks of 5 s plus 1 s.
        queue.set_node_placement.assert_called_once_with(
            "e1", ["big", "mid"], ttl_seconds=16
        )
        self.assertTrue(controller.has_eligibility())
        self.assertEqual(controller.eligibility_for("big"), [("e1", 1.0)])

    def test_plan_is_renewed_every_tick(self):
        _, queue = self.run_controller(3, [edge_node()], prior_demand={"big": 0.05})
        self.assertEqual(queue.set_node_placement.call_count, 3)
        for call in queue.set_node_placement.call_args_list:
            self.assertEqual(call, mock.call("e1", ["big"], ttl_seconds=16))

    def test_static_policy_plans_only_once(self):
        with mock.patch(
            "src.services.allocation_controller.solve_placement", wraps=solve_placement
        ) as solver:
            _, queue = self.run_controller(
                3, [edge_node()], demand={"big": 5.0}, policy="static", T_cool=0.0
            )
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(queue.set_node_placement.call_count, 3)  # renewed

    def test_placement_rule_fires_once_loading_can_pay_off(self):
        # The prior places nothing. Demand for mid then stays far below
        # R_min (0.1 req/s), but above 1/T_plan, so mid is placed.
        controller, queue = self.run_controller(
            2, [edge_node()], prior_demand={"small": 0.0}, demand={"mid": 0.05}, T_cool=0.0
        )
        self.assertEqual(queue.set_node_placement.call_args_list[0][0][1], [])
        self.assertEqual(queue.set_node_placement.call_args[0][1], ["mid"])
        self.assertEqual(controller.eligibility_for("mid"), [("e1", 1.0)])

    def test_never_places_a_model_on_a_node_too_small_for_it(self):
        _, queue = self.run_controller(
            1, [edge_node(memory_budget_gb=10.0)], prior_demand={"big": 1.0}
        )
        queue.set_node_placement.assert_called_once_with("e1", [], ttl_seconds=16)


class WorkerTargetTest(FakeWorkerTestCase):
    def reconcile(self, target):
        self.redis.get_node_placement.return_value = target
        self.worker._last_reconcile = 0.0
        self.worker._reconcile_placement()
        return self.worker._model_state_snapshot()["resident_models"]

    def test_loads_one_missing_target_model_per_call(self):
        self.make_worker(memory_budget_gb=20.0)
        self.assertEqual(self.reconcile(["mid", "small"]), ["mid"])
        self.assertEqual(self.reconcile(["mid", "small"]), ["mid", "small"])

    def test_target_load_never_unloads_other_target_models(self):
        self.make_worker(memory_budget_gb=18.0)
        self.reconcile(["big"])
        # mid (4 GB) does not fit next to big (16 GB) within 18 GB.
        self.assertEqual(self.reconcile(["big", "mid"]), ["big"])

    def test_target_load_unloads_models_outside_the_target_first(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        worker._load_model("small")
        self.reconcile(["big"])
        # 2 + 16 + 4 > 20: small, outside the target, makes room for mid.
        self.assertEqual(self.reconcile(["big", "mid"]), ["big", "mid"])

    def test_task_load_unloads_target_models_only_as_last_resort(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        worker._load_model("small")
        self.reconcile(["big"])
        worker._load_model("mid")  # a task: small goes first, big can stay
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["big", "mid"])
        worker.memory_budget_gb = 17.0
        # Unloading mid is not enough (16 + 2 > 17), so big goes as well.
        worker._load_model("small")
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["small"])

    def test_target_models_load_only_while_the_queue_is_empty(self):
        self.make_worker(memory_budget_gb=20.0)
        self.redis.get_queue_length.return_value = 3
        self.assertEqual(self.reconcile(["mid"]), [])
        self.assertEqual(self.worker._model_state_snapshot()["target_models"], ["mid"])
        self.redis.get_queue_length.return_value = 0
        self.assertEqual(self.reconcile(["mid"]), ["mid"])

    def test_node_falls_back_to_lru_when_its_target_expires(self):
        worker = self.make_worker(memory_budget_gb=20.0)
        self.reconcile(["mid"])
        worker._load_model("small")  # a task; mid is now least recently used
        self.reconcile(None)  # the target expired
        self.assertIsNone(worker._model_state_snapshot()["target_models"])
        # Plain LRU: mid goes first. With mid still a target, small would go.
        worker._load_model("big")
        self.assertEqual(worker._model_state_snapshot()["resident_models"], ["small", "big"])

    def test_without_a_target_the_node_loads_on_demand_only(self):
        self.make_worker(memory_budget_gb=20.0)
        self.assertEqual(self.reconcile(None), [])
        self.assertIsNone(self.worker._model_state_snapshot()["target_models"])


class PlacementKeyTest(unittest.TestCase):
    def test_target_placement_is_written_with_an_expiry(self):
        from src.queue.redis_client import RedisClient

        client = RedisClient.__new__(RedisClient)  # no connection attempt
        client.redis = mock.MagicMock()
        client.placement_prefix = "placement:node:"
        client.set_node_placement("e1", ["mid"], ttl_seconds=16)
        client.redis.set.assert_called_once_with("placement:node:e1", '["mid"]', ex=16)


class InMemoryStore:
    """The Redis keys the controller, the workers and the orchestrator share,
    with values passed through JSON as Redis would."""

    def __init__(self):
        self.telemetry, self.placement = {}, {}

    def set_node_telemetry(self, node_id, payload, ttl_seconds=None):
        self.telemetry[node_id] = json.loads(json.dumps(payload))

    def get_all_node_telemetry(self):
        return list(self.telemetry.values())

    def set_node_placement(self, node_id, model_ids, ttl_seconds=None):
        self.placement[node_id] = json.loads(json.dumps(list(model_ids)))

    def get_node_placement(self, node_id):
        return self.placement.get(node_id)

    def get_queue_length(self, queue_name):
        return 0


class EndToEndPlacementTest(FakeWorkerTestCase):
    def test_plan_is_loaded_and_used_for_routing(self):
        store = InMemoryStore()
        worker = self.make_worker(redis_client=store, memory_budget_gb=20.0)
        worker._publish_telemetry()

        async def scenario():
            controller = AllocationController(
                queue_client=store,
                model_configs=MODELS,
                demand_provider=dict,
                prior_demand={"big": 0.05, "mid": 0.02},
            )
            await controller._tick()
            return controller

        controller = asyncio.run(scenario())
        self.assertEqual(store.placement["e1"], ["big", "mid"])
        for _ in range(2):  # one load per reconcile
            worker._last_reconcile = 0.0
            worker._reconcile_placement()
        worker._publish_telemetry()
        telemetry = store.telemetry["e1"]
        self.assertEqual(telemetry["resident_models"], ["big", "mid"])
        self.assertEqual(telemetry["target_models"], ["big", "mid"])

        orch = make_orchestrator()
        orch.allocation_controller = controller
        surviving = orch._filter_nodes(store.get_all_node_telemetry())
        self.assertEqual(orch._pick_queue_for_model("big", surviving), worker.queue_name)
        self.assertEqual(orch._pending_load_time(surviving[0], "big"), 0.0)


class StubController:
    def __init__(self, weights):
        self.weights = weights

    def has_eligibility(self):
        return True

    def eligibility_for(self, model_id):
        return self.weights.get(model_id, [])


class OrchestratorPlacementTest(unittest.TestCase):
    def nodes(self):
        return [
            node(node_id="a", queue_name="qa", resident_models=["big"]),
            node(node_id="b", queue_name="qb", resident_models=["big"]),
        ]

    def test_replicas_are_used_when_they_meet_the_deadline(self):
        orch = make_orchestrator()
        orch.allocation_controller = StubController({"big": [("a", 1.0)]})
        options = orch._node_options_for_model("big", self.nodes())
        self.assertEqual([n["node_id"] for n, _ in options], ["a"])

    def test_other_nodes_serve_as_overflow(self):
        orch = make_orchestrator()
        orch.allocation_controller = StubController({"big": [("a", 1.0)]})
        busy_a, b = self.nodes()
        busy_a["queue_length"] = 100
        options = orch._node_options_for_model("big", [busy_a, b])
        self.assertEqual([n["node_id"] for n, _ in options], ["b"])

    def test_model_without_replica_is_not_restricted(self):
        orch = make_orchestrator()
        orch.allocation_controller = StubController({})
        self.assertEqual(len(orch._node_options_for_model("big", self.nodes())), 2)

    def test_shadow_demand_counts_the_unmasked_choice(self):
        router = mock.MagicMock()
        router.select_model.return_value = "small"
        router.preferred_model.return_value = "big"
        queue = mock.MagicMock()
        queue.get_all_node_telemetry.return_value = []
        orch = Orchestrator(
            feature_service=mock.MagicMock(),
            router_service=router,
            evaluation_service=mock.MagicMock(),
            queue_client=queue,
            model_configs=MODELS,
        )
        asyncio.run(orch.process_query("q", wait_for_result=False))
        self.assertEqual(list(orch.consume_rps_window()), ["small"])
        self.assertEqual(list(orch.consume_shadow_window()), ["big"])
        self.assertEqual(orch.consume_shadow_window(), {})  # reset after reading


class PreferredModelTest(unittest.TestCase):
    def test_preferred_model_ignores_the_mask(self):
        bandit = LinUCB(["a", "b"], context_dimension=2, alpha=0.1, regularization=1.0)
        for _ in range(20):
            bandit.update("a", 1.0, context=[1.0, 0.0])
        chosen = bandit.select_model([1.0, 0.0], available_models=["b"])
        self.assertEqual(chosen, "b")
        self.assertEqual(bandit.last_preferred_model, "a")

    def test_ties_are_broken_at_random(self):
        bandit = LinUCB(["a", "b", "c"], context_dimension=2, alpha=1.0, regularization=1.0, seed=0)
        preferred = set()
        for _ in range(50):
            bandit.select_model([1.0, 0.0])
            preferred.add(bandit.last_preferred_model)
        self.assertEqual(preferred, {"a", "b", "c"})


class PlacementPriorTest(unittest.TestCase):
    def test_reads_copy_blocks_of_the_dump(self):
        dump = (
            "COPY public.queries (query_id, text, dataset) FROM stdin;\n"
            "1\tWhat?\tmmlu\n"
            "\\.\n"
            "COPY public.other (x) FROM stdin;\n"
            "9\n"
            "\\.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.sql.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(dump)
            tables = read_tables(path, {"queries"})
        self.assertEqual(tables["queries"], [{"query_id": "1", "text": "What?", "dataset": "mmlu"}])

    def test_counts_which_model_gives_the_highest_reward(self):
        queries = [{"query_id": "1", "dataset": "mmlu"}, {"query_id": "2", "dataset": "mmlu"}]

        def row(query, model, accuracy):
            return {
                "query_id": query, "model_id": model, "accuracy": str(accuracy),
                "energy_consumption": "1.0", "input_tokens": "10", "output_tokens": "10",
            }

        results = [
            row("1", "small", 1.0), row("1", "phi-4-14b", 0.0),
            row("2", "small", 0.0), row("2", "phi-4-14b", 1.0),
        ]
        # GreenServ's id phi-4-14b maps to phi-4 in config/models.yaml.
        shares = model_shares(results, queries, {"small", "phi-4"})
        self.assertEqual(shares, {"small": 0.5, "phi-4": 0.5})


if __name__ == "__main__":
    unittest.main()
