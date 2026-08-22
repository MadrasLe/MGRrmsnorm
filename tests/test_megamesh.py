"""Unit checks for the isolated MegaMesh layer."""

from __future__ import annotations

import megagemm.mesh.router as router_module
from megagemm.mesh import MeshHTTPError, MeshRouter, plan_agron_layer_stages, plan_layer_stages


def test_megamesh_public_imports_work():
    assert MeshRouter.__name__ == "MeshRouter"


def test_plan_layer_stages_weighted_contiguous_ranges():
    plan = plan_layer_stages(
        10,
        "http://worker-a:8088@1#a,http://worker-b:8088@3#b",
        devices=["cuda:0", "cuda:1"],
    )

    assert plan == [
        {
            "stage_id": 0,
            "layer_start": 0,
            "layer_end": 3,
            "num_layers": 3,
            "worker_url": "http://worker-a:8088",
            "name": "a",
            "weight": 1.0,
            "device": "cuda:0",
        },
        {
            "stage_id": 1,
            "layer_start": 3,
            "layer_end": 10,
            "num_layers": 7,
            "worker_url": "http://worker-b:8088",
            "name": "b",
            "weight": 3.0,
            "device": "cuda:1",
        },
    ]


def test_agron_plan_can_weight_layers_and_reorder_mesh_path():
    profile = {
        "nodes": [
            {"name": "slow", "speed": 1.0},
            {"name": "fast", "speed": 3.0},
            {"name": "mid", "speed": 2.0},
        ],
        "links": [
            {"src": "slow", "dst": "fast", "latency_ms": 50.0},
            {"src": "slow", "dst": "mid", "latency_ms": 1.0},
            {"src": "mid", "dst": "fast", "latency_ms": 1.0},
            {"src": "fast", "dst": "mid", "latency_ms": 1.0},
            {"src": "mid", "dst": "slow", "latency_ms": 1.0},
        ],
    }
    plan = plan_agron_layer_stages(
        12,
        "ttp://slow:9090@1#slow,ttp://fast:9091@1#fast,ttp://mid:9092@1#mid",
        node_profiles=profile["nodes"],
        link_profiles=profile["links"],
        objective="latency",
        allow_reorder=True,
        max_candidates=1000,
    )

    assert plan["planner"] == "agron-v0"
    assert [stage["name"] for stage in plan["stages"]] != ["slow", "fast", "mid"]
    assert sum(stage["num_layers"] for stage in plan["stages"]) == 12
    fast_stage = next(stage for stage in plan["stages"] if stage["name"] == "fast")
    slow_stage = next(stage for stage in plan["stages"] if stage["name"] == "slow")
    assert fast_stage["num_layers"] >= slow_stage["num_layers"]


def test_router_failover_reassigns_failed_bucket():
    old_request_json = router_module.request_json

    def fake_request_json(endpoint, path, payload=None, timeout=120.0):
        if endpoint.name == "bad":
            raise RuntimeError("worker unavailable")
        prompts = payload["prompts"]
        return {
            "ok": True,
            "worker": endpoint.name,
            "outputs": [f"{prompt}:ok" for prompt in prompts],
            "elapsed_ms": 1.0,
            "generated_tokens": len(prompts),
            "tokens_per_second": float(len(prompts)),
        }

    try:
        router_module.request_json = fake_request_json
        router = MeshRouter("bad-host:8088@1#bad,good-host:8088@1#good")
        outputs, stats = router.generate_batch_with_stats(
            ["a", "b", "c"],
            max_new_tokens=1,
            failover=True,
        )
    finally:
        router_module.request_json = old_request_json

    assert outputs == ["a:ok", "b:ok", "c:ok"]
    assert len(stats["failures"]) == 1
    assert stats["workers"]


def test_router_can_fail_fast_without_failover():
    old_request_json = router_module.request_json

    def fake_request_json(endpoint, path, payload=None, timeout=120.0):
        if endpoint.name == "bad":
            raise RuntimeError("worker unavailable")
        prompts = payload["prompts"]
        return {
            "ok": True,
            "worker": endpoint.name,
            "outputs": [f"{prompt}:ok" for prompt in prompts],
        }

    try:
        router_module.request_json = fake_request_json
        router = MeshRouter("bad-host:8088@1#bad,good-host:8088@1#good")
        try:
            router.generate_batch_with_stats(["a", "b"], failover=False)
        except MeshHTTPError as exc:
            assert "did not receive every output" in str(exc)
        else:
            raise AssertionError("expected MeshHTTPError")
    finally:
        router_module.request_json = old_request_json


if __name__ == "__main__":
    test_megamesh_public_imports_work()
    test_plan_layer_stages_weighted_contiguous_ranges()
    test_agron_plan_can_weight_layers_and_reorder_mesh_path()
    test_router_failover_reassigns_failed_bucket()
    test_router_can_fail_fast_without_failover()
    print("MegaMesh tests passed")
