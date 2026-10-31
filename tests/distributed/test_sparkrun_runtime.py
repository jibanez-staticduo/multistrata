"""Contract tests against installed SparkRun 0.4.1; no Docker or GPU calls."""

import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sparkrun.plugins")
if not hasattr(__import__("sparkrun.plugins", fromlist=["RuntimePlugin"]), "RuntimePlugin"):
    pytest.skip("requires SparkRun 0.4.1 public plugin API", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "packaging/experimental/strata-tp3"
sys.path.insert(0, str(PACKAGE))

from sparkrun.plugins import Recipe, RuntimePlugin
from strata_sparkrun import SPARKRUN_PLUGIN_API_VERSION, StrataNativeRuntime


@pytest.fixture
def recipe():
    import yaml
    return Recipe(yaml.safe_load((PACKAGE / "smoke.yaml").read_text()))


def test_recipe_pins_image_python_and_runtime_falls_back_to_python3(recipe):
    assert recipe.defaults.get("python") == "/usr/bin/python3"
    minimal = Recipe({"model": "strata-transport-smoke", "defaults": {"tensor_parallel": 3}})
    command = StrataNativeRuntime().generate_node_command(minimal, {}, "unused", 3, 0)
    assert shlex.split(command)[0] == "python3"


def test_three_rank_commands_and_no_early_probe(recipe):
    runtime = StrataNativeRuntime()
    assert type(runtime).__bases__ == (RuntimePlugin,)
    assert SPARKRUN_PLUGIN_API_VERSION == 1
    assert runtime.cluster_strategy() == "native"
    assert runtime.native_rendezvous_port(recipe, num_nodes=3, init_port=30123) is None
    parallelism = SimpleNamespace(tensor_parallel=3, pipeline_parallel=1,
                                  data_parallel=1, expert_parallel=1)
    assert runtime.world_size(parallelism, recipe=recipe, cluster=None) == 3
    assert runtime.validate_recipe(recipe) == []
    for rank in range(3):
        parts = shlex.split(runtime.generate_node_command(recipe, {}, "10.71.14.15", 3, rank))
        assert parts == ["/usr/bin/python3", "/experiment/run_transport_smoke.py", "--config",
                         f"/experiment/smoke-config/rank{rank}.json", "--evidence-dir",
                         f"/experiment/smoke-evidence/rank{rank}", "--execute"]
        assert not any("credential" in part or "vllm" in part for part in parts)
    parallelism.pipeline_parallel = 3
    with pytest.raises(ValueError):
        runtime.world_size(parallelism, recipe=recipe, cluster=None)


def test_shell_quoting_and_bad_topology(recipe):
    runtime = StrataNativeRuntime()
    override = {"config_dir": "/experiment/a b'$(touch nope)", "python": "/python weird"}
    parts = shlex.split(runtime.generate_node_command(recipe, override, "unused", 3, 2))
    assert parts[0] == override["python"]
    assert parts[3] == override["config_dir"] + "/rank2.json"
    for overrides in ({"pipeline_parallel": 2}, {"tensor_parallel": 1}, {"mode": "unknown"},
                      {"mode": "distributed"}):
        with pytest.raises(ValueError):
            runtime.generate_node_command(recipe, overrides, "unused", 3, 0)
    for nodes, rank in ((2, 0), (3, 3), (3, -1)):
        with pytest.raises(ValueError):
            runtime.generate_node_command(recipe, {}, "unused", nodes, rank)


def test_explicit_future_distributed_binary(recipe):
    overrides = {"mode": "distributed", "distributed_executable": "/custom/engine",
                 "distributed_args": ["--rank", "{rank}", "--world", "{world_size}"]}
    assert shlex.split(StrataNativeRuntime().generate_node_command(
        recipe, overrides, "unused", 3, 1)) == ["/custom/engine", "--rank", "1", "--world", "3"]


def test_native_lifecycle_delegation(recipe, monkeypatch):
    runtime = StrataNativeRuntime()
    calls = []
    monkeypatch.setattr(runtime, "_run_native_cluster", lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(runtime, "_stop_native_cluster", lambda *a, **kw: calls.append((a, kw)) or 0)
    assert runtime._run_cluster(["dg1", "dg2", "dg3"], "image", recipe=recipe,
                                dry_run=True, cluster_id="test") == 0
    assert calls[0]["dry_run"] is True
    assert calls[0]["cluster_id"] == "test"
    assert runtime._stop_cluster(["dg1", "dg2", "dg3"], "test", dry_run=True) == 0
    assert calls[1][1]["dry_run"] is True


def test_installed_entrypoint_discovery(tmp_path):
    entries = importlib.metadata.entry_points(group="sparkrun.plugins")
    assert any(ep.name == "strata-native" and ep.value == "strata_sparkrun" for ep in entries)
    config = tmp_path / "config.yaml"
    config.write_text("integrations:\n  strata-native: true\n")
    code = """
from sparkrun.application import initialize
from sparkrun.core.bootstrap import get_runtime
context = initialize()
assert get_runtime('strata-native', context.variables).runtime_name == 'strata-native'
"""
    env = dict(os.environ, SPARKRUN_CONFIG_DIR=str(tmp_path), SPARKRUN_NO_TELEMETRY="1",
               SPARKRUN_NO_DEFAULT_PLUGINS="1")
    env.pop("SPARKRUN_NO_INSTALLED_PLUGINS", None)
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60,
                   capture_output=True, text=True)
