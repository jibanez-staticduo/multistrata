"""Native SparkRun lifecycle for the experimental three-rank Strata smoke."""

from __future__ import annotations

import logging
import shlex

from sparkrun.plugins import RuntimePlugin

logger = logging.getLogger(__name__)


class StrataNativeRuntime(RuntimePlugin):
    runtime_name = "strata-native"
    requires_capability = frozenset({"gb10"})

    def cluster_strategy(self):
        return "native"

    def native_apis(self, recipe):
        return []

    def native_rendezvous_port(self, recipe, overrides=None, *, num_nodes=1, init_port=25000):
        # A bare TCP probe would consume the authenticated smoke connection.
        # Workers already retry the private bootstrap address in their config.
        return None

    def default_executor_config(self):
        return {"entrypoint": ""}

    def known_config_keys(self):
        return frozenset({"mode", "python", "smoke_runner", "config_dir", "evidence_dir",
                          "distributed_executable", "distributed_args"})

    @staticmethod
    def _config(recipe, overrides):
        config = recipe.build_config_chain(overrides or {})
        for key, expected in (("tensor_parallel", 3), ("pipeline_parallel", 1),
                              ("data_parallel", 1), ("expert_parallel", 1), ("ep_size", 1)):
            if int(config.get(key, expected)) != expected:
                raise ValueError("strata-native requires TP=3, PP=1, DP=1, EP=1")
        if getattr(recipe, "command", None):
            raise ValueError("strata-native uses generated rank commands; remove command")
        return config

    def world_size(self, parallelism, *, recipe, cluster):
        if (parallelism.tensor_parallel != 3 or parallelism.pipeline_parallel != 1
                or parallelism.data_parallel != 1 or parallelism.expert_parallel != 1):
            raise ValueError("strata-native requires TP=3, PP=1, DP=1, EP=1")
        return 3

    def validate_recipe(self, recipe):
        issues = super().validate_recipe(recipe)
        try:
            self.generate_node_command(recipe, {}, "10.71.14.15", 3, 0)
        except (ValueError, TypeError) as exc:
            issues.append(self.recipe_error(str(exc)))
        return issues

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1,
                         head_ip=None, skip_keys=frozenset()):
        if not is_cluster:
            raise ValueError("strata-native requires a three-host cluster")
        return self.generate_node_command(recipe, overrides, head_ip, num_nodes, 0,
                                          skip_keys=skip_keys)

    def generate_node_command(self, recipe, overrides, head_ip, num_nodes, node_rank,
                              init_port=25000, skip_keys=frozenset(), hosts=None,
                              placement=None):
        if num_nodes != 3 or type(node_rank) is not int or node_rank not in range(3):
            raise ValueError("strata-native requires three nodes and rank 0, 1 or 2")
        config = self._config(recipe, overrides)
        mode = config.get("mode", "smoke")
        values = {"rank": node_rank, "world_size": 3}
        if mode == "smoke":
            config_dir = str(config.get("config_dir", "/experiment/smoke-config")).rstrip("/")
            evidence_dir = str(config.get("evidence_dir", "/experiment/smoke-evidence")).rstrip("/")
            parts = [str(config.get("python", "python")),
                     str(config.get("smoke_runner", "/experiment/run_transport_smoke.py")),
                     "--config", f"{config_dir}/rank{node_rank}.json",
                     "--evidence-dir", f"{evidence_dir}/rank{node_rank}", "--execute"]
        elif mode == "distributed":
            # No distributed server binary is shipped or assumed by this plugin.
            executable = config.get("distributed_executable")
            args = config.get("distributed_args")
            if not isinstance(executable, str) or not executable.startswith("/"):
                raise ValueError("distributed mode requires an explicit absolute distributed_executable")
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise ValueError("distributed mode requires distributed_args as an argv list")
            parts = [executable, *(arg.format_map(values) for arg in args)]
        else:
            raise ValueError("mode must be smoke or distributed")
        if any("\x00" in part or "\n" in part or "\r" in part for part in parts):
            raise ValueError("command arguments must be single-line strings")
        command = shlex.join(parts)
        logger.info("Strata native rank %d command: %s", node_rank, command)
        return command

    def _run_cluster(self, hosts, image, serve_command="", recipe=None, overrides=None,
                     **kwargs):
        if len(hosts) != 3:
            raise ValueError("strata-native requires exactly three hosts")
        return self._run_native_cluster(
            hosts=hosts, image=image, serve_command=serve_command, recipe=recipe,
            overrides=overrides, banner_title="Strata Native TP3", node_label="Strata rank",
            **kwargs)

    def _stop_cluster(self, hosts, cluster_id, config=None, dry_run=False):
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)
