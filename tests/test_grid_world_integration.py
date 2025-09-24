import functools
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from twisterl.utils import prepare_algorithm


@functools.lru_cache(maxsize=1)
def _ensure_grid_world_available():
    """Install the external grid_world example if it is not already importable."""
    module_name = "grid_world"
    try:
        importlib.import_module(module_name)
        return module_name
    except ModuleNotFoundError:
        pass

    pytest.importorskip(
        "maturin", reason="Grid World example needs maturin to build the extension"
    )
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "grid_world"
    env = os.environ.copy()
    venv_bin = Path(sys.executable).resolve().parent
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "develop",
            "--release",
        ],
        check=True,
        cwd=str(example_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    importlib.invalidate_caches()
    importlib.import_module(module_name)
    return module_name


def test_grid_world_external_env_works_with_twisterl():
    module_name = _ensure_grid_world_available()

    grid_world_module = importlib.import_module(module_name)
    GridWorld = grid_world_module.GridWorld

    env = GridWorld(4, 4, 10, 2)
    env.reset()
    # Observations & states should match the board size
    state = env.get_state()
    assert len(state) == 16
    assert {0, 1, 2, 3}.issuperset(set(state))

    algo_config = {
        "env_cls": f"{module_name}.GridWorld",
        "policy_cls": "twisterl.nn.policy.BasicPolicy",
        "algorithm_cls": "twisterl.rl.ppo.PPO",
        "env": {"width": 4, "height": 4, "max_steps": 10, "difficulty": 2},
        "policy": {
            "embedding_size": 32,
            "common_layers": [],
            "policy_layers": [],
            "value_layers": [],
            "device": "cpu",
        },
        "algorithm": {
            "device": "cpu",
            "collecting": {
                "num_cores": 1,
                "num_episodes": 1,
                "lambda": 0.95,
                "gamma": 0.95,
            },
            "training": {
                "num_epochs": 1,
                "vf_coef": 0.8,
                "ent_coef": 0.0,
                "clip_ratio": 0.1,
                "normalize_advantage": False,
            },
            "optimizer": {"lr": 0.001},
            "logging": {"log_freq": 0, "checkpoint_freq": 0},
            "learning": {
                "diff_threshold": 0.9,
                "diff_metric": "ppo_deterministic",
                "diff_max": 3,
            },
        },
    }

    algo = prepare_algorithm(algo_config)

    # Ensure the basic interactions work as expected
    algo.env.reset()
    observed = algo.env.observe()
    assert len(observed) == algo.env.obs_shape()[0]

    collected, _ = algo.collect()
    assert collected.obs
    assert collected.additional_data["rets"]
