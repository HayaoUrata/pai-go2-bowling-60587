"""Preserve a genuine zero-update neural policy for the before/after comparison."""

import copy
import json
from pathlib import Path

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from environment import Go2Env

ROOT = Path(__file__).resolve().parent


def main():
    cfg = json.loads((ROOT / "logs" / "baseline_01" / "config.json").read_text())
    target = ROOT / "logs" / "untrained_01"
    target.mkdir(exist_ok=False)
    cfg["args"].update(run="untrained_01", iterations=0, resume=None)
    cfg["train"]["run_name"] = "untrained_01"
    (target / "config.json").write_text(json.dumps(cfg, indent=2))
    gs.init(backend=gs.metal, logging_level="warning", seed=1)
    env = Go2Env(1024, copy.deepcopy(cfg["env"]), copy.deepcopy(cfg["obs"]),
                 copy.deepcopy(cfg["reward"]), copy.deepcopy(cfg["command"]))
    runner = OnPolicyRunner(env, cfg["train"], log_dir=None, device=gs.device)
    runner.save(str(target / "checkpoint_0000.pt"))
    runner.export_policy_to_jit(str(target), "policy.pt")
    print("UNTRAINED_POLICY_SAVED: zero optimizer updates", flush=True)
    gs.destroy()


if __name__ == "__main__":
    main()
