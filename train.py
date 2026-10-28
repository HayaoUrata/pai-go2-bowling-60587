"""Train Go2 policies and preserve progress videos at every checkpoint."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from go2_train import get_cfgs, get_train_cfg
from environment import Go2Env
from video import record_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--video-every", type=int, default=50)
    parser.add_argument("--push-max", type=float, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variable-commands", action="store_true")
    args = parser.parse_args()
    log_dir = ROOT / "logs" / args.run
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg.update(termination_if_roll_greater_than=30, termination_if_pitch_greater_than=30)
    if args.variable_commands:
        command_cfg.update(lin_vel_x_range=[0.0,0.75],lin_vel_y_range=[-0.18,0.18],ang_vel_range=[-0.6,0.6])
        env_cfg.update(command_mode="mixed",resampling_time_s=2.0)
        reward_cfg["reward_scales"]["tracking_ang_vel"] = 1.0
    cfgs = dict(env=env_cfg, obs=obs_cfg, reward=reward_cfg, command=command_cfg)
    train_cfg = get_train_cfg(args.run)
    train_cfg["save_interval"] = 50
    (log_dir / "config.json").write_text(json.dumps(dict(**cfgs, train=train_cfg, args=vars(args)),
                                                    indent=2, default=str))
    gs.init(backend=gs.metal, precision="32", logging_level="warning", seed=args.seed)
    print(f"DEVICE={gs.device}; BACKEND={gs.backend}; START={time.strftime('%H:%M:%S')}", flush=True)
    env = Go2Env(args.num_envs, **{f"{k}_cfg": copy.deepcopy(v) for k, v in cfgs.items()},
                 push_max=args.push_max)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    if args.resume:
        runner.load(str(args.resume), map_location=str(gs.device))
    start_iteration = runner.current_learning_iteration
    evaluation_cfg = copy.deepcopy(cfgs)
    evaluation_cfg["env"]["episode_length_s"] = 60
    evaluation_cfg["env"].pop("command_mode", None)
    evaluation_cfg["command"].update(lin_vel_x_range=[0.5,0.5],lin_vel_y_range=[0,0],ang_vel_range=[0,0])
    eval_env = Go2Env(1, **{f"{k}_cfg": v for k, v in evaluation_cfg.items()},
                      camera_res=(960, 540), auto_reset=False)
    milestones = list(range(0, args.iterations, args.video_every)) + [args.iterations]
    for completed in milestones:
        policy = runner.get_inference_policy(device=gs.device)
        name = f"{args.run}_iter{start_iteration + completed:04d}"
        record_rollout(eval_env, policy, ROOT / "videos" / f"{name}_flat.mp4",
                       label=name, duration=10, force_n=0)
        if completed > 0:
            record_rollout(eval_env, policy, ROOT / "videos" / f"{name}_push40N.mp4",
                           label=name, duration=12, force_n=40)
        runner.alg.train_mode()
        if completed == args.iterations:
            break
        steps = min(args.video_every, args.iterations - completed)
        started = time.perf_counter()
        runner.learn(num_learning_iterations=steps, init_at_random_ep_len=(completed == 0))
        runner.current_learning_iteration = start_iteration + completed + steps
        runner.save(str(log_dir / f"checkpoint_{runner.current_learning_iteration:04d}.pt"))
        print(f"CHUNK_DONE iterations={runner.current_learning_iteration} seconds={time.perf_counter()-started:.2f}", flush=True)
    runner.export_policy_to_jit(str(log_dir), "policy.pt")
    gs.destroy()


if __name__ == "__main__":
    main()
