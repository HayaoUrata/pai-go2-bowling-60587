# Third-party sources

The Go2 PPO environment and baseline hyperparameters are adapted from Genesis World's
official locomotion example (retrieved 2026-10-28):

- https://github.com/Genesis-Embodied-AI/genesis-world/tree/main/examples/locomotion
- Original copies: `vendor/go2_env.py`, `vendor/go2_train.py`.
- Apache-2.0 license: `vendor/GENESIS_LICENSE`.

Our additions are time-limited lateral forces during training, variable speed/stop/yaw
commands, automatic progress videos, matched-condition evaluation, and the kick-bowling
task. The kick uses original analytic leg IK, a four-leg weight-shift trajectory,
and PD position control. It is not an end-to-end RL-trained bowling policy.
The six-pin asset, task sequencing, score evaluation, video editing, and report
generation were added for this project.

The Unitree Go2 URDF and meshes are loaded unchanged from Genesis's bundled assets.
PPO is provided by rsl-rl (BSD-3-Clause): https://github.com/leggedrobotics/rsl_rl.

A separate PPO standing-recovery actor is fine-tuned from the locomotion actor on
recorded post-kick poses. The camera-matched stage comparisons and gradual
controller handover are additional project-specific modifications.
