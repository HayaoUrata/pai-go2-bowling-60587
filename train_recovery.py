"""Fine-tune a separate PPO standing policy on measured post-kick poses.

The walking actor and the scripted kick trajectory remain separate. This actor
is used only after the kicking foot returns to the floor.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz
from rsl_rl.runners import OnPolicyRunner
from environment import Go2Env

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from go2_train import get_train_cfg


class RecoveryEnv(Go2Env):
    def __init__(self,*args,states,**kwargs):
        self.recovery_states=torch.tensor([s['qpos'] for s in states],device=gs.device,dtype=torch.float32)
        # Recorded lane surface is 2 cm above the training plane.
        self.recovery_states[:,:2]=0
        self.recovery_states[:,2]-=.02
        super().__init__(*args,**kwargs)

    def _reset_idx(self,envs_idx=None):
        super()._reset_idx(envs_idx)
        mask=torch.ones(self.num_envs,device=self.device,dtype=torch.bool) if envs_idx is None else envs_idx
        idx=torch.randint(len(self.recovery_states),(self.num_envs,),device=self.device)
        q=self.recovery_states[idx].clone()
        # Retain regular standing poses in 20% of resets.
        nominal=self.init_qpos.expand_as(q)
        q=torch.where((torch.rand(self.num_envs,device=self.device)<.20)[:,None],nominal,q)
        self.robot.set_qpos(q[mask],envs_idx=mask,zero_velocity=True,skip_forward=True)
        self.base_pos[mask]=q[mask,:3]
        self.base_quat[mask]=q[mask,3:7]
        self.projected_gravity[mask]=transform_by_quat(self.global_gravity,inv_quat(q[mask,3:7]))
        joints=q[:,7:][:,self.motors_dof_idx.long()-6]
        self.dof_pos[mask]=joints[mask]
        self.actions[mask]=(joints[mask]-self.default_dof_pos)/.25
        self.last_actions[mask]=self.actions[mask]
        self.base_euler[mask]=quat_to_xyz(q[mask,3:7],rpy=True,degrees=True)

    def _reward_orientation(self):
        return self.projected_gravity[:,:2].square().sum(1)

    def _reward_angular_motion(self):
        return self.base_ang_vel[:,:2].square().sum(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='recovery_01')
    p.add_argument('--iterations',type=int,default=150)
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--resume',type=Path,default=ROOT/'models/checkpoints/locomotion_0750.pt')
    p.add_argument('--export-prefix',default='recovery')
    p.add_argument('--orientation-penalty',type=float,default=8)
    args=p.parse_args()
    out=ROOT/'logs'/args.run;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'models/config.json').read_text())
    cfg['env'].update(episode_length_s=6,base_init_pos=[0,0,.32],termination_if_roll_greater_than=40,termination_if_pitch_greater_than=40)
    cfg['command'].update(lin_vel_x_range=[0,0],lin_vel_y_range=[0,0],ang_vel_range=[0,0])
    cfg['reward']['reward_scales'].update(orientation=-args.orientation_penalty,angular_motion=-.08,lin_vel_z=-2,base_height=-60,action_rate=-.015,similar_to_default=-.05)
    train=get_train_cfg(args.run)
    train['algorithm'].update(learning_rate=.0003,desired_kl=.005,entropy_coef=.005)
    train['save_interval']=50
    states=json.loads((ROOT/'results/kick_recovery_states.json').read_text())
    (out/'config.json').write_text(json.dumps(dict(**cfg,train=train,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},reset_pose_count=len(states),source_checkpoint=str(args.resume.relative_to(ROOT)) if args.resume.is_absolute() and args.resume.is_relative_to(ROOT) else str(args.resume)),indent=2))
    gs.init(backend=gs.metal,logging_level='warning',seed=2)
    env=RecoveryEnv(args.num_envs,**{f'{k}_cfg':copy.deepcopy(v) for k,v in cfg.items()},states=states)
    runner=OnPolicyRunner(env,train,str(out),device=gs.device)
    # Start from the learned locomotion weights, with a fresh optimizer for this task.
    checkpoint=torch.load(args.resume,map_location=str(gs.device),weights_only=False)
    runner.alg.actor.load_state_dict(checkpoint['actor_state_dict'])
    runner.alg.critic.load_state_dict(checkpoint['critic_state_dict'])
    with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(max=.35)
    runner.export_policy_to_jit(str(out),'recovery_0000.pt')
    for start in range(0,args.iterations,args.checkpoint_every):
        steps=min(args.checkpoint_every,args.iterations-start)
        began=time.perf_counter()
        runner.learn(num_learning_iterations=steps,init_at_random_ep_len=(start==0))
        completed=start+steps;runner.current_learning_iteration=completed
        runner.save(str(out/f'checkpoint_{completed:04d}.pt'))
        runner.export_policy_to_jit(str(ROOT/'models'),f'{args.export_prefix}_{completed:04d}.pt')
        print('RECOVERY_CHECKPOINT',completed,'seconds',round(time.perf_counter()-began,2),flush=True)
    gs.destroy()

if __name__=='__main__':main()
