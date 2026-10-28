"""Bowling with the unmodified Go2: PPO approach/stop, analytic leg IK, PD kick."""

import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path

import genesis as gs
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image,ImageDraw

from environment import Go2Env
from render_utils import font

ROOT=Path(__file__).resolve().parent


def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def leg_ik(foot_body,leg="FR"):
    # Right-front leg dimensions and axes are taken from the bundled Go2 URDF.
    side=1 if leg.endswith("L") else -1
    front=1 if leg.startswith("F") else -1
    px,py,pz=foot_body-np.array([front*0.1934,side*0.0465,0.0])
    lateral=0.0955;length=0.213
    radius=max(float(np.hypot(py,pz)),lateral+1e-6)
    hip=np.arctan2(pz,py)+np.arccos(np.clip(side*lateral/radius,-1,1))
    hip=float(np.clip(hip,-1.0472,1.0472))
    leg_z=-np.sin(hip)*py+np.cos(hip)*pz
    cosine=(px*px+leg_z*leg_z-2*length*length)/(2*length*length)
    calf=float(np.clip(-np.arccos(np.clip(cosine,-1,1)),-2.70,-0.85))
    thigh=np.arctan2(-px,-leg_z)-np.arctan2(length*np.sin(calf),length+length*np.cos(calf))
    return np.array([hip,np.clip(thigh,-1.55,3.45),calf],dtype=np.float32)


def ease(a,b,u):
    u=float(np.clip(u,0,1));u=u*u*(3-2*u)
    return a+(b-a)*u


@torch.inference_mode()
def play(env,network,name,mass,aim_y,swing_time,ball_y=0.0,fps=25,wide=False,side=False,stage="complete",recovery=None,handover=2.81,save_states=False,recovery_name=None,dual_view=False,ball_x=.85,detector=None,record=True,hold_extra=0.0):
    env.scene.reset();obs=env.reset()
    env.ball.set_pos(torch.tensor([[ball_x,ball_y,0.14]],device=gs.device))
    env.ball.set_mass(mass)
    # In vision mode the controller gets only this RGB-D estimate, never true ball coordinates.
    detection=detector.detect(name) if detector is not None else None
    estimated_ball=np.asarray(detection["position"],dtype=np.float32) if detection is not None else None
    knocked=np.zeros(6,dtype=bool);samples=[];recovery_states=[]
    settle_start=None;kick_start=None;foot_start=None;ball_start=None;approach_failed=False
    phases=[];kick_peak_speed=0.0;stiff=False;support=None;support_base=None;support_rotation=None
    calf=env.robot.get_link("FR_calf")
    out=ROOT/"videos"/f"kick_{name}.mp4"
    with ExitStack() as stack:
        writer=stack.enter_context(imageio.get_writer(str(out),fps=fps,codec="libx264",quality=8,macro_block_size=1)) if record else None
        side_out=out.with_name(out.stem.removesuffix("_wide")+"_side.mp4") if dual_view and record else None
        side_writer=stack.enter_context(imageio.get_writer(str(side_out),fps=fps,codec="libx264",quality=8,macro_block_size=1)) if dual_view and record else None
        for step in range(800):
            t=step*0.02
            pos=env.robot.get_pos()[0].cpu().numpy()
            quat=env.robot.get_quat()[0].cpu().numpy()
            ball=estimated_ball if estimated_ball is not None else env.ball.get_pos()[0].cpu().numpy()
            yaw=np.arctan2(rotation(quat)[1,0],rotation(quat)[0,0])
            target=estimated_ball[:2]+np.array([-.43,.142]) if estimated_ball is not None else np.array([0.42,ball_y+0.142])
            error=target-pos[:2]
            if settle_start is None and not approach_failed:
                phase="接近"
                vx=float(np.clip(2.0*error[0],0,0.55))
                vy=float(np.clip(2.0*error[1],-.18,.18))
                commands=[np.cos(yaw)*vx+np.sin(yaw)*vy,-np.sin(yaw)*vx+np.cos(yaw)*vy,float(np.clip(-2*yaw,-.6,.6))]
                if np.linalg.norm(error)<0.065:
                    settle_start=t;commands=[0,0,0]
                elif t>6:
                    approach_failed=True;commands=[0,0,0];phase="接近できず終了"
            else:
                commands=[0,0,0];phase="接近できず終了" if approach_failed else "停止"
            env.commands[0,:]=torch.tensor(commands,dtype=torch.float32,device=gs.device)
            env._update_observation();actions=network(env.get_observations()["policy"])
            if stage!="walk-only" and settle_start is not None and t-settle_start>=0.70 and kick_start is None:
                kick_start=t
                cp=calf.get_pos()[0].cpu().numpy();cq=calf.get_quat()[0].cpu().numpy()
                foot_start=cp+rotation(cq)@np.array([0,0,-0.213])
                ball_start=ball.copy()
                support_base=pos.copy();support_rotation=rotation(quat)
                support={}
                for leg in ["FR","FL","RR","RL"]:
                    link=env.robot.get_link(f"{leg}_calf")
                    support[leg]=link.get_pos()[0].cpu().numpy()+rotation(link.get_quat()[0].cpu().numpy())@np.array([0,0,-.213])
                if stage=="complete":
                    env.robot.set_dofs_kp([100.0]*3+[80.0]*9,env.motors_dof_idx)
                    env.robot.set_dofs_kv([2.0]*3+[3.0]*9,env.motors_dof_idx)
                    stiff=True
                print("KICK_START",name,t,"base",pos.tolist(),"ball",ball.tolist(),flush=True)
            if stage=="unstable" and kick_start is not None:
                # Historical prototype: swing only FR, leaving support to walking policy.
                # No whole-body weight shift or dedicated three-leg support control.
                dt=t-kick_start
                direction=np.array([3.10,aim_y])-ball_start[:2]
                direction/=np.linalg.norm(direction)
                load=ball_start.copy();load[:2]-=.34*direction;load[2]=.14
                follow=ball_start.copy();follow[:2]-=.005*direction;follow[2]=.14
                if dt<.24:foot=ease(foot_start,load,dt/.24);phase="構える"
                elif dt<.32:foot=load;phase="構える"
                elif dt<.32+swing_time:foot=ease(load,follow,(dt-.32)/swing_time);phase="キック"
                elif dt<.40+swing_time:foot=follow;phase="キック"
                elif dt<.72+swing_time:foot=ease(follow,foot_start,(dt-.40-swing_time)/.32);phase="足を戻す"
                else:foot=None;phase="結果"
                if foot is not None:
                    joint=leg_ik(rotation(quat).T@(foot-pos))
                    actions[0,:3]=(torch.as_tensor(joint,device=gs.device)-env.default_dof_pos[:3])/.25
            elif kick_start is not None:
                dt=t-kick_start
                direction=np.array([3.10,aim_y])-ball_start[:2]
                direction/=np.linalg.norm(direction)
                load=ball_start.copy();load[:2]-=0.34*direction;load[2]=0.14
                follow=ball_start.copy();follow[:2]-=0.005*direction;follow[2]=0.14
                shift=np.array([-.060,.120,-.015])
                total=2.65+swing_time+hold_extra
                if dt<.65:
                    desired_base=ease(support_base,support_base+shift,dt/.65)
                elif dt<1.55+hold_extra+swing_time:
                    desired_base=support_base+shift
                else:
                    desired_base=ease(support_base+shift,support_base,(dt-1.55-hold_extra-swing_time)/.90)
                if dt<.65:
                    foot=foot_start;phase="体重を移す"
                elif dt<.87:
                    foot=ease(foot_start,load,(dt-.65)/.22);phase="構える"
                elif dt<.91+hold_extra:
                    foot=load;phase="構える"
                elif dt<.91+hold_extra+swing_time:
                    foot=ease(load,follow,(dt-.91-hold_extra)/swing_time);phase="キック"
                elif dt<.97+hold_extra+swing_time:
                    foot=follow;phase="キック"
                elif dt<1.32+hold_extra+swing_time:
                    foot=ease(follow,foot_start,(dt-.97-hold_extra-swing_time)/.35);phase="足を戻す"
                elif dt<total:
                    foot=foot_start;phase="姿勢を戻す"
                else:
                    foot=None;phase="結果"
                if foot is not None:
                    for i,leg in enumerate(["FR","FL","RR","RL"]):
                        target_foot=foot if leg=="FR" else support[leg]
                        if leg=="FR" and dt>=.65:
                            # Compensate the measured base motion during the fast swing.
                            body=rotation(quat).T@(target_foot-pos)
                        else:
                            body=support_rotation.T@(target_foot-desired_base)
                        joint=leg_ik(body,leg)
                        actions[0,i*3:i*3+3]=(torch.as_tensor(joint,device=gs.device)-env.default_dof_pos[i*3:i*3+3])/.25
                elif stiff:
                    env.robot.set_dofs_kp([20.0]*12,env.motors_dof_idx)
                    env.robot.set_dofs_kv([.5]*12,env.motors_dof_idx)
                    stiff=False
            if recovery is not None and kick_start is not None and stage=="complete":
                elapsed=t-kick_start
                if elapsed>=handover:
                    fraction=float(np.clip((elapsed-handover)/.30,0,1))
                    recovery_actions=recovery(env.get_observations()["policy"])
                    actions=(1-fraction)*actions+fraction*recovery_actions
                    kp=np.array([100.0]*3+[80.0]*9)*(1-fraction)+20*fraction
                    kd=np.array([2.0]*3+[3.0]*9)*(1-fraction)+.5*fraction
                    env.robot.set_dofs_kp(kp.tolist(),env.motors_dof_idx)
                    env.robot.set_dofs_kv(kd.tolist(),env.motors_dof_idx)
                    phase="立位を安定させる" if elapsed<5 else "結果"
            obs,_,_,_=env.step(actions)
            if save_states and kick_start is not None and 1.3<=t-kick_start<=2.9:
                recovery_states.append(dict(t=t,qpos=env.robot.get_qpos()[0].cpu().tolist()))
            if kick_start is not None:
                kick_peak_speed=max(kick_peak_speed,float(env.ball.get_vel()[0,0].item()))
            if step%round(50/fps):continue
            pos=env.robot.get_pos()[0].cpu().numpy();ball=env.ball.get_pos()[0].cpu().numpy()
            pin_angles=[]
            for i,pin in enumerate(env.pins):
                q=pin.get_quat()[0].cpu().numpy()
                angle=float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
                knocked[i]=angle>45
                pin_angles.append(angle)
            score=int(knocked.sum())
            if record:
                if side:
                    env.camera.set_pose(up=(0,0,1),pos=(.45,-1.85,.85),lookat=(.45,0,.14))
                elif wide:
                    env.camera.set_pose(up=(0,0,1),pos=(1.8,-4.4,2.8),lookat=(1.25,0,0.15))
                elif ball[0]>2.4 and kick_start is not None:
                    env.camera.set_pose(up=(0,0,1),pos=(3.9,-1.9,1.30),lookat=(3.2,0,0.12))
                elif phase in ["構える","キック","足を戻す"] and ball[0]<1.60:
                    env.camera.set_pose(up=(0,0,1),pos=pos+np.array([.9,-1.3,.75]),lookat=pos+np.array([.3,-.05,-.1]))
                else:
                    env.camera.set_pose(up=(0,0,1),pos=(2.05,-3.4,2.30),lookat=(1.25,0,0.12))
                frame=np.asarray(env.camera.render()[0]).copy()
                if frame.ndim==4:frame=frame[0]
                im=Image.fromarray(frame);d=ImageDraw.Draw(im)
                d.rectangle((0,0,960,46),fill=(14,24,35));d.text((18,7),"4脚ロボのキック・ボウリング",font=font(27),fill=(239,245,249))
                d.text((755,7),"等速再生",font=font(23),fill=(176,199,216))
                d.rectangle((0,490,960,540),fill=(14,24,35));d.text((18,497),phase,font=font(25),fill=(176,199,216))
                score=int(knocked.sum());d.text((610 if score==6 else 710,493),f"STRIKE! 6 / 6" if score==6 else f"{score} / 6 本",font=font(32),fill=(68,208,165))
                writer.append_data(np.asarray(im))
                if side_writer is not None:
                    env.camera.set_pose(up=(0,0,1),pos=(.45,-1.85,.85),lookat=(.45,0,.14))
                    side_frame=np.asarray(env.camera.render()[0]).copy()
                    if side_frame.ndim==4:side_frame=side_frame[0]
                    side_image=Image.fromarray(side_frame)
                    side_image.paste(im.crop((0,0,960,46)),(0,0))
                    side_image.paste(im.crop((0,490,960,540)),(0,490))
                    side_writer.append_data(np.asarray(side_image))
            tilt=float(env.base_euler[0,:2].abs().max().item())
            foot_actual=calf.get_pos()[0].cpu().numpy()+rotation(calf.get_quat()[0].cpu().numpy())@np.array([0,0,-.213])
            samples.append(dict(t=t,phase=phase,robot=pos.tolist(),ball=ball.tolist(),foot=foot_actual.tolist(),
                                ball_vx=float(env.ball.get_vel()[0,0].item()),tilt=tilt,score=score,pin_angles=pin_angles))
    result=dict(name=name,mass_kg=mass,aim_y=aim_y,ball_y=ball_y,fps=fps,playback_speed=1,
                stage=stage,swing_time_s=swing_time,score=int(knocked.sum()),
                kick_start_s=kick_start,peak_ball_vx=kick_peak_speed,attachments=False,
                method={"complete":"PPO approach/stop; four-leg IK weight shift; right-front-foot IK/PD kick",
                        "walk-only":"PPO approach/stop only; kick disabled",
                        "unstable":"PPO approach/stop; prototype front-leg IK kick without support control"}[stage],samples=samples)
    result["max_tilt_deg"]=max(s["tilt"] for s in samples)
    result["final_tilt_deg"]=samples[-1]["tilt"]
    result["standing_at_end"]=bool(samples[-1]["tilt"]<10 and samples[-1]["robot"][2]>.20)
    result["approach_failed"]=approach_failed
    if recovery is not None:result["method"]+="; PPO post-kick standing recovery with 0.3s blend"
    result["recovery_policy"]=recovery is not None
    result["recovery_model"]=recovery_name
    result["handover_s"]=handover if recovery is not None else None
    if save_states:
        (ROOT/"results/kick_recovery_states.json").write_text(json.dumps(recovery_states,indent=2))
    result["source_episode"]=name
    result["hold_extra_s"]=hold_extra
    result["ball_x"]=ball_x
    result["vision_detection"]=detection
    result["video_recorded"]=record
    if detection is not None:
        result["detection_error_xy_m"]=float(np.linalg.norm(estimated_ball[:2]-np.array([ball_x,ball_y])))
    out.with_suffix(".json").write_text(json.dumps(result,indent=2))
    if side_out is not None:
        side_result=dict(result,name=side_out.stem.removeprefix("kick_"),camera="side")
        side_out.with_suffix(".json").write_text(json.dumps(side_result,indent=2))
    print("KICK_VIDEO_SAVED" if record else "KICK_EVALUATION_SAVED",out.name if record else out.with_suffix(".json").name,"score",result["score"],"peak_ball_vx",kick_peak_speed,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--name",default="probe01")
    parser.add_argument("--mass",type=float,default=1.0);parser.add_argument("--aim-y",type=float,default=.05)
    parser.add_argument("--aim-ys",nargs="+",type=float)
    parser.add_argument("--ball-y",type=float,default=0)
    parser.add_argument("--ball-ys",nargs="+",type=float)
    parser.add_argument("--fps",type=int,choices=[25,50],default=25)
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--wide",action="store_true")
    parser.add_argument("--side",action="store_true")
    parser.add_argument("--dual-view",action="store_true",help="Record simultaneous wide/side footage from one physics rollout")
    parser.add_argument("--model",default="locomotion")
    parser.add_argument("--recovery-model")
    parser.add_argument("--handover",type=float,default=1.5)
    parser.add_argument("--save-states",action="store_true")
    parser.add_argument("--stage",choices=["complete","walk-only","unstable"],default="complete")
    parser.add_argument("--swing-time",type=float,default=.14);args=parser.parse_args()
    cfg=json.loads((ROOT/"models/config.json").read_text());ec=copy.deepcopy(cfg["env"])
    ec.update(base_init_pos=[-.65,.142,.42],episode_length_s=60)
    gs.init(backend=gs.metal,logging_level="warning",seed=1)
    env=Go2Env(1,ec,cfg["obs"],cfg["reward"],cfg["command"],camera_res=(960,540),auto_reset=False,
               bowling=True,bowling_ball_radius=.12)
    network=torch.jit.load(str(ROOT/"models"/f"{args.model}.pt"),map_location=str(gs.device)).eval()
    recovery=torch.jit.load(str(ROOT/"models"/f"{args.recovery_model}.pt"),map_location=str(gs.device)).eval() if args.recovery_model else None
    if args.ball_ys:
        for i,y in enumerate(args.ball_ys):
            play(env,network,f"{args.name}_{i+1:02d}",args.mass,args.aim_y,args.swing_time,y,args.fps,args.wide,args.side,args.stage,recovery,args.handover,args.save_states,args.recovery_model,args.dual_view,record=not args.no_video)
    elif args.aim_ys:
        for i,aim in enumerate(args.aim_ys):
            play(env,network,f"{args.name}_{i+1:02d}",args.mass,aim,args.swing_time,args.ball_y,args.fps,args.wide,args.side,args.stage,recovery,args.handover,args.save_states,args.recovery_model,args.dual_view,record=not args.no_video)
    else:
        play(env,network,args.name,args.mass,args.aim_y,args.swing_time,args.ball_y,args.fps,args.wide,args.side,args.stage,recovery,args.handover,args.save_states,args.recovery_model,args.dual_view,record=not args.no_video)
    gs.destroy()


if __name__=="__main__":main()
