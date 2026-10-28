"""RGB-D ball detection and bowling with measured, rather than ground-truth, ball position."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch
from scipy.optimize import least_squares
from PIL import Image,ImageDraw

from environment import Go2Env
from kick_bowling import ROOT,play
from render_utils import font

class BallDetector:
    """Receives only a calibrated RGB-D camera. No entity IDs or simulator ball state."""
    def __init__(self,camera):self.camera=camera

    def detect(self,name):
        self.camera.set_pose(pos=(.85,-1.4,1.8),lookat=(.85,0,.14),up=(0,0,1))
        rgb,depth,_,_=self.camera.render(rgb=True,depth=True)
        rgb=np.asarray(rgb);depth=np.asarray(depth)
        if rgb.ndim==4:rgb=rgb[0]
        if depth.ndim==3:depth=depth[0]
        c=rgb.astype(float)
        mask=((c[:,:,0]>1.8*c[:,:,1])&(c[:,:,1]>1.25*c[:,:,2])&(c[:,:,0]>45)).astype(np.uint8)*255
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
        candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=25]
        if not candidates:raise RuntimeError('Ball not detected; trial stopped without ground-truth fallback')
        component=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
        selected=(labels==component).astype(np.uint8)
        interior=cv2.erode(selected,np.ones((3,3),np.uint8))
        ys,xs=np.where(interior)
        z=depth[ys,xs];valid=np.isfinite(z)&(z>0)&(z<5)
        xs=xs[valid];ys=ys[valid];z=z[valid]
        K=np.asarray(self.camera.intrinsics)
        pc=np.stack(((xs+.5-K[0,2])*z/K[0,0],(ys+.5-K[1,2])*z/K[1,1],z),axis=1)
        T=np.linalg.inv(np.asarray(self.camera.extrinsics)).reshape(4,4)
        points=pc@T[:3,:3].T+T[:3,3]
        # Fit the known 12 cm ball radius to visible RGB-selected surface points.
        linear=np.linalg.lstsq(np.c_[2*points,np.ones(len(points))],(points*points).sum(1),rcond=None)[0][:3]
        fit=least_squares(lambda center:np.linalg.norm(points-center,axis=1)-.12,linear,loss='soft_l1',f_scale=.002)
        depth_center=fit.x
        position=depth_center.copy()
        residual=float(np.sqrt(np.mean((np.linalg.norm(points-position,axis=1)-.12)**2)))
        if not np.all(np.isfinite(position)) or residual>.008 or not .08<position[2]<.20:
            raise RuntimeError(f'Unreliable detection: residual={residual}, estimate={position}')
        # The ball rests on a calibrated flat surface: its center is 0.14 m high.
        # Use the image silhouette center for XY; depth sphere fit validates the detection.
        contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        ellipse=cv2.fitEllipse(max(contours,key=cv2.contourArea))
        u,v=ellipse[0]
        ray=T[:3,:3]@np.array([(u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],1])
        position=T[:3,3]+((.14-T[2,3])/ray[2])*ray
        if np.linalg.norm(position-depth_center)>.01:raise RuntimeError("RGB/depth estimates disagree")
        position[:2]=np.round(position[:2],3)  # Millimeter quantization suppresses subpixel jitter.
        x,y,w,h=stats[component,:4]
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        draw.rectangle((int(x)-3,int(y)-3,int(x+w)+3,int(y+h)+3),outline=(50,255,150),width=3)
        draw.rectangle((0,0,640,40),fill=(14,24,35));draw.text((14,6),'カメラ画像から球を検出',font=font(23),fill='white')
        dest=ROOT/'results/vision';dest.mkdir(parents=True,exist_ok=True)
        im.save(dest/f'{name}_detection.png')
        data=dict(position=position.tolist(),pixel_centroid=centroids[component].tolist(),surface_points=len(points),sphere_fit_rms_m=residual,depth_sphere_center=depth_center.tolist(),source='RGB ellipse center + calibrated ground-plane projection, RGB-D sphere validation, 1mm quantization; no object ID or true position')
        (dest/f'{name}_detection.json').write_text(json.dumps(data,indent=2))
        print('DETECTED',name,data,flush=True)
        return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='vision_nominal')
    p.add_argument('--dx',type=float,default=0);p.add_argument('--dy',type=float,default=0)
    p.add_argument('--grid',action='store_true');p.add_argument('--held-out',action='store_true');p.add_argument('--no-video',action='store_true')
    p.add_argument('--dual-view',action='store_true');p.add_argument('--fps',type=int,default=25)
    p.add_argument('--hold-extra',type=float,default=0);p.add_argument('--aim-y',type=float,default=0);args=p.parse_args()
    cfg=json.loads((ROOT/'models/config.json').read_text());ec=copy.deepcopy(cfg['env'])
    ec.update(base_init_pos=[-.65,.142,.42],episode_length_s=60)
    gs.init(backend=gs.metal,logging_level='warning',seed=1)
    env=Go2Env(1,ec,cfg['obs'],cfg['reward'],cfg['command'],camera_res=(960,540),auto_reset=False,bowling=True,bowling_ball_radius=.12,vision_camera_res=(640,480))
    walk=torch.jit.load(str(ROOT/'models/locomotion.pt'),map_location=str(gs.device)).eval()
    recovery=torch.jit.load(str(ROOT/'models/recovery.pt'),map_location=str(gs.device)).eval()
    detector=BallDetector(env.robot_camera)
    offsets=[(x,y) for x in [-.04,0,.04] for y in [-.04,0,.04]] if args.grid else [(args.dx,args.dy)]
    if args.held_out:offsets=[(.0173,-.0234),(-.0306,.0122),(.0264,.0327)]
    for i,(dx,dy) in enumerate(offsets):
        name=f'{args.name}_{i+1:02d}' if args.grid or args.held_out else args.name
        play(env,walk,name,1,args.aim_y,.16,ball_y=dy,fps=args.fps,wide=True,recovery=recovery,handover=1.5+args.hold_extra,recovery_name='recovery',dual_view=args.dual_view,ball_x=.85+dx,detector=detector,record=not args.no_video,hold_extra=args.hold_extra)
    gs.destroy()

if __name__=='__main__':main()
