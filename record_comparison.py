"""Record the three baseline stages with the submission's fixed camera."""
import copy
import json
import genesis as gs
import torch
from kick_bowling import ROOT,play
from environment import Go2Env


def main():
    cfg=json.loads((ROOT/'models/config.json').read_text())
    ec=copy.deepcopy(cfg['env']);ec.update(base_init_pos=[-.65,.142,.42],episode_length_s=60)
    gs.init(backend=gs.metal,logging_level='warning',seed=1)
    env=Go2Env(1,ec,cfg['obs'],cfg['reward'],cfg['command'],camera_res=(960,540),auto_reset=False,bowling=True,bowling_ball_radius=.12)
    for name,model,stage in [('untrained_comparison','untrained','complete'),('walk_only','locomotion','walk-only'),('unstable','locomotion','unstable')]:
        network=torch.jit.load(str(ROOT/'models'/f'{model}.pt'),map_location=str(gs.device)).eval()
        play(env,network,name,1.0,0,.16,fps=50,wide=True,stage=stage)
    gs.destroy()

if __name__=='__main__':main()
