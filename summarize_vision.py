"""Aggregate all final camera-based trials, including partial pin scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    for group,prefix,count in [('grid','vision_final_grid',9),('unseen','vision_final_unseen',3)]:
        for i in range(1,count+1):
            name=f'{prefix}_{i:02d}'
            d=json.loads((ROOT/'videos'/f'kick_{name}.json').read_text())
            hit=bool(d['peak_ball_vx']>.1 and d['kick_start_s'] is not None)
            rows.append(dict(name=name,group=group,dx_m=round(d['ball_x']-.85,6),dy_m=d['ball_y'],
                detected=d['vision_detection'] is not None,detection_error_mm=d['detection_error_xy_m']*1000,
                kicked=hit,standing=d['standing_at_end'],pins=d['score'],max_tilt_deg=d['max_tilt_deg'],
                video=f'videos/kick_{name}.mp4'))
    result=dict(method='fixed calibrated RGB-D camera; RGB ellipse and known ball-center plane; depth validation; no ground-truth ball position in control',
        assumptions=['orange ball radius 0.12 m','flat floor top z=0.02 m','fixed camera calibration','ball stationary until kick','single initial detection'],
        trials=rows)
    for group in ['grid','unseen']:
        rs=[r for r in rows if r['group']==group]
        result[group]=dict(trials=len(rs),detected=sum(r['detected'] for r in rs),kick_and_stand=sum(r['kicked'] and r['standing'] for r in rs),
            all_six=sum(r['pins']==6 for r in rs),max_detection_error_mm=max(r['detection_error_mm'] for r in rs))
    (ROOT/'results/vision_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({g:result[g] for g in ['grid','unseen']},indent=2))

if __name__=='__main__':main()
