"""Summarize actual recorded recovery evaluations; no success-rate inference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def metrics(path):
    d=json.loads(path.read_text())
    samples=d['samples']
    after=[s for s in samples if s['t']>=d['kick_start_s']+1.8]
    tail=[s for s in samples if s['t']>=11]
    return dict(file=str(path.relative_to(ROOT)),score=d['score'],standing=d['standing_at_end'],
        peak_tilt_deg=max(s['tilt'] for s in samples),recovery_peak_tilt_deg=max(s['tilt'] for s in after),
        final_tilt_deg=samples[-1]['tilt'],last_5s_mean_tilt_deg=sum(s['tilt'] for s in tail)/len(tail),
        learned_recovery=d.get('recovery_model'),handover_s=d.get('handover_s'))

if __name__=='__main__':
    paths=[ROOT/'videos/kick_final_wide.json',ROOT/'videos/kick_earlier_handover.json']
    paths+=sorted((ROOT/'videos').glob('kick_recovery_*.json'))
    result=[metrics(p) for p in paths]
    (ROOT/'results/recovery_evaluations.json').write_text(json.dumps(result,indent=2))
    for r in result:
        print(Path(r['file']).stem, 'pins',r['score'],'peak',round(r['peak_tilt_deg'],2),
              'post',round(r['recovery_peak_tilt_deg'],2),'end',round(r['final_tilt_deg'],2))
