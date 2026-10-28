"""Package the reviewed code, models, report, and source footage for submission."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent

def main():
    names=['.gitignore','README.md','THIRD_PARTY.md','requirements.txt',
           'environment.py','train.py','video.py','render_utils.py','kick_bowling.py',
           'edit_bowling.py','make_report.py','make_bowling_report.py','initialize_untrained.py',
           'train_recovery.py','record_comparison.py','analyze_recovery.py','package_submission.py',
           'assets/bowling_pin.urdf','vendor/go2_env.py','vendor/go2_train.py','vendor/GENESIS_LICENSE',
           'models/config.json','models/locomotion.pt','models/untrained.pt','models/recovery.pt',
           'models/locomotion_training_config.json','models/untrained_training_config.json',
           'models/recovery_initial_training_config.json','models/recovery_balance_training_config.json',
           'models/manifest.json','models/checkpoints/locomotion_0750.pt','models/checkpoints/recovery_0350.pt',
           'results/bowling_strike.png','results/kick_recovery_states.json',
           'results/recovery_selection.json','results/recovery_evaluations.json',
           'videos/kick_earlier_handover.json','videos/kick_recovery_balance_0200.json',
           'output/pdf/説明資料_PAI最終課題_60587.pdf','output/video/PAI最終課題_60587.srt']
    for n in ['untrained_comparison','walk_only','unstable','final_wide','final_side','stable_wide','stable_side']:
        names += [f'videos/kick_{n}.{ext}' for ext in ['mp4','json']]
    names += ['combine_submission.py','results/submission_timeline.json',
              'output/video/PAI最終課題_60587.mp4',
              'output/video/PAI最終課題_60587_統合版.mp4',
              'output/video/PAI最終課題_60587_統合版.srt',
              'vision_bowling.py','summarize_vision.py','edit_vision.py',
              'results/vision_summary.json','results/vision_tuning_results.json',
              'output/video/PAI最終課題_60587_画像認識版.mp4']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'videos').glob('kick_vision_final_*')) if p.suffix in ['.mp4','.json']]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'results/vision').glob('vision_final_*')) if p.suffix in ['.png','.json']]
    out=ROOT/'output/PAI最終課題_60587_コード.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(names)):
            path=ROOT/name
            if not path.is_file():raise FileNotFoundError(name)
            z.write(path,'go2-kick-bowling/'+name)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        print('PACKAGE_VERIFIED',out.name,len(z.namelist()),out.stat().st_size)

if __name__=='__main__':main()
