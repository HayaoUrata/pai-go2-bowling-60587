"""Combine both experiments without cutting narration or speeding up trials."""
import json
import subprocess
from pathlib import Path
from PIL import Image,ImageDraw
from render_utils import font

ROOT=Path(__file__).resolve().parent
VIDEO=ROOT/'output/video'
ASSETS=ROOT/'output/bowling_assets/combined_60587'


def stamp(t):
    n=round(t*1000)
    return f'{n//3600000:02d}:{n//60000%60:02d}:{n//1000%60:02d},{n%1000:03d}'


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    first=VIDEO/'PAI最終課題_60587.mp4'
    second=VIDEO/'PAI最終課題_60587_画像認識版.mp4'
    final=VIDEO/'PAI最終課題_60587_統合版.mp4'
    summary=json.loads((ROOT/'results/vision_summary.json').read_text())
    rows=summary['trials'];standing=sum(r['kicked'] and r['standing'] for r in rows);strikes=sum(r['pins']==6 for r in rows)
    im=Image.new('RGB',(1280,720),(14,24,35));d=ImageDraw.Draw(im)
    d.text((80,110),'結果',font=font(55),fill='white')
    d.text((84,245),'既知の球位置：6本倒して立位維持',font=font(38),fill=(68,208,165))
    d.text((84,350),f'カメラ検出：{standing}条件でキック・立位維持',font=font(36),fill=(239,245,249))
    d.text((84,415),f'全6本成功は {len(rows)}条件中{strikes}条件',font=font(36),fill=(239,245,249))
    d.text((84,515),'位置ずれ時の全倒しは、今後の課題',font=font(29),fill=(176,199,216))
    d.text((80,660),'PAI最終課題  |  60587',font=font(24),fill=(176,199,216))
    card=ASSETS/'summary.png';im.save(card)
    cuts=[(0,0,16.4),(0,19.4,29.4),(0,31.4,41.4),(0,43.4,85.4),(1,0,82),(1,82,90)]
    filters=[];offset=0;timeline=[]
    for i,(source,start,end) in enumerate(cuts):
        dur=end-start
        video_filter=f'[2:v]trim=duration={dur}' if i==len(cuts)-1 else f'[{source}:v]trim=start={start}:end={end}'
        filters.append(video_filter+f',setpts=PTS-STARTPTS,fps=25,setsar=1,format=yuv420p[v{i}]')
        filters.append(f'[{source}:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,aresample=48000[a{i}]')
        timeline.append(dict(source=first.name if source==0 else second.name,source_start_s=start,source_end_s=end,start_s=round(offset,3),end_s=round(offset+dur,3)))
        offset+=dur
    filters.append(''.join(f'[v{i}][a{i}]' for i in range(len(cuts)))+f'concat=n={len(cuts)}:v=1:a=1[v][a]')
    script=ASSETS/'combine.ffscript';script.write_text(';\n'.join(filters))
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(first),'-i',str(second),'-loop','1','-framerate','25','-i',str(card),'-filter_complex_script',str(script),'-map','[v]','-map','[a]','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-t',str(round(offset,3)),'-metadata','title=PAI最終課題_60587','-movflags','+faststart',str(final)],check=True)
    (ROOT/'results/submission_timeline.json').write_text(json.dumps(dict(duration_s=round(offset,3),segments=timeline,chapters=[dict(start_s=0,title='歩行とキックの4段階比較'),dict(start_s=78.4,title='カメラ検出と位置ずれの実験'),dict(start_s=160.4,title='全体の結果')]),ensure_ascii=False,indent=2))
    # Speech texts and their original offsets are retained in the edit logs.
    subtitle=[]
    for source,path in enumerate([ROOT/'output/bowling_assets/four_stages_60587_recovery/timeline.json',ROOT/'output/bowling_assets/vision_60587/timeline.json']):
        if not path.exists():continue
        speeches=json.loads(path.read_text())
        for seg in speeches:
            if not seg.get('speech'):continue
            begin=seg['start_s'];end=begin+seg.get('speech_seconds',min(seg['seconds']-.3,12))
            for chunk,(src,left,right) in zip(timeline,cuts):
                if source==src and left<=begin<right:
                    subtitle.append((chunk['start_s']+begin-left,chunk['start_s']+min(end,right)-left,seg['speech']))
                    break
    subtitle.sort()
    final.with_suffix('.srt').write_text(''.join(f'{i}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n' for i,(start,end,text) in enumerate(subtitle,1)))
    print('COMBINED_VIDEO',final.name,round(offset,3),flush=True)

if __name__=='__main__':main()
