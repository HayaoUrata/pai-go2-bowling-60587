"""Footage-first submission edit: short captions, narration, and a normal replay."""

import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

from render_utils import font

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/"output/bowling_assets/four_stages_60587_recovery";ASSETS.mkdir(parents=True,exist_ok=True)
OUT=ROOT/"output/video";OUT.mkdir(parents=True,exist_ok=True)
FPS=25


def card(ending=False,heading=None):
    im=Image.new("RGB",(1280,720),(14,24,35));d=ImageDraw.Draw(im)
    if heading:
        d.text((80,280),heading,font=font(55),fill=(239,245,249))
        return np.asarray(im)
    d.text((80,245),"STRIKE!  6 / 6" if ending else "4脚ロボでボウリング",font=font(65),fill=(68,208,165) if ending else (239,245,249))
    d.text((84,350),"足で蹴る。全部倒す。" if not ending else "6本すべて倒して、姿勢を維持",font=font(35),fill=(176,199,216))
    d.text((80,650),"PAI最終課題  |  60587",font=font(25),fill=(176,199,216))
    return np.asarray(im)


def stamp(seconds):
    n=round(seconds*1000);return f"{n//3600000:02d}:{n//60000%60:02d}:{n//1000%60:02d},{n%1000:03d}"


def main():
    segments=[
        dict(kind="title",seconds=8,speech="四脚ロボットが足で球を蹴り、六本のピンを倒す課題です。歩行の学習と、キック制御の改良を、四段階で比較します。"),
        dict(file="kick_untrained_comparison.mp4",start=0,seconds=10,caption="① 歩行の学習前",speech="まず、歩行を学ぶ前です。球へ向かう指令を出しても近づけず、キックに進めません。"),
        dict(file="kick_walk_only.mp4",start=0,seconds=12,caption="② 歩行のみ：キックなし",speech="次は、学習済みの歩行です。球の前まで近づいて停止できますが、キック制御はまだ使っていません。"),
        dict(file="kick_unstable.mp4",start=0,seconds=12,caption="③ キック試作：姿勢が崩れる",speech="同じ歩行に、前足を振る制御を加えました。しかし、体重移動と支え方が不十分で、姿勢を崩してしまいます。"),
        dict(file="kick_stable_wide.mp4",start=0,seconds=16,caption="④ 完成：姿勢維持も追加学習",speech="完成版です。支持制御の改良に加え、キック後に立つ動作も追加学習しました。六本すべて倒し、姿勢も保ちます。"),
        dict(kind="transition",heading="完成版を横からリプレイ",seconds=3,speech=""),
        dict(file="kick_stable_side.mp4",start=2,seconds=16,rate=.25,caption="完成版・横視点リプレイ",speech="横からスローで見ます。先に体重を移して三本の脚で支え、右前足で蹴ります。歩行と、蹴った後の姿勢維持は強化学習です。キック軌道には逆運動学とピーディー制御を使います。"),
        dict(file="kick_stable_wide.mp4",start=6,seconds=10,caption="全景で結果を確認",speech="全景で結果を確認します。六本が倒れた後も、ロボットは立った姿勢を維持しています。"),
        dict(kind="ending",seconds=4,speech=""),
    ]
    for i,s in enumerate(segments):
        if not s["speech"]:continue
        audio=ASSETS/f"voice_{i:02d}.aiff"
        subprocess.run(["say","-v","Kyoko","-r","215","-o",str(audio),s["speech"]],check=True)
        duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(audio)]))
        s["audio"]=str(audio);s["speech_seconds"]=duration
        s["seconds"]=max(s["seconds"],round(duration+.6,2))
    silent=OUT/"bowling_four_stages_silent.mp4";offset=0;timeline=[]
    with imageio.get_writer(str(silent),fps=FPS,codec="libx264",bitrate="4M",macro_block_size=1) as writer:
        for i,s in enumerate(segments):
            n=round(s["seconds"]*FPS)
            timeline.append(dict(**s,start_s=offset/FPS,end_s=(offset+n)/FPS))
            if s.get("kind"):
                frame=card(s["kind"]=="ending",s.get("heading"))
                for _ in range(n):writer.append_data(frame)
            else:
                reader=imageio.get_reader(str(ROOT/"videos"/s["file"]));meta=reader.get_meta_data();srcfps=meta["fps"];available=reader.count_frames()
                try:
                    for j in range(n):
                        physical_t=s["start"]+(j/FPS)*s.get("rate",1)
                        k=min(round(physical_t*srcfps),available-1)
                        im=Image.fromarray(reader.get_data(k)).resize((1280,720),Image.Resampling.LANCZOS)
                        d=ImageDraw.Draw(im);d.rectangle((0,0,1280,62),fill=(14,24,35))
                        d.text((24,10),s["caption"],font=font(31),fill=(239,245,249))
                        d.text((1050,13),"1/4倍速" if s.get("rate",1)!=1 else "等速",font=font(26),fill=(176,199,216))
                        writer.append_data(np.asarray(im))
                finally:reader.close()
            offset+=n;print("SEGMENT",i,"end",offset/FPS,flush=True)
    final=OUT/"PAI最終課題_60587.mp4"
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(silent)];filters=[];audio_i=0
    for s in timeline:
        if "audio"not in s:continue
        audio_i+=1;cmd +=["-i",s["audio"]]
        delay=round(s["start_s"]*1000);filters.append(f"[{audio_i}:a]aresample=48000,adelay={delay}:all=1[a{audio_i}]")
    inputs="".join(f"[a{i}]"for i in range(1,audio_i+1))
    filters.append(f"{inputs}amix=inputs={audio_i}:normalize=0,volume=5dB,alimiter=limit=0.95[aout]")
    cmd +=["-filter_complex",";".join(filters),"-map","0:v","-map","[aout]","-c:v","copy","-c:a","aac","-b:a","160k","-t",str(offset/FPS),"-movflags","+faststart",str(final)]
    subprocess.run(cmd,check=True)
    (ASSETS/"timeline.json").write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
    srt=""
    for i,s in enumerate(timeline):
        if not s["speech"]:continue
        end=min(s["end_s"],s["start_s"]+s["speech_seconds"]+.3)
        lines="\n".join(s["speech"][k:k+30]for k in range(0,len(s["speech"]),30))
        srt+=f"{i+1}\n{stamp(s['start_s'])} --> {stamp(end)}\n{lines}\n\n"
    final.with_suffix(".srt").write_text(srt)
    print("FINAL_BOWLING_VIDEO",final.name,offset/FPS,flush=True)


if __name__=="__main__":main()
