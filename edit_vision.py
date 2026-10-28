"""A separate full-frame video for the camera-based bowling experiment."""
import json
import subprocess
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from edit_bowling import card,stamp
from render_utils import font
from kick_bowling import ROOT

FPS=25

def main():
    summary=json.loads((ROOT/'results/vision_summary.json').read_text())
    assets=ROOT/'output/bowling_assets/vision_60587';assets.mkdir(parents=True,exist_ok=True)
    segments=[dict(kind='title',seconds=6,speech='球の位置をカメラで検出し、歩行とキックにつなげます。')]
    cases=[('vision_final_grid_05','基準位置','画像から球の中心を求め、その位置に合わせて近づきます。球の真の座標は、制御には使っていません。'),
           ('vision_final_grid_06','球を横へ4 cm','次は球を横に四センチずらしました。検出した位置に合わせて、止まる位置とキックの軌道を変えます。'),
           ('vision_final_unseen_01','別の位置 ①','ここからは調整に使っていない位置です。前後と左右の両方をずらして、同じ制御で試します。'),
           ('vision_final_unseen_03','別の位置 ②','この位置でも球を蹴り、立った姿勢を保ちます。ただし、倒せるピンの本数には、まだばらつきがあります。')]
    for name,label,speech in cases:
        segments.append(dict(kind='image',image=f'results/vision/{name}_detection.png',caption=label+'：カメラ画像',seconds=3,speech=''))
        segments.append(dict(file=f'kick_{name}.mp4',caption=label,seconds=16,speech=speech))
    segments.append(dict(kind='result',seconds=8,speech='位置ずれへの対応は確認できました。すべての条件で六本倒すことは、今後の課題です。'))
    for i,s in enumerate(segments):
        if not s['speech']:continue
        audio=assets/f'voice_{i:02d}.aiff'
        subprocess.run(['say','-v','Kyoko','-r','215','-o',str(audio),s['speech']],check=True)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(audio)]))
        s['audio']=str(audio);s['seconds']=max(s['seconds'],duration+.4)
    silent=ROOT/'output/video/vision_silent.mp4';timeline=[];offset=0
    with imageio.get_writer(str(silent),fps=FPS,codec='libx264',bitrate='4M',macro_block_size=1) as writer:
        for s in segments:
            n=round(s['seconds']*FPS);timeline.append(dict(**s,start_s=offset/FPS))
            if 'file' in s:
                reader=imageio.get_reader(str(ROOT/'videos'/s['file']));srcfps=reader.get_meta_data()['fps'];count=reader.count_frames()
                frames=(Image.fromarray(reader.get_data(min(round(j/FPS*srcfps),count-1))).resize((1280,720)) for j in range(n))
            else:
                frame=Image.new('RGB',(1280,720),(14,24,35));d=ImageDraw.Draw(frame)
                if s['kind']=='title':
                    d.text((80,250),'カメラで球を見つけて蹴る',font=font(54),fill='white');d.text((84,345),'球の位置ずれへの対応を検証',font=font(30),fill=(176,199,216))
                elif s['kind']=='image':
                    pic=ImageOps.contain(Image.open(ROOT/s['image']).convert('RGB'),(1000,650));frame.paste(pic,((1280-pic.width)//2,65))
                else:
                    g=summary['grid'];d.text((80,170),'前後・左右 ±4 cm：9条件',font=font(45),fill='white')
                    d.text((84,295),f"キックして立位維持　{g['kick_and_stand']} / 9",font=font(39),fill=(68,208,165))
                    d.text((84,385),f"6本すべて倒す　　　{g['all_six']} / 9",font=font(39),fill=(239,245,249))
                    d.text((84,500),'各条件1回のシミュレーション結果',font=font(27),fill=(176,199,216))
                if s['kind']!='image':d.text((80,660),'PAI最終課題  |  60587',font=font(24),fill=(176,199,216))
                frames=(frame.copy() for _ in range(n))
            for im in frames:
                if s.get('caption'):
                    d=ImageDraw.Draw(im);d.rectangle((0,0,1280,62),fill=(14,24,35));d.text((24,12),s['caption'],font=font(31),fill='white')
                    if 'file' in s:d.text((1110,14),'等速',font=font(26),fill=(176,199,216))
                writer.append_data(np.asarray(im))
            if 'file' in s:reader.close()
            offset+=n
    out=ROOT/'output/video/PAI最終課題_60587_画像認識版.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(silent)];filters=[];i=0
    for s in timeline:
        if 'audio' not in s:continue
        i+=1;cmd+=['-i',s['audio']];filters.append(f"[{i}:a]aresample=48000,adelay={round(s['start_s']*1000)}:all=1[a{i}]")
    filters.append(''.join(f'[a{k}]' for k in range(1,i+1))+f'amix=inputs={i}:normalize=0,volume=5dB,alimiter=limit=0.95[aout]')
    cmd+=['-filter_complex',';'.join(filters),'-map','0:v','-map','[aout]','-c:v','copy','-c:a','aac','-b:a','160k','-t',str(offset/FPS),'-movflags','+faststart',str(out)]
    subprocess.run(cmd,check=True)
    (assets/'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
    print('VISION_VIDEO',out.name,offset/FPS)

if __name__=='__main__':main()
