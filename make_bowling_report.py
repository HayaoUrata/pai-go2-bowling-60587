"""Short experiment report for the Go2 bowling submission."""
import json
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Image, Spacer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
INK=colors.HexColor('#222222')
BODY=ParagraphStyle('body',fontName='JP',fontSize=10,leading=16,wordWrap='CJK',spaceAfter=9,textColor=INK)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=8.5,leading=13,spaceAfter=6)
HEAD=ParagraphStyle('head',parent=BODY,fontSize=12,leading=19,spaceBefore=12,spaceAfter=7,keepWithNext=True)
TITLE=ParagraphStyle('title',parent=BODY,fontSize=19,leading=27,spaceAfter=8)
CAPTION=ParagraphStyle('caption',parent=SMALL,alignment=1,spaceBefore=4,spaceAfter=8)


def p(text,style=BODY):return Paragraph(text,style)


def table(rows,widths):
    t=Table([[p(str(c),SMALL) for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('LINEABOVE',(0,0),(-1,0),.7,colors.HexColor('#777777')),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#777777')),
        ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cccccc')),
    ]))
    return t


def footer(c,doc):
    c.setFont('JP',8);c.setFillColor(colors.HexColor('#666666'))
    c.drawString(46,25,'Physical AI 応用1 最終課題　60587')
    c.drawRightString(A4[0]-46,25,str(doc.page))


def main():
    names=['untrained_comparison','walk_only','unstable','stable_wide']
    stages=[json.loads((ROOT/'videos'/f'kick_{n}.json').read_text()) for n in names]
    final=stages[-1]
    before=json.loads((ROOT/'videos/kick_final_wide.json').read_text())
    frame=ROOT/'results/bowling_strike.png'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','8','-i',str(ROOT/'videos/kick_stable_wide.mp4'),'-frames:v','1',str(frame)],check=True)
    story=[
        p('4脚ロボットでボウリング',TITLE),
        p('OmniCampus ID：60587　　2026年10月28日',SMALL),
        p('1. 取り組んだ課題',HEAD),
        p('Go2が球の前まで歩き、右前足で蹴って6本のピンを倒すタスクを作成した。歩行を使った移動に、足で物を動かす動作を加えたいと考え、ボウリングを選んだ。ピンを倒すだけでなく、蹴った後もロボットが立っていることを目標とした。'),
        p('2. 制御と学習',HEAD),
        p('シミュレータにはGenesisを使った。球の位置はシミュレータから取得し、球の手前で止まるように速度指令を与える。歩行にはPPOで学習した方策を使用した。入力は胴体の姿勢・角速度、関節の角度・速度、速度指令など45値で、12関節の目標角度の補正量を出力する。'),
        p('キックでは足先の軌道を手設計し、逆運動学（IK）で関節角度に変換してPD制御で追従させた。球は足との接触で転がり、ピンとの衝突はGenesisが計算する。歩行とキックに加え、後述する姿勢の崩れに対応するため、蹴った後の立位維持にも別のPPO方策を用意した。'),
        table([
            ['学習対象','条件'],
            ['歩行・停止','基本歩行100反復、横方向の外乱を加えて150反復、停止・速度変更・旋回を含む指令で500反復。計750反復。1,024並列環境、24ステップ／反復、seed=1。'],
            ['キック後の立位維持','歩行方策を初期値として追加学習。キック後に記録した79姿勢と通常の立位を初期状態とし、速度は0にして開始した。512並列環境、24ステップ／反復、seed=2。450反復まで学習し、投球で評価した350反復時点を採用した。'],
            ['共通の設定','Actor・Criticは512→256→128のMLP（ELU）。PPOは5 epoch、4 minibatch。制御は50 Hz、物理計算は2サブステップ。報酬やその他の設定値は同梱の設定ファイルに記載した。'],
        ],[106,397]),
        p('3. うまくいかなかった点と修正',HEAD),
        p('最初は前足を上げて振るだけだったため、支持を失って転倒した。そこで、足を上げる前に胴体の目標位置を後方6 cm、左12 cm、下1.5 cmへ移し、残りの3脚で支えるようにした。前足の目標軌道は約33.5 cm、振り出し時間は0.16秒とした。'),
        p('この変更で6本を倒せたが、蹴った後に胴体を戻す区間で約20°傾いた。復帰用の方策には、胴体の傾き・角速度・高さのずれ・急な関節動作を抑える報酬を与えた。150反復の後に傾きへの罰則を強め、さらに200反復したモデルを採用した。切り替えも見直し、キック開始1.5秒後から0.3秒かけて復帰用の方策へ移るようにした。'),
        PageBreak(),
        p('4. 結果',HEAD),
        p('球は質量1.0 kg、半径0.12 m、ピンは各0.10 kg、高さ約0.315 mとした。6本のピンを三角形に並べ、側面にバンパーのある5.6×1.4 mのレーンを使った。鉛直から45°を超えて傾いたピンを倒れたものとして数えた。'),
    ]
    rows=[['比較した段階','観察した動作','倒した本数']]
    observations=[
        ('歩行の学習前','球の前まで到達できなかった。'),
        ('歩行のみ','球の前まで歩いて停止した。キックは行わない。'),
        ('キックの試作','前足を振ろうとして姿勢を崩し、転倒した。'),
        ('完成版','6本を倒し、蹴った後も立位を保った。'),
    ]
    for (label,observation),data in zip(observations,stages):
        rows.append([label,observation,f"{data['score']} / 6"])
    story += [
        table(rows,[111,316,76]),Spacer(1,7),
        p('球とピンの配置、開始姿勢、全景カメラをそろえて比較した。歩行のみ・キックの試作・完成版では同じ歩行方策を使い、キックの制御を変更している。完成版には復帰用の方策も加えているため、この4段階は学習回数だけの比較ではない。',SMALL),
        p(f"完成版では、終了時にも6本すべてが倒れていた。ロボットの最大傾きは{before['max_tilt_deg']:.1f}°から{final['max_tilt_deg']:.1f}°に減り、終了時は{final['final_tilt_deg']:.2f}°だった。ここでの傾きはロール・ピッチの絶対値の大きい方である。改善には復帰用方策の追加学習と、制御を切り替えるタイミングの両方が寄与している。"),
        Image(str(frame),width=336,height=189),
        p('図1　完成版の投球後。6本のピンが倒れ、Go2は立位を保っている。',CAPTION),
        p('5. 今回確認できていないこと',HEAD),
        p('今回の結果は、球の位置とレーンの条件を固定した場合のものである。球の位置や摩擦を変えたときの成功率までは調べていない。また、球の位置は既知としており、カメラ画像からの検出は行っていない。次に試すなら、球を置く位置を少しずつ変え、どの範囲まで蹴れるかを確認したい。実機での動作も未検証である。'),
        p('6. 実行環境・コード・参考資料',HEAD),
        p('Apple M5 Pro（16コアGPU、24 GBメモリ）でMetal／MPSを使用した。Python 3.12、Genesis 1.4.5、PyTorch 2.9.1、rsl-rl 5.5.1で動作を確認した。',SMALL),
        p('実行手順はREADME.mdに記載した。kick_bowling.pyがタスク実行、train_recovery.pyが復帰用方策の学習を担当する。歩行方策はmodels/locomotion.pt、復帰用方策はmodels/recovery.ptに保存した。提出動画は1分32.4秒で、全景と横視点は同じエピソードから録画している。',SMALL),
        p('歩行環境はGenesisの公式例を基にした。原本とApache-2.0ライセンスはvendor/、変更点はTHIRD_PARTY.mdに記載した。',SMALL),
        p('<link href="https://genesis-world.readthedocs.io/en/v1.4.5/user_guide/policy_training/examples/locomotion.html" color="#333333">[1] Genesis World, Go2 locomotion example</link><br/><link href="https://github.com/leggedrobotics/rsl_rl" color="#333333">[2] rsl-rl（PPO実装、BSD-3-Clause）</link>',SMALL),
    ]
    path=OUT/'説明資料_PAI最終課題_60587.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=39,bottomMargin=48,title='4脚ロボットでボウリング',author='60587').build(story,onFirstPage=footer,onLaterPages=footer)
    print('REPORT_SAVED',path)

if __name__=='__main__':main()
