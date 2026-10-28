"""Stream actual simulator frames to MP4, with conditions and state visible."""

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=10)
def font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Japanese font required: install Noto Sans CJK or use macOS Arial Unicode.")


def paint_hud(frame, label, t, speed, distance, roll, force_n, active, approaching, recovering, event):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if label.startswith("untrained"):
        title = "学習前（0回）"
    elif label.startswith("robust"):
        title = "学習後：外乱を加えて学習"
    elif label.startswith("control"):
        title = "学習後：通常の歩行学習"
    elif label.startswith("baseline"):
        match = re.search(r"iter(\d+)", label)
        title = f"基本歩行の学習（{int(match[1])}回）" if match else "基本歩行の学習"
    else:
        title = label
    draw.rectangle((0, 0, width, 112), fill=(14, 24, 35))
    draw.text((18, 7), title, font=font(28), fill=(242, 247, 250))
    draw.text((width - 216, 12), f"実時間 {t:04.1f} 秒", font=font(22), fill=(190, 208, 221))
    draw.text((18, 47), f"目標 0.50 m/s　 現在 {speed:+.2f} m/s　 前進 {distance:.2f} m",
              font=font(23), fill=(237, 244, 248))
    draw.text((18, 81), "外乱中：8倍スロー" if active else "通常再生（等速）",
              font=font(22), fill=(255, 160, 85) if active else (185, 207, 222))
    draw.text((655, 81), f"左右の傾き {roll:+.1f}°", font=font(22), fill=(185, 207, 222))
    if active or approaching:
        y = height - 112
        draw.rectangle((16, y, 720, height - 12), fill=(8, 16, 24),
                       outline=(255, 130, 50), width=3 if active else 1)
        headline = f"横から {force_n:g} N の力" if active else f"あと {event-t:.1f} 秒で横から押す"
        detail = "実際の印加 0.20秒 ／ この区間は8倍スロー" if active else f"同じ条件の外乱：{force_n:g} N を0.20秒"
        draw.text((32, y + 10), headline, font=font(30), fill=(255, 160, 85))
        draw.text((32, y + 57), detail, font=font(23), fill=(238, 245, 249))
    elif recovering:
        draw.rectangle((16, height-56, 650, height-12), fill=(8, 16, 24))
        draw.text((30, height-53), "外乱のあと：姿勢と歩行速度を確認", font=font(24), fill=(238, 245, 249))
    return np.asarray(image)


@torch.inference_mode()
def record_rollout(env, policy, output, label, duration=12, force_n=0):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env.scene.reset()
    obs = env.reset()
    env.eval_force = torch.zeros((1, 1, 3), device=env.device)
    # reset() defers forward kinematics; get_pos() can still return the prior rollout.
    start_x = float(env.init_base_pos[0].item())
    samples = []
    writer = imageio.get_writer(str(output), fps=25, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p")
    started = time.perf_counter()
    video_frames = 0
    events = [event for event in [3.0, 7.0] if event + 0.2 <= duration]
    try:
        for step in range(round(duration / env.dt)):
            t = step * env.dt
            # Progress clips use one fixed command even when training varies it.
            env.commands[:] = torch.tensor([0.5,0.0,0.0],device=env.device)
            env._update_observation()
            obs = env.get_observations()
            active = any(event <= t < event + 0.2 for event in events)
            env.eval_force[0, 0, 1] = force_n if active else 0
            obs, _, _, _ = env.step(policy(obs))
            slow_motion = bool(active and force_n)
            if step % 2 and not slow_motion:
                continue
            pos = env.robot.get_pos()[0].cpu().numpy()
            env.camera.set_pose(pos=pos + np.array([1.0, -1.4, 0.7]), lookat=pos)
            env.scene.clear_debug_objects()
            if active and force_n:
                # The arrow tip is at the trunk; +world-Y is the actual force direction.
                length = abs(force_n) / 100.0
                vec = np.array([0.0, np.copysign(length, force_n), 0.0])
                env.scene.draw_debug_arrow(pos=pos - vec, vec=vec, radius=0.03,
                                           color=(1.0, 0.33, 0.12, 1.0))
            frame = np.asarray(env.camera.render()[0]).copy()
            if frame.ndim == 4:
                frame = frame[0]
            if slow_motion:
                # Screen-space outline keeps the physical world-space arrow readable
                # even when parts of the 3D arrow overlap the robot mesh.
                def project(world):
                    point = env.camera.extrinsics @ np.append(world, 1.0)
                    pixel = env.camera.intrinsics @ point[:3]
                    return tuple(np.rint(pixel[:2] / pixel[2]).astype(int))
                tail, tip = project(pos - vec), project(pos)
                cv2.arrowedLine(frame, tail, tip, (8, 16, 24), 18, cv2.LINE_AA, tipLength=0.28)
                cv2.arrowedLine(frame, tail, tip, (255, 115, 35), 11, cv2.LINE_AA, tipLength=0.28)
            speed = float(env.base_lin_vel[0, 0].item())
            roll = float(env.base_euler[0, 0].item())
            pitch = float(env.base_euler[0, 1].item())
            fall = bool(pos[2] < 0.12 or abs(roll) > 35 or abs(pitch) > 35)
            samples.append(dict(t=t, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                                vx=speed, roll=roll, pitch=pitch, push=active, fallen=fall))
            next_event = next((event for event in events if event > t), 0)
            approaching = bool(force_n and next_event and next_event - 0.7 <= t < next_event)
            recovering = bool(force_n and any(event + 0.2 <= t < event + 1.5 for event in events))
            frame = paint_hud(frame, label, t, speed, float(pos[0])-start_x, roll,
                              force_n, slow_motion, approaching, recovering, next_event)
            # Active-force frames are sampled at 50 Hz and displayed at 1/8 speed.
            # 10 simulation frames x 4 copies / 25 fps = 1.6 s for a 0.20 s push.
            for _ in range(4 if slow_motion else 1):
                writer.append_data(frame)
                video_frames += 1
    finally:
        writer.close()
        env.eval_force = None
        env.scene.clear_debug_objects()
    metadata = dict(label=label, duration_s=duration, video_duration_s=video_frames / 25,
                    fps=25, playback_speed="1x; 0.125x while force is active",
                    force_n=force_n, force_duration_s=0.2, force_times_s=events,
                    arrow_scale_m_per_newton=0.01,
                    displacement_x=float(env.robot.get_pos()[0, 0].item()) - start_x,
                    displacement_reference="initial configured base x",
                    ever_fallen=any(s["fallen"] for s in samples), samples=samples)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"VIDEO_SAVED {output.name} elapsed={time.perf_counter()-started:.1f}s distance={metadata['displacement_x']:.2f}m fallen={metadata['ever_fallen']}", flush=True)
    return metadata
