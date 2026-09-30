import os
import re
import platform
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import imageio_ffmpeg

_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def _ff() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── DURATION ──────────────────────────────────────────────────────────────────

def get_video_duration(path: str) -> float:
    result = subprocess.run(
        [_ff(), "-i", path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )
    m = re.search(r"Duration:\s+(\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def get_video_framerate(path: str) -> float:
    """Detect source frame rate; returns a sensible value capped at 60."""
    result = subprocess.run(
        [_ff(), "-i", path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )
    # Match patterns like "29.97 fps", "60 fps", "25 tbr" etc.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)", result.stderr)
    if m:
        fps = float(m.group(1))
        if 10.0 <= fps <= 120.0:
            return min(fps, 60.0)
    return 25.0


def _safe_fps(fps: Optional[float]) -> float:
    try:
        val = float(fps or 0)
    except (TypeError, ValueError):
        val = 0.0
    if not (10.0 <= val <= 60.0):
        return 30.0
    return round(val, 3)


def _fps_filter(fps: Optional[float]) -> str:
    return f"fps={_safe_fps(fps):.3f},settb=AVTB"


def get_video_dimensions(path: str) -> tuple[int, int]:
    result = subprocess.run(
        [_ff(), "-i", path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )
    m = re.search(r"(\d{2,4})x(\d{2,4})", result.stderr)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1920, 1080


# ── BROWSER-SAFE PREVIEW PROXY ─────────────────────────────────────────────────

# Audio codecs HTML5 <video> can actually decode in Chrome/Firefox/Edge.
_BROWSER_SAFE_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}


def get_audio_codec(path: str) -> Optional[str]:
    """Best-effort audio codec name of the first audio stream (e.g. 'ac3', 'aac')."""
    result = subprocess.run(
        [_ff(), "-i", path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )
    m = re.search(r"Audio:\s*([a-zA-Z0-9_]+)", result.stderr)
    return m.group(1).lower() if m else None


def needs_audio_proxy(path: str) -> bool:
    codec = get_audio_codec(path)
    return codec is not None and codec not in _BROWSER_SAFE_AUDIO


def _pick_audio_stream(path: str) -> Optional[str]:
    """Stream specifier (e.g. '0:2') for the file's default audio track,
    falling back to the first audio track if none is flagged default."""
    result = subprocess.run(
        [_ff(), "-i", path],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )
    # Match whole "Stream #0:N(...): Audio: ..." lines so chapter markers
    # like "Chapter #0:1: start ..." (which also contain "#0:N") can't be
    # mistaken for stream lines.
    audio_specs = re.findall(r"Stream #(\d+:\d+)\([^)]*\):\s*Audio:[^\n]*", result.stderr)
    lines = re.findall(r"Stream #\d+:\d+\([^)]*\):\s*Audio:[^\n]*", result.stderr)
    if not audio_specs:
        return None
    for spec, line in zip(audio_specs, lines):
        if "(default)" in line:
            return spec
    return audio_specs[0]


def make_preview_proxy(src: str, dst: str) -> str:
    """Remux src into dst with video copied as-is and audio transcoded to AAC
    so browsers that can't decode AC3/E-AC3 (etc.) still get sound."""
    audio_spec = _pick_audio_stream(src)
    cmd = [_ff(), "-y", "-i", src, "-map", "0:v:0"]
    if audio_spec:
        cmd += ["-map", audio_spec, "-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", dst]
    subprocess.run(cmd, capture_output=True, check=True)
    return dst


# ── SEASON/EPISODE DETECTION ──────────────────────────────────────────────────

def detect_season_episode(filename: str) -> Optional[str]:
    name = Path(filename).stem
    # S01E02
    m = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", name)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    # 1x02
    m = re.search(r"(\d{1,2})[xXхХ](\d{1,2})", name)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    # Сезон N Серия M
    m = re.search(r"[Сс]езон\s*(\d+)[,.]?\s*[Сс]ери[яи]\s*(\d+)", name, re.I)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    # Season N Episode M
    m = re.search(r"[Ss]eason\s*(\d+)[,.]?\s*[Ee]pisode\s*(\d+)", name, re.I)
    if m:
        return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"
    return None


# ── THUMBNAIL ─────────────────────────────────────────────────────────────────

def extract_thumbnail(src: str, time: float, out: str) -> bool:
    try:
        subprocess.run(
            [_ff(), "-y", "-ss", str(max(0.0, time)), "-i", src,
             "-vframes", "1", "-q:v", "3", "-f", "image2", out],
            capture_output=True, timeout=30,
        )
        return Path(out).exists()
    except Exception:
        return False


# ── SUBTITLES ─────────────────────────────────────────────────────────────────

def _srt_time(sec: float) -> str:
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    ms  = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(clips: list[dict], segments_map: dict, trans_dur: float = 0.0) -> str:
    lines = []
    idx   = 1
    t_off = 0.0
    for clip in clips:
        sf   = clip.get("source_file", "")
        segs = segments_map.get(sf, [])
        cs, ce = clip["clip_start"], clip["clip_end"]
        for seg in segs:
            if seg["end"] < cs or seg["start"] > ce:
                continue
            out_s = t_off + max(0.0, seg["start"] - cs)
            out_e = t_off + min(ce - cs, seg["end"] - cs)
            if out_e > out_s:
                lines += [str(idx), f"{_srt_time(out_s)} --> {_srt_time(out_e)}", seg["text"], ""]
                idx += 1
        t_off += (ce - cs) + trans_dur
    return "\n".join(lines)


# ── CHAPTER MARKERS ───────────────────────────────────────────────────────────

def write_chapter_meta(clips: list[dict], path: str, trans_dur: float = 0.0):
    with open(path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n\n")
        t = 0
        for i, clip in enumerate(clips):
            dur_ms = int(clip["clip_duration"] * 1000)
            title  = clip.get("reason", f"Moment {i+1}")[:60]
            f.write("[CHAPTER]\nTIMEBASE=1/1000\n")
            f.write(f"START={t}\nEND={t + dur_ms}\n")
            f.write(f"title=Момент {i+1}: {title}\n\n")
            t += dur_ms + int(trans_dur * 1000)


# ── FILTER HELPERS ────────────────────────────────────────────────────────────

_COLOR_FILTERS = {
    "warm":    "curves=r='0/0 0.5/0.55 1/1':g='0/0 0.5/0.50 1/0.96':b='0/0 0.5/0.44 1/0.86'",
    "cool":    "curves=r='0/0 0.5/0.44 1/0.86':g='0/0 0.5/0.50 1/0.96':b='0/0 0.5/0.56 1/1'",
    "cinema":  "curves=all='0/0.04 0.25/0.28 0.75/0.78 1/0.96',eq=saturation=0.85:contrast=1.08",
    "vintage": "colorbalance=rs=0.15:gs=0.05:bs=-0.2:rm=0.05:bm=-0.05,eq=saturation=0.75",
}


def _font_spec() -> str:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            esc = fp.replace("\\", "/").replace(":", "\\:")
            return f"fontfile='{esc}':"
    return ""


def _drawtext(text: str, corner: str = "br", size: int = 36) -> str:
    esc = (text.replace("\\", "\\\\")
               .replace("'",  "\\'")
               .replace(":",  "\\:")
               .replace("%",  "\\%"))
    pos = {
        "br": "x=w-tw-20:y=h-th-20",
        "bl": "x=20:y=h-th-20",
        "tr": "x=w-tw-20:y=20",
        "tl": "x=20:y=20",
    }.get(corner, "x=w-tw-20:y=h-th-20")
    return (
        f"drawtext={_font_spec()}text='{esc}':fontsize={size}:"
        f"fontcolor=white:alpha=0.85:{pos}:"
        f"box=1:boxcolor=black@0.5:boxborderw=8"
    )


def _bridge_drawtext(text: str, duration: float = 4.0, size: int = 30) -> str:
    """Short context caption shown for the first `duration` s of a clip,
    explaining what was skipped since the previous clip."""
    esc = (text.replace("\\", "\\\\")
               .replace("'",  "\\'")
               .replace(":",  "\\:")
               .replace("%",  "\\%"))
    return (
        f"drawtext={_font_spec()}text='{esc}':fontsize={size}:"
        f"fontcolor=white:alpha=0.95:x=(w-tw)/2:y=40:"
        f"box=1:boxcolor=black@0.6:boxborderw=10:"
        f"enable='lt(t,{duration:.2f})'"
    )


def _scale_filter(preset: str) -> Optional[str]:
    if preset == "youtube":
        return "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    if preset == "square":
        return "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2"
    return None  # "original" or "shorts" handled separately


def _build_vf(opts: dict, clip_dur: float = 0.0, fps: Optional[float] = None) -> Optional[str]:
    parts = []
    scale = _scale_filter(opts.get("preset", "youtube"))
    if scale:
        parts.append(scale)

    # fade in (video)
    fi = float(opts.get("fade_in", 0))
    if fi > 0:
        parts.append(f"fade=t=in:st=0:d={fi:.2f}")

    if opts.get("flip"):
        parts.append("hflip")
    color = _COLOR_FILTERS.get(opts.get("color", "none"))
    if color:
        parts.append(color)

    # brightness / contrast / saturation
    br  = float(opts.get("brightness",  100))
    ct  = float(opts.get("contrast",    100))
    sat = float(opts.get("saturation",  100))
    eq_parts = []
    if abs(br  - 100) > 0.5: eq_parts.append(f"brightness={(br - 100)/100:.3f}")
    if abs(ct  - 100) > 0.5: eq_parts.append(f"contrast={ct/100:.3f}")
    if abs(sat - 100) > 0.5: eq_parts.append(f"saturation={sat/100:.3f}")
    if eq_parts:
        parts.append("eq=" + ":".join(eq_parts))

    speed = float(opts.get("speed", 1.0))
    if abs(speed - 1.0) > 0.001:
        parts.append(f"setpts=PTS/{speed:.4f}")
    parts.append("setpts=PTS-STARTPTS")  # reset timestamps to 0 for clean concat
    parts.append(_fps_filter(fps))
    parts.append("format=yuv420p")

    # fade out (video) — must come after format
    fo = float(opts.get("fade_out", 0))
    if fo > 0 and clip_dur > 0:
        st = max(0.0, clip_dur / max(speed, 0.1) - fo)
        parts.append(f"fade=t=out:st={st:.2f}:d={fo:.2f}")

    wm = opts.get("watermark_text", "").strip()
    if wm:
        parts.append(_drawtext(wm, corner=opts.get("watermark_corner", "br")))

    bridge = (opts.get("bridge_text") or "").strip()
    if bridge:
        parts.append(_bridge_drawtext(bridge))
    return ",".join(parts) if parts else "format=yuv420p"


def _build_af(opts: dict, clip_dur: float = 0.0) -> Optional[str]:
    """Return audio filter string, or None if no audio processing is needed."""
    parts: list[str] = []
    vol   = float(opts.get("volume",   100))
    speed = float(opts.get("speed",    1.0))
    fi    = float(opts.get("fade_in",  0))
    fo    = float(opts.get("fade_out", 0))

    if abs(vol - 100) > 0.5:
        parts.append(f"volume={vol/100:.3f}")
    if opts.get("normalize_audio"):
        parts.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    if fi > 0:
        parts.append(f"afade=t=in:st=0:d={fi:.3f}")
    if abs(speed - 1.0) > 0.001:
        s = speed
        while s > 2.0 + 1e-4:
            parts.append("atempo=2.0"); s /= 2.0
        while s < 0.5 - 1e-4:
            parts.append("atempo=0.5"); s /= 0.5
        parts.append(f"atempo={s:.6f}")
    if fo > 0 and clip_dur > 0:
        out_dur = clip_dur / max(speed, 0.1)
        st = max(0.0, out_dur - fo)
        parts.append(f"afade=t=out:st={st:.3f}:d={fo:.3f}")

    return ",".join(parts) if parts else None


def _final_audio_chain(
    base_af: Optional[str],
    *,
    pad: bool = True,
    duration: Optional[float] = None,
) -> str:
    """Normalize audio timestamps for stable clip joins."""
    parts: list[str] = []
    if base_af:
        parts.append(base_af)
    parts.extend(["aresample=async=1:first_pts=0", "asetpts=PTS-STARTPTS"])
    if pad:
        if duration and duration > 0:
            parts.append(f"apad=whole_dur={duration:.3f}")
        else:
            parts.append("apad")
    return ",".join(parts)


# ── BLUR / BLEEP HELPERS ──────────────────────────────────────────────────────

def _build_blur_chain(blurs: list, vw: int, vh: int) -> tuple:
    """Returns (fc_lines, output_label) for chained blur regions."""
    if not blurs:
        return [], None
    lines = []
    cur = "vbase"
    for i, blur in enumerate(blurs):
        x = max(0, int(float(blur.get("x", 0)) * vw))
        y = max(0, int(float(blur.get("y", 0)) * vh))
        w = max(4, int(float(blur.get("w", 0.2)) * vw))
        h = max(4, int(float(blur.get("h", 0.2)) * vh))
        w += w % 2; h += h % 2          # must be even
        x = min(x, max(0, vw - w));  y = min(y, max(0, vh - h))
        w = min(w, vw - x);          h = min(h, vh - y)
        s  = max(2, int(blur.get("strength", 15)))
        sh = blur.get("shape", "rect")
        nxt = f"vb{i}"

        blur_f = f"boxblur=luma_radius={s}:luma_power=2:chroma_radius={s}:chroma_power=2"

        if sh == "oval":
            cx, cy = max(1, w // 2), max(1, h // 2)
            geq = (
                f"r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='if(lte(pow((X-{cx}.0)/{cx}.0,2)"
                f"+pow((Y-{cy}.0)/{cy}.0,2),1),255,0)'"
            )
            lines += [
                f"[{cur}]split[sa{i}][sb{i}]",
                f"[sb{i}]crop={w}:{h}:{x}:{y},{blur_f},format=rgba,geq={geq}[br{i}]",
                f"[sa{i}][br{i}]overlay={x}:{y}[{nxt}]",
            ]
        elif sh == "poly":
            pts = blur.get("pts", [])
            if len(pts) >= 3:
                # pixel coords relative to crop box origin
                px = [float(p["x"]) * vw - x for p in pts]
                py = [float(p["y"]) * vh - y for p in pts]
                # ensure CCW winding
                area = sum(
                    px[k] * py[(k+1) % len(px)] - px[(k+1) % len(px)] * py[k]
                    for k in range(len(px))
                ) / 2
                if area < 0:
                    px, py = px[::-1], py[::-1]
                n = len(px)
                terms = []
                for k in range(n):
                    dx = px[(k+1) % n] - px[k];  dy = py[(k+1) % n] - py[k]
                    terms.append(f"gte({dx:.2f}*(Y-{py[k]:.2f})-{dy:.2f}*(X-{px[k]:.2f}),0)")
                alpha_expr = "*".join(f"({t})" for t in terms)
                geq = (
                    f"r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                    f"a='if(gte({alpha_expr},1),255,0)'"
                )
                lines += [
                    f"[{cur}]split[sa{i}][sb{i}]",
                    f"[sb{i}]crop={w}:{h}:{x}:{y},{blur_f},format=rgba,geq={geq}[br{i}]",
                    f"[sa{i}][br{i}]overlay={x}:{y}[{nxt}]",
                ]
            else:  # degenerate — fall back to rect
                lines += [
                    f"[{cur}]split[sa{i}][sb{i}]",
                    f"[sb{i}]crop={w}:{h}:{x}:{y},{blur_f}[br{i}]",
                    f"[sa{i}][br{i}]overlay={x}:{y}[{nxt}]",
                ]
        else:  # rect
            lines += [
                f"[{cur}]split[sa{i}][sb{i}]",
                f"[sb{i}]crop={w}:{h}:{x}:{y},{blur_f}[br{i}]",
                f"[sa{i}][br{i}]overlay={x}:{y}[{nxt}]",
            ]
        cur = nxt
    return lines, cur


def _build_bleep_chain(bleeps: list, base_af: Optional[str]) -> tuple:
    """Returns (fc_lines, audio_output_label) for bleep censoring."""
    lines = []
    mute_expr = "+".join(
        f"between(t,{float(b['s']):.3f},{float(b['e']):.3f})" for b in bleeps
    )
    if base_af:
        lines.append(f"[0:a]{base_af},volume=enable='{mute_expr}':volume=0[amuted]")
    else:
        lines.append(f"[0:a]volume=enable='{mute_expr}':volume=0[amuted]")
    for i, b in enumerate(bleeps):
        d  = max(0.05, float(b["e"]) - float(b["s"]))
        dm = int(float(b["s"]) * 1000)
        lines += [
            f"aevalsrc=0.5*sin(2*PI*1000*t):s=44100:c=stereo:d={d:.3f}[bt{i}]",
            f"[bt{i}]adelay={dm}|{dm}[bd{i}]",
        ]
    inp = "[amuted]" + "".join(f"[bd{i}]" for i in range(len(bleeps)))
    lines.append(f"{inp}amix=inputs={len(bleeps)+1}:normalize=0:duration=first[acen]")
    return lines, "acen"


def _cut_advanced(src: str, pre: float, fine: float, dur: float, out: str,
                  o: dict, bleeps: list, blurs: list, fps: float = 25.0) -> str:
    """cut_clip path when bleep censoring or blur regions are present."""
    vw, vh = get_video_dimensions(src)
    base_vf = _build_vf(o, dur, fps)
    base_af = _build_af(o, dur)
    out_dur = dur / max(float(o.get("speed", 1.0)), 0.1)

    fc: list[str] = [f"[0:v]{base_vf}[vbase]"]

    # blur chain
    blur_lines, blur_out = _build_blur_chain(blurs, vw, vh)
    fc.extend(blur_lines)
    video_out = blur_out or "vbase"

    # bleep / audio chain — apad at the end prevents AAC encoder from cutting last samples
    if bleeps:
        bleep_lines, bleep_out = _build_bleep_chain(bleeps, base_af)
        fc.extend(bleep_lines)
        fc.append(f"[{bleep_out}]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS,apad=whole_dur={out_dur:.3f}[aout]")
    elif base_af:
        fc.append(f"[0:a]{_final_audio_chain(base_af, duration=out_dur)}[aout]")
    else:
        fc.append(f"[0:a]{_final_audio_chain(None, duration=out_dur)}[aout]")
    audio_out = "aout"

    cmd = [_ff(), "-y",
           "-ss", str(pre), "-i", src,
           "-ss", str(fine),
           "-t", str(dur),
           "-filter_complex", ";".join(fc),
           "-map", f"[{video_out}]",
           "-map", "[aout]",
           "-fps_mode", "cfr",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-r", f"{_safe_fps(fps):.3f}", "-c:a", "aac", "-b:a", "128k",
           "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


# ── CUT ───────────────────────────────────────────────────────────────────────

def cut_clip(src: str, start: float, end: float, out: str, opts: Optional[dict] = None) -> str:
    o       = {**(opts or {})}
    per     = o.pop("_opts", {}) or {}
    o       = {**o, **{k: v for k, v in per.items() if v is not None}}

    bleeps  = [b for b in (o.pop("bleeps", None) or []) if b.get("e", 0) > b.get("s", 0)]
    blurs   = [b for b in (o.pop("blurs",  None) or []) if b.get("w", 0) > 0 and b.get("h", 0) > 0]

    preset  = o.get("preset", "youtube")
    pre     = max(0.0, start - 3.0)
    fine    = start - pre
    dur     = end - start
    speed   = float(o.get("speed", 1.0))

    fps = o.pop("_fps", None) or get_video_framerate(src)

    if preset == "shorts":
        return _cut_shorts(src, pre, fine, dur, speed, out, o, fps)

    if bleeps or blurs:
        return _cut_advanced(src, pre, fine, dur, out, o, bleeps, blurs, fps)

    vf = _build_vf(o, dur, fps)
    af = _build_af(o, dur)
    # Reset PTS and pad audio to video length; this prevents concat gaps and AAC tail loss.
    out_dur = dur / max(speed, 0.1)
    af_final = _final_audio_chain(af, duration=out_dur)

    cmd = [_ff(), "-y", "-ss", str(pre), "-i", src, "-ss", str(fine), "-t", str(dur),
           "-vf", vf, "-af", af_final,
           "-fps_mode", "cfr",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-r", f"{_safe_fps(fps):.3f}", "-c:a", "aac", "-b:a", "192k",
           "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


def _cut_shorts(src: str, pre: float, fine: float, dur: float, speed: float, out: str, o: dict, fps: float = 25.0) -> str:
    """9:16 output with blurred background."""
    extra = []
    if o.get("flip"):
        extra.append("hflip")
    color = _COLOR_FILTERS.get(o.get("color", "none"))
    if color:
        extra.append(color)
    # brightness / contrast / saturation (same as standard path)
    br  = float(o.get("brightness",  100))
    ct  = float(o.get("contrast",    100))
    sat = float(o.get("saturation",  100))
    eq_parts = []
    if abs(br  - 100) > 0.5: eq_parts.append(f"brightness={(br - 100)/100:.3f}")
    if abs(ct  - 100) > 0.5: eq_parts.append(f"contrast={ct/100:.3f}")
    if abs(sat - 100) > 0.5: eq_parts.append(f"saturation={sat/100:.3f}")
    if eq_parts:
        extra.append("eq=" + ":".join(eq_parts))
    if abs(speed - 1.0) > 0.001:
        extra.append(f"setpts=PTS/{speed:.4f}")
    wm = o.get("watermark_text", "").strip()
    if wm:
        extra.append(_drawtext(wm, corner=o.get("watermark_corner", "br")))

    extra_str = ("," + ",".join(extra)) if extra else ""

    # bridge text needs `t` relative to the clip start, so it must come after
    # setpts=PTS-STARTPTS resets the timeline, unlike the other `extra` filters above.
    post = ""
    bridge = (o.get("bridge_text") or "").strip()
    if bridge:
        post = "," + _bridge_drawtext(bridge)

    fc = (
        "[0:v]split=2[v1][v2];"
        "[v1]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=40[bg];"
        "[v2]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2{extra_str},{_fps_filter(fps)},setpts=PTS-STARTPTS,format=yuv420p{post}[vout]"
    )

    af_parts: list[str] = []
    vol = float(o.get("volume", 100))
    if abs(vol - 100) > 0.5:
        af_parts.append(f"volume={vol/100:.3f}")
    if o.get("normalize_audio"):
        af_parts.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    if abs(speed - 1.0) > 0.001:
        s = speed
        while s > 2.0 + 1e-4:
            af_parts.append("atempo=2.0"); s /= 2.0
        while s < 0.5 - 1e-4:
            af_parts.append("atempo=0.5"); s /= 0.5
        af_parts.append(f"atempo={s:.6f}")
    out_dur = dur / max(speed, 0.1)
    audio_chain = _final_audio_chain(",".join(af_parts) if af_parts else None, duration=out_dur)
    fc += ";" + "[0:a]" + audio_chain + "[aout]"

    cmd = [_ff(), "-y",
           "-ss", str(pre), "-i", src,
           "-ss", str(fine),
           "-t", str(dur),
           "-filter_complex", fc,
           "-map", "[vout]", "-map", "[aout]",
           "-fps_mode", "cfr",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-r", f"{_safe_fps(fps):.3f}", "-c:a", "aac", "-b:a", "128k",
           "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


# ── TRANSITION PREP ───────────────────────────────────────────────────────────

def prepare_transition(src: str, out: str, opts: dict) -> str:
    preset = opts.get("preset", "youtube")
    vf     = _scale_filter(preset) or "scale=1920:1080"
    fps    = _safe_fps(opts.get("_fps") or get_video_framerate(src))
    subprocess.run(
        [_ff(), "-y", "-i", src,
         "-vf", vf + f",{_fps_filter(fps)},setpts=PTS-STARTPTS,format=yuv420p",
         "-fps_mode", "cfr", "-r", f"{fps:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k",
         "-ar", "44100", "-ac", "2",
         "-movflags", "+faststart", out],
        capture_output=True, check=True,
    )
    return out


# ── CONCAT ────────────────────────────────────────────────────────────────────

def concat_clips(clips: list[str], out: str, opts: Optional[dict] = None,
                 transition_path: Optional[str] = None,
                 temp_dir: Optional[str] = None) -> str:
    if not clips:
        raise ValueError("No clips to concat")
    if len(clips) == 1 and not transition_path:
        shutil.copy(clips[0], out)
        return out

    if transition_path and temp_dir and len(clips) > 1:
        # transitions require filter_complex (mixed sources/params)
        order = []
        for i, clip in enumerate(clips):
            order.append(clip)
            if i < len(clips) - 1:
                t_copy = os.path.join(temp_dir, f"trans_{i:04d}.mp4")
                shutil.copy(transition_path, t_copy)
                order.append(t_copy)
        return _concat_filter(order, out)

    # No transitions: filter_complex ensures A/V sync (chunked to avoid input limits)
    return _concat_filter(clips, out)


def _concat_filter(clips: list[str], out: str, fps: Optional[float] = None) -> str:
    """Re-encode concat via filter_complex — for transitions. Chunked if > 8 inputs."""
    out_fps = _safe_fps(fps or (get_video_framerate(clips[0]) if clips else 30.0))
    CHUNK = 8
    if len(clips) > CHUNK:
        # Use a dedicated temp dir so chunk files never appear in the outputs folder
        # and are cleaned up by the OS even if our process crashes.
        with tempfile.TemporaryDirectory(prefix="cfchunk_") as tmp_dir:
            chunk_outs: list[str] = []
            for idx in range(0, len(clips), CHUNK):
                batch = clips[idx:idx + CHUNK]
                chunk_path = os.path.join(tmp_dir, f"_chunk_{idx:04d}.mp4")
                _concat_filter(batch, chunk_path, out_fps)
                chunk_outs.append(chunk_path)
            return _concat_filter(chunk_outs, out, out_fps)

    n      = len(clips)
    inputs = []
    for p in clips:
        inputs += ["-i", p]
    # Normalize timestamps again at concat time. This also protects old temp
    # clips made before per-clip audio PTS reset was added.
    prep: list[str] = []
    streams: list[str] = []
    for i in range(n):
        prep.append(f"[{i}:v:0]setpts=PTS-STARTPTS,{_fps_filter(out_fps)}[v{i}]")
        prep.append(f"[{i}:a:0]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a{i}]")
        streams.append(f"[v{i}][a{i}]")
    fc = ";".join(prep + [f"{''.join(streams)}concat=n={n}:v=1:a=1[vout][aout]"])
    proc = subprocess.run([
        _ff(), "-y", *inputs,
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", f"{out_fps:.3f}",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", out,
    ], capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg concat failed (rc={proc.returncode}):\n{stderr}")
    return out


# ── PROCESS VIDEO ─────────────────────────────────────────────────────────────

def _dedup_clips(clips: list[dict]) -> list[dict]:
    """Last-resort dedup before rendering: remove clips whose clip_start/clip_end
    overlap more than 30% with a higher-scored clip from the same source file.
    Preserve frontend timeline order for editor/reorder workflows."""
    if len(clips) <= 1:
        return clips
    ranked = sorted(enumerate(clips), key=lambda item: item[1].get("score", 0), reverse=True)
    kept: list[tuple[int, dict]] = []
    for original_idx, c in ranked:
        cs, ce = c.get("clip_start", 0), c.get("clip_end", 0)
        cd = max(1.0, ce - cs)
        same_src = [k for _, k in kept if k.get("source_file") == c.get("source_file")]
        overlap = max(
            (min(ce, k["clip_end"]) - max(cs, k["clip_start"])) / cd
            for k in same_src
        ) if same_src else 0.0
        if overlap < 0.30:
            kept.append((original_idx, c))
    kept.sort(key=lambda item: item[0])
    return [c for _, c in kept]


def process_video(
    clips: list[dict],
    out: str,
    temp_dir: str,
    opts: Optional[dict] = None,
    segments_map: Optional[dict] = None,
    on_progress=None,
) -> str:
    o = opts or {}
    clips = _dedup_clips(clips)  # safety net: remove any duplicates before cutting
    total = len(clips)
    cut_paths = [os.path.join(temp_dir, f"clip_{i:04d}.mp4") for i in range(total)]
    done_count = [0]

    # Pre-probe FPS per unique source file — avoids N redundant ffmpeg probes
    fps_cache: dict[str, float] = {}
    for clip in clips:
        src = clip.get("source_file") or clip.get("src", "")
        if src and src not in fps_cache:
            fps_cache[src] = get_video_framerate(src)

    def _cut_one(i):
        clip = clips[i]
        src  = clip.get("source_file") or clip.get("src", "")
        clip_opts = {**o, "_opts": clip.get("_opts") or {}, "_fps": fps_cache.get(src)}
        if clip.get("bridge_text"):
            clip_opts["bridge_text"] = clip["bridge_text"]
        cut_clip(src, clip["clip_start"], clip["clip_end"], cut_paths[i], clip_opts)
        done_count[0] += 1
        if on_progress:
            on_progress(
                f"Cutting clip {done_count[0]}/{total}",
                int(64 + done_count[0] / total * 22),
            )

    cpu = os.cpu_count() or 4
    workers = min(max(cpu // 2, 2), 4, total)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_cut_one, i): i for i in range(total)}
        for f in as_completed(futures):
            f.result()  # re-raise any exception

    # subtitles
    srt_path = None
    if o.get("subtitles") and segments_map:
        trans_dur = 0.0
        if o.get("transition_path"):
            trans_dur = get_video_duration(o["transition_path"])
        srt_content = generate_srt(clips, segments_map, trans_dur)
        if srt_content.strip():
            srt_path = os.path.join(temp_dir, "subs.srt")
            Path(srt_path).write_text(srt_content, encoding="utf-8")

    if on_progress:
        on_progress(f"Merging {len(cut_paths)} clips…", 87)

    merged = os.path.join(temp_dir, "merged.mp4")
    concat_clips(cut_paths, merged, o,
                 transition_path=o.get("transition_path"),
                 temp_dir=temp_dir)

    # chapters + subtitles pass
    needs_extra = o.get("chapters") or srt_path
    if needs_extra:
        if on_progress:
            on_progress("Adding chapters & subtitles…", 93)
        extra_inputs = []
        map_meta_idx = None

        if o.get("chapters"):
            meta_path = os.path.join(temp_dir, "chapters.txt")
            trans_dur = get_video_duration(o["transition_path"]) if o.get("transition_path") else 0.0
            write_chapter_meta(clips, meta_path, trans_dur)
            extra_inputs += ["-i", meta_path]
            map_meta_idx = 1

        vf_subs = None
        if srt_path:
            esc = srt_path.replace("\\", "/")
            # escape drive colon on Windows
            esc = re.sub(r"^([A-Za-z]):", r"\1\\:", esc)
            vf_subs = f"subtitles='{esc}'"

        cmd = [_ff(), "-y", "-i", merged, *extra_inputs]
        if map_meta_idx is not None:
            cmd += ["-map_metadata", str(map_meta_idx)]
        if vf_subs:
            cmd += ["-vf", vf_subs]
        fps = _safe_fps(get_video_framerate(merged))
        cmd += ["-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-r", f"{fps:.3f}",
                "-c:a", "copy", "-movflags", "+faststart", out]
        subprocess.run(cmd, capture_output=True, check=True)
    else:
        shutil.move(merged, out)

    if on_progress:
        on_progress("Done!", 100)

    return out


def render_quality_report(path: str, clips: list[dict]) -> dict:
    """Lightweight QA for the rendered output."""
    report = {"duration": 0.0, "expected_duration": 0.0, "issues": []}
    actual = get_video_duration(path)
    expected = sum(float(c.get("clip_duration", 0)) for c in clips)
    report["duration"] = round(actual, 3)
    report["expected_duration"] = round(expected, 3)

    if expected > 0 and actual + 1.0 < expected * 0.97:
        report["issues"].append({
            "type": "duration_short",
            "message": f"Rendered duration {actual:.2f}s is shorter than expected {expected:.2f}s",
        })

    try:
        proc = subprocess.run(
            [
                _ff(), "-hide_banner", "-nostats", "-loglevel", "info",
                "-i", path, "-vn",
                "-af", "silencedetect=noise=-45dB:duration=0.45",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=120,
        )
        spans: list[tuple[float, float]] = []
        cur: Optional[float] = None
        for line in proc.stderr.splitlines():
            m = _SILENCE_START_RE.search(line)
            if m:
                cur = float(m.group(1))
            m = _SILENCE_END_RE.search(line)
            if m and cur is not None:
                spans.append((cur, float(m.group(1))))
                cur = None

        seam = 0.0
        for i, clip in enumerate(clips[:-1], start=1):
            seam += float(clip.get("clip_duration", 0))
            for ss, se in spans:
                if ss <= seam <= se and se - ss >= 0.7:
                    report["issues"].append({
                        "type": "seam_silence",
                        "clip": i,
                        "time": round(seam, 3),
                        "message": f"Long silence around join after clip {i}",
                    })
                    break

        for ss, se in spans:
            if actual > 0 and se >= actual - 0.15 and se - ss >= 0.8:
                report["issues"].append({
                    "type": "tail_silence",
                    "time": round(ss, 3),
                    "message": "Long silence at the end of rendered video",
                })
                break
    except Exception as exc:
        report["issues"].append({"type": "qa_failed", "message": str(exc)})

    return report
