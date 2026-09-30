import os
import re
import json
import hashlib
import subprocess
import bisect
import httpx
import imageio_ffmpeg
from pathlib import Path
from typing import Optional

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

DEEPSEEK_URL        = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL      = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 8192   # deepseek-chat supports up to 8192 output tokens

STYLE_DESCRIPTIONS = {
    "highlights":    "funny, surprising, or emotionally impactful moments",
    "entertainment": "jokes, reactions, energetic exchanges, and memorable quotes",
    "gaming":        "impressive plays, funny fails, hype moments, and commentary peaks",
    "educational":   "key insights, important explanations, and memorable takeaways",
    "condensed":     "condensed retelling — preserves the complete story, cuts only filler",
}

# Categories DeepSeek assigns to moments — used for style-based filtering in select_moments
MOMENT_CATEGORIES = ("funny", "reaction", "emotional", "educational", "dramatic", "general")

# Which categories are prioritised for each style. Moments outside this set
# get a heavy score penalty when the style is active.
_STYLE_PREF_CATS: dict[str, set[str]] = {
    "highlights":    {"funny", "reaction", "emotional", "educational", "dramatic", "general"},
    "entertainment": {"funny", "reaction"},
    "gaming":        {"reaction", "dramatic", "funny"},
    "educational":   {"educational"},
    "condensed":     {"funny", "reaction", "emotional", "educational", "dramatic", "general"},
}

# For condensed mode the scoring threshold is lower — we want to include most story content.
_STYLE_MIN_SCORE: dict[str, float] = {
    "condensed": 5.0,
}

MIN_CLIP_SEC      = 8.0
MERGE_GAP         = 12.0   # max gap (s) between moments to merge into one clip
SCENE_HEADROOM    = 2.0    # default pre-roll before AI timestamps
SCENE_TAILROOM    = 3.5    # default post-roll after AI timestamps
BOUNDARY_START_PAD = 0.18   # keep a little audio before the first spoken word
BOUNDARY_END_PAD   = 0.35   # keep a little audio after the last spoken word
SENTENCE_TAIL_PAD  = 0.65   # let the last word/sentence breathe before cutting
_CHUNK_DURATION   = 30 * 60  # analyse in 30-min windows for long videos
_CHUNK_OVERLAP    =  2 * 60  # 2-min overlap so boundary moments aren't missed
MIN_PAUSE         = 0.4     # segment-level sentence boundary threshold
MIN_PAUSE_RECORD  = 0.10   # min gap to include in pause list (avoids breath micro-gaps)
MIN_PAUSE_PHRASE  = 0.18    # min gap to be a valid cut point
MIN_PAUSE_NATURAL = 0.45    # preferred natural pause
MIN_PAUSE_TOPIC   = 0.85    # strong topic/scene boundary

_Q_PAT            = re.compile(r'[?？]\s*[»"\'"]?\s*$')
_SILENCE_START_RE = re.compile(r'silence_start:\s*([\d.]+)')
_SILENCE_END_RE   = re.compile(r'silence_end:\s*([\d.]+)')
_CACHE_DIR: Optional[Path] = None


# ── CACHE ─────────────────────────────────────────────────────────────────────

def set_cache_dir(path: Path):
    global _CACHE_DIR
    _CACHE_DIR = path
    path.mkdir(parents=True, exist_ok=True)


def _recover_partial_moments(raw: str) -> list[dict]:
    """
    Extract all complete moment objects from a truncated JSON response.
    Used when DeepSeek hits max_tokens mid-response and the outer array is broken.
    Matches flat JSON objects — moment entries never have nested {} so this is safe.
    """
    pat = re.compile(r'\{(?:"(?:[^"\\]|\\.)*"|[^"{}])*\}')
    moments = []
    for m in pat.finditer(raw):
        try:
            obj = json.loads(m.group())
            if "start" in obj and "end" in obj:
                moments.append(obj)
        except json.JSONDecodeError:
            continue
    return moments


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_cached(key: str, suffix: str):
    if not _CACHE_DIR:
        return None
    p = _CACHE_DIR / f"{key}{suffix}"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cached(key: str, suffix: str, data):
    if not _CACHE_DIR:
        return
    (_CACHE_DIR / f"{key}{suffix}").write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            default=lambda x: float(x) if hasattr(x, "item") else str(x),
        ),
        encoding="utf-8",
    )


# ── VISUAL ANALYSIS ───────────────────────────────────────────────────────────

# Pass-1 (I-frame) MAFD thresholds — frames are 1-4 s apart
_VIS_CUT_MAFD   = 40.0   # hard scene change
_VIS_STILL_MAFD = 10.0   # very similar → calm region
_VIS_HOT_MAFD   = 80.0   # heavy action → avoid cutting here

# Pass-2 (1fps) MAFD thresholds — frames are exactly 1 s apart
# Calibrated for talking-head content: normal speech is MAFD 5-25, large gestures 20-40.
# Scene changes are MAFD 50+. Dissolves are rare in YT content and span 3+ seconds.
_FPS1_CUT_MAFD      = 40.0   # was 22 — 40+ at 1fps = real cut, not expression/laugh
_FPS1_DISSOLVE_MIN  = 20.0   # was 7  — avoids treating speech motion as dissolve
_FPS1_DISSOLVE_MAX  = 50.0   # was 28 — real dissolves can reach higher MAFD
_FPS1_DISSOLVE_SECS =  3.0   # was 1.5 — real film dissolves last 3+ seconds

_VIS_W, _VIS_H = 80, 45  # thumbnail size for MAFD computation

_SHOWINFO_PTS_RE = re.compile(r'pts_time:([\d.]+)')
_VISUAL_INDEX_CACHE: dict[int, dict] = {}


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _mafd_pass(video_path: str, vf: str, timeout: int) -> tuple[list, list]:
    """Run one ffmpeg grayscale pass, return (times[], scores[[t,mafd]])."""
    W, H       = _VIS_W, _VIS_H
    frame_size = W * H
    cmd = [
        _ffmpeg(),
        "-hide_banner", "-nostats", "-loglevel", "info",
        "-i", video_path,
        "-vf", vf,
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]
    proc   = subprocess.run(cmd, capture_output=True, timeout=timeout)
    stderr = proc.stderr.decode("utf-8", errors="replace")
    times  = [float(m.group(1)) for m in _SHOWINFO_PTS_RE.finditer(stderr)]
    raw    = proc.stdout
    n      = len(raw) // frame_size
    frames = [raw[i * frame_size : (i + 1) * frame_size] for i in range(n)]
    count  = min(len(times), len(frames))
    if count < 2:
        return [], []
    scores: list[list] = []
    for i in range(1, count):
        f1, f2 = frames[i - 1], frames[i]
        if _HAS_NUMPY:
            a    = _np.frombuffer(f1, dtype=_np.uint8).astype(_np.int16)
            b    = _np.frombuffer(f2, dtype=_np.uint8).astype(_np.int16)
            mafd = float(_np.mean(_np.abs(a - b)))
        else:
            mafd = sum(abs(x - y) for x, y in zip(f1, f2)) / frame_size
        scores.append([times[i], round(mafd, 3)])
    return times, scores


def extract_visual_data(video_path: str) -> dict:
    """
    Two-pass visual analysis:
      Pass 1: I-frames only  (-skip_frame nokey) — hard scene cuts, hot/still regions
      Pass 2: 1 fps decode   — catches dissolves, fades, cuts missed between keyframes
    Both passes run in parallel via ThreadPoolExecutor.
    Returns {
        "scene_cuts":    [t, ...],          # hard cuts (both passes merged)
        "still_moments": [t, ...],          # calm regions (I-frame pass)
        "hot_moments":   [t, ...],          # heavy action (I-frame pass)
        "dissolves":     [[s,e], ...],      # gradual transition regions (1fps pass)
        "kf_scores":     [[t,mafd], ...],   # I-frame raw scores
        "fps1_scores":   [[t,mafd], ...],   # 1fps raw scores
        "silence_spans": [[s,e], ...],      # audio silence regions
    }
    Falls back to empty lists on any error.
    """
    import concurrent.futures

    _EMPTY = {
        "scene_cuts": [], "still_moments": [], "hot_moments": [],
        "dissolves": [], "kf_scores": [], "fps1_scores": [], "silence_spans": [],
    }

    try:
        vf1 = f"scale={_VIS_W}:{_VIS_H},showinfo"
        vf2 = f"fps=1,scale={_VIS_W}:{_VIS_H},showinfo"

        # Pass 1 needs a special flag — can't go through _mafd_pass directly
        def _pass1():
            W, H       = _VIS_W, _VIS_H
            frame_size = W * H
            cmd = [
                _ffmpeg(),
                "-hide_banner", "-nostats", "-loglevel", "info",
                "-skip_frame", "nokey",
                "-i", video_path,
                "-vf", vf1,
                "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
            ]
            proc   = subprocess.run(cmd, capture_output=True, timeout=120)
            stderr = proc.stderr.decode("utf-8", errors="replace")
            times  = [float(m.group(1)) for m in _SHOWINFO_PTS_RE.finditer(stderr)]
            raw    = proc.stdout
            n      = len(raw) // frame_size
            frames = [raw[i * frame_size : (i + 1) * frame_size] for i in range(n)]
            count  = min(len(times), len(frames))
            if count < 2:
                return [], []
            scores: list[list] = []
            for i in range(1, count):
                f1, f2 = frames[i - 1], frames[i]
                if _HAS_NUMPY:
                    a    = _np.frombuffer(f1, dtype=_np.uint8).astype(_np.int16)
                    b    = _np.frombuffer(f2, dtype=_np.uint8).astype(_np.int16)
                    mafd = float(_np.mean(_np.abs(a - b)))
                else:
                    mafd = sum(abs(x - y) for x, y in zip(f1, f2)) / frame_size
                scores.append([times[i], round(mafd, 3)])
            return times, scores

        def _pass2():
            return _mafd_pass(video_path, vf2, timeout=300)

        def _audio():
            try:
                scmd = [
                    _ffmpeg(),
                    "-hide_banner", "-nostats", "-loglevel", "info",
                    "-i", video_path, "-vn",
                    "-af", "silencedetect=noise=-35dB:duration=0.15",
                    "-f", "null", "-",
                ]
                sp = subprocess.run(scmd, capture_output=True, text=True, timeout=120)
                spans: list[list] = []
                cur_s: Optional[float] = None
                for line in sp.stderr.splitlines():
                    m = _SILENCE_START_RE.search(line)
                    if m:
                        cur_s = float(m.group(1))
                    m = _SILENCE_END_RE.search(line)
                    if m and cur_s is not None:
                        spans.append([cur_s, float(m.group(1))])
                        cur_s = None
                return spans
            except Exception:
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f1_fut = ex.submit(_pass1)
            f2_fut = ex.submit(_pass2)
            au_fut = ex.submit(_audio)
            _, kf_scores      = f1_fut.result()
            _, fps1_scores    = f2_fut.result()
            silence_spans     = au_fut.result()

        # ── I-frame results ───────────────────────────────────────────────────
        scene_cuts    = [r[0] for r in kf_scores if r[1] >= _VIS_CUT_MAFD]
        still_moments = [r[0] for r in kf_scores if r[1] <= _VIS_STILL_MAFD]
        hot_moments   = [r[0] for r in kf_scores if r[1] >= _VIS_HOT_MAFD]

        # ── Merge 1fps cuts (missed between keyframes) ────────────────────────
        for t, mafd in fps1_scores:
            if mafd >= _FPS1_CUT_MAFD:
                if not any(abs(t - sc) < 1.5 for sc in scene_cuts):
                    scene_cuts.append(t)
        scene_cuts.sort()

        # ── Dissolve / fade detection from 1fps sustained elevation ──────────
        # A dissolve spans 1.5+ seconds where MAFD stays in [7, 28].
        # Cutting in the middle of a dissolve looks bad.
        dissolves: list[list] = []
        in_diss    = False
        diss_start = 0.0
        for i, (t, mafd) in enumerate(fps1_scores):
            if _FPS1_DISSOLVE_MIN <= mafd <= _FPS1_DISSOLVE_MAX:
                if not in_diss:
                    in_diss, diss_start = True, t
            else:
                if in_diss and (t - diss_start) >= _FPS1_DISSOLVE_SECS:
                    dissolves.append([diss_start, t])
                in_diss = False
        if in_diss and fps1_scores:
            t_last = fps1_scores[-1][0]
            if (t_last - diss_start) >= _FPS1_DISSOLVE_SECS:
                dissolves.append([diss_start, t_last])

        return {
            "scene_cuts":    scene_cuts,
            "still_moments": still_moments,
            "hot_moments":   hot_moments,
            "dissolves":     dissolves,
            "kf_scores":     kf_scores,
            "fps1_scores":   fps1_scores,
            "silence_spans": silence_spans,
        }
    except Exception:
        return _EMPTY


def _visual_index(visual: dict) -> dict:
    key = id(visual)
    cached = _VISUAL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx = {
        "scene_cuts": sorted(float(x) for x in visual.get("scene_cuts", []) or []),
        "still_moments": sorted(float(x) for x in visual.get("still_moments", []) or []),
        "hot_moments": sorted(float(x) for x in visual.get("hot_moments", []) or []),
        "silence_spans": sorted((float(s), float(e)) for s, e in visual.get("silence_spans", []) or []),
        "dissolves": sorted((float(s), float(e)) for s, e in visual.get("dissolves", []) or []),
    }
    _VISUAL_INDEX_CACHE[key] = idx
    return idx


def _near_values(values: list[float], t: float, radius: float) -> list[float]:
    if not values:
        return []
    lo = bisect.bisect_left(values, t - radius)
    hi = bisect.bisect_right(values, t + radius)
    return values[lo:hi]


def _visual_score_bonus(t: float, visual: Optional[dict]) -> int:
    """
    Combined audio + visual bonus for a candidate cut at time t.

    Audio (silence spans — ground truth from FFmpeg silencedetect):
      +14: t falls inside a detected silence span
      +10: t within 0.15 s of silence boundary
      +6:  t within 0.4 s of silence midpoint

    Visual (I-frame MAFD + 1fps MAFD):
      +15: within 0.5 s of hard scene cut (decays linearly)
      +8:  within 1.5 s of hard scene cut
      +5:  within 1.5 s of calm/still region
      -10: within 1.5 s of heavy action, no other reward
      -12: t falls inside a detected dissolve/fade region (cutting mid-dissolve looks bad)
    """
    if not visual:
        return 0

    vidx = _visual_index(visual)
    scene_cuts    = vidx["scene_cuts"]
    still_moments = vidx["still_moments"]
    hot_moments   = vidx["hot_moments"]
    silence_spans = vidx["silence_spans"]
    dissolves     = vidx["dissolves"]

    bonus = 0

    # ── Dissolve penalty (applied first — overrides bonuses for mid-dissolve cuts)
    for ds, de in dissolves:
        if ds > t + 1.5:
            break
        if de < t - 1.5:
            continue
        if ds <= t <= de:
            bonus = -12
            break

    # ── Audio silence bonus (highest priority — true signal silence) ─────────
    for ss, se in silence_spans:
        if ss > t + 0.4:
            break
        if se < t - 0.4:
            continue
        if ss <= t <= se:
            bonus = max(bonus, 14)
            break
        d_edge = min(abs(t - ss), abs(t - se))
        d_mid  = abs(t - (ss + se) / 2)
        if d_edge <= 0.15:
            bonus = max(bonus, 10)
        elif d_mid <= 0.4:
            bonus = max(bonus, 6)

    # ── Visual scene-cut bonus (decays linearly with distance) ───────────────
    for sc in _near_values(scene_cuts, t, 1.5):
        d = abs(t - sc)
        if d <= 0.5:
            bonus = max(bonus, int(15 * (1.0 - d / 0.5)))
        elif d <= 1.5:
            bonus = max(bonus, int(8 * (1.0 - (d - 0.5) / 1.0)))

    # ── Still-region bonus ───────────────────────────────────────────────────
    if bonus < 5:
        for st in _near_values(still_moments, t, 1.5):
            if abs(t - st) <= 1.5:
                bonus = max(bonus, 5)
                break

    # ── Hot-moment penalty (only when nothing better was found) ──────────────
    if bonus <= 0:
        for hm in _near_values(hot_moments, t, 1.5):
            if abs(t - hm) <= 1.5:
                bonus = min(bonus, -10)
                break

    return max(-12, min(20, bonus))


# ── AI ANALYSIS ───────────────────────────────────────────────────────────────

NARRATIVE_ROLES = frozenset(("intro", "setup", "development", "climax", "resolution"))

# Effective-score bonus for narrative roles that preserve story coherence.
# intro/setup/resolution are routinely under-scored by DeepSeek because they're
# "less exciting" individually — but removing them destroys the narrative arc.
_NARRATIVE_BONUS: dict[str, float] = {
    "intro":       2.0,
    "setup":       1.5,
    "resolution":  1.5,
    "development": 0.5,
    "climax":      0.0,
}


async def map_story_beats(
    segments: list[dict],
    api_key: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> list[dict]:
    """
    Phase-1 of story-first modes.
    Ask DeepSeek to read the transcript and extract essential story beats
    that together define the complete narrative arc.
    Returns [{id, time, description, importance}] sorted by time.
    importance: "critical" | "major" | "minor"
    """
    if cache_key:
        cached = load_cached(cache_key, "_story_beats_v2.json")
        if cached is not None:
            return cached

    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return []

    if not segments:
        return []

    total_sec = segments[-1]["end"] - segments[0]["start"]
    if total_sec > _CHUNK_DURATION:
        t_start = segments[0]["start"]
        t_end = segments[-1]["end"]
        ranges: list[tuple[float, float]] = []
        t = t_start
        while t < t_end:
            ce = min(t + _CHUNK_DURATION, t_end)
            ranges.append((t, ce))
            if ce >= t_end:
                break
            t = ce - _CHUNK_OVERLAP

        merged: list[dict] = []
        for idx, (cs, ce) in enumerate(ranges):
            chunk_segs = [s for s in segments if s["start"] < ce + 5 and s["end"] > cs - 5]
            if not chunk_segs:
                continue
            chunk_cache = f"{cache_key}_sbck{idx}" if cache_key else None
            merged.extend(await map_story_beats(chunk_segs, api_key=key, cache_key=chunk_cache))

        merged.sort(key=lambda b: b.get("time", 0))
        deduped: list[dict] = []
        for beat in merged:
            bt = beat.get("time", 0)
            desc = beat.get("description", "").strip().lower()
            duplicate = any(
                abs(bt - prev.get("time", 0)) < 45
                or (desc and desc == prev.get("description", "").strip().lower())
                for prev in deduped
            )
            if not duplicate:
                deduped.append(beat)
        for i, beat in enumerate(deduped, 1):
            beat["id"] = i
        if cache_key and deduped:
            save_cached(cache_key, "_story_beats_v2.json", deduped)
        return deduped

    transcript = _to_text(segments)
    total_min  = segments[-1]["end"] / 60 if segments else 0

    prompt = f"""You are a story analyst. Read the following transcript ({total_min:.1f} min) and extract the 6-14 most essential story beats that define the complete narrative arc.

A "story beat" is a moment when something important changes:
- The situation/context is established
- A key revelation, twist, or secret uncovered
- A character makes an important decision
- The main conflict escalates or resolves
- An emotional turning point
- A pivotal exchange that changes the relationship between characters
- A setup pays off later, or a later event would be confusing without it

TRANSCRIPT:
{transcript}

For each beat, output:
  id          : integer 1..N
  time        : integer seconds (when this beat occurs)
  description : one sentence — what happens and why it matters to the story
  importance  : "critical" (must be shown), "major" (very helpful), or "minor" (adds depth)

Return ONLY valid JSON:
{{"story_beats": [{{"id": <int>, "time": <int_seconds>, "description": "<one sentence>", "importance": "critical|major|minor"}}]}}"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "You are a narrative analyst. Identify story beats precisely. "
                        "Output valid JSON only — no markdown, no explanation."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": DEEPSEEK_MAX_TOKENS,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()

    try:
        raw   = resp.json()["choices"][0]["message"]["content"]
        data  = json.loads(raw)
        beats = []
        for b in data.get("story_beats", []):
            beats.append({
                "id":          int(b["id"]),
                "time":        float(b["time"]),
                "description": str(b.get("description", "")),
                "importance":  str(b.get("importance", "major")).lower(),
            })
        beats.sort(key=lambda x: x["time"])
    except Exception:
        beats = []

    if cache_key and beats:
        save_cached(cache_key, "_story_beats_v2.json", beats)
    return beats


async def _analyze_chunked(
    segments: list[dict],
    vid_duration: float,
    target_minutes: int,
    unlimited: bool,
    style: str,
    api_key: Optional[str],
    cache_key: Optional[str],
    force_reanalyze: bool,
    story_beats: Optional[list],
    montage_profile: str = "auto",
) -> list[dict]:
    """
    Analyse a long transcript by splitting into overlapping _CHUNK_DURATION windows.
    Each chunk is cached independently so re-runs skip already-analysed chunks.
    """
    t_start = segments[0]["start"]
    t_end   = segments[-1]["end"]

    ranges: list[tuple[float, float]] = []
    t = t_start
    while t < t_end:
        chunk_end = min(t + _CHUNK_DURATION, t_end)
        ranges.append((t, chunk_end))
        if chunk_end >= t_end:
            break
        t = chunk_end - _CHUNK_OVERLAP

    all_moments: list[dict] = []
    for idx, (cs, ce) in enumerate(ranges):
        # 5-second slop so words at chunk edges aren't lost
        chunk_segs = [s for s in segments if s["start"] < ce + 5 and s["end"] > cs - 5]
        if not chunk_segs:
            continue

        chunk_dur    = ce - cs
        chunk_target = max(2, round(target_minutes * chunk_dur / vid_duration))
        chunk_cache  = f"{cache_key}_ck{idx}" if cache_key else None
        chunk_beats = None
        if story_beats:
            chunk_beats = [
                b for b in story_beats
                if cs - 60 <= float(b.get("time", 0)) <= ce + 60
            ] or None

        moments = await analyze_transcript(
            chunk_segs, chunk_target, unlimited, style,
            api_key, chunk_cache, force_reanalyze, chunk_beats, montage_profile,
        )
        all_moments.extend(moments)

    if style == "condensed":
        return all_moments
    return _deduplicate_moments(all_moments)


async def analyze_transcript(
    segments: list[dict],
    target_minutes: int = 20,
    unlimited: bool = False,
    style: str = "highlights",
    api_key: Optional[str] = None,
    cache_key: Optional[str] = None,
    force_reanalyze: bool = False,
    story_beats: Optional[list] = None,
    montage_profile: str = "auto",
) -> list[dict]:
    # v8: long-video compression budget + stricter condensed prompts
    adapt_tag    = "_adapt" if story_beats else ""
    profile_tag = f"_{montage_profile}" if montage_profile and montage_profile != "auto" else ""
    settings_tag = f"_ai_{style}{profile_tag}_t{target_minutes}_v8{adapt_tag}_{'unl' if unlimited else 'lim'}"
    if cache_key and not force_reanalyze:
        cached = load_cached(cache_key, f"{settings_tag}.json")
        if cached is not None:
            return cached

    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DeepSeek API key not set")

    # Auto-chunk transcripts longer than _CHUNK_DURATION
    vid_start = segments[0]["start"] if segments else 0
    total_sec = (segments[-1]["end"] if segments else 0) - vid_start
    if total_sec > _CHUNK_DURATION:
        return await _analyze_chunked(
            segments, total_sec, target_minutes, unlimited, style,
            api_key, cache_key, force_reanalyze, story_beats, montage_profile,
        )

    total_sec_abs = segments[-1]["end"] if segments else 0
    prompt    = _build_prompt(_to_text(segments), target_minutes, total_sec_abs, style, unlimited,
                              story_beats=story_beats, montage_profile=montage_profile)

    _system_msgs = {
        "highlights": (
            "You are an elite video editor creating viral highlight reels. "
            "Your standard is brutally high — only the top 10-15% of content earns a score ≥7. "
            "Pick the absolute best moments: funny, surprising, emotional, or impressive."
        ),
        "entertainment": (
            "You are a comedy specialist creating viral funny clips. "
            "Find moments that make viewers genuinely laugh, react with surprise, or feel entertained. "
            "Comedy timing matters: include the setup line or reaction beat when it is needed for the punchline to work. "
            "Serious, educational, or dramatic moments get score ≤4 unless they are direct setup for a joke."
        ),
        "gaming": (
            "You are a gaming highlights editor specializing in viral clips. "
            "Find clutch plays, epic fails, hype reactions, big wins, and funny commentary. "
            "Skip all routine gameplay, farming, loading, or exposition."
        ),
        "educational": (
            "You are an educational content curator extracting key insights. "
            "Find only the clearest explanations, counterintuitive facts, 'aha moment' reveals, and memorable analogies. "
            "Skip entertainment, jokes, introductions, and repetition."
        ),
        "condensed": (
            "You are a professional editor creating a condensed version of this video. "
            "Your job is to KEEP everything that carries story value and CUT only pure filler. "
            "Score 5+ = include. Include the beginning, middle and end of the video. "
            "This is NOT a highlights reel — it is a complete condensed retelling."
        ),
    }
    sys_msg = _system_msgs.get(style, _system_msgs["highlights"])
    sys_msg += (
        " All timestamps must be integer seconds only — NEVER MM:SS format. "
        "Output valid JSON only — no markdown, no explanation, no extra text."
    )

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.15,
                "max_tokens": DEEPSEEK_MAX_TOKENS,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Response was truncated (max_tokens hit mid-JSON). Recover complete objects.
        recovered = _recover_partial_moments(raw)
        if not recovered:
            raise
        data = {"moments": recovered}

    result = []
    valid_cats = set(MOMENT_CATEGORIES)
    for m in data.get("moments", []):
        try:
            cat  = str(m.get("category", "general")).lower().strip()
            if cat not in valid_cats:
                cat = "general"
            role = str(m.get("narrative_role", "development")).lower().strip()
            if role not in NARRATIVE_ROLES:
                role = "development"
            entry: dict = {
                "start":          float(m["start"]),
                "end":            float(m["end"]),
                "score":          float(m.get("score", 5.0)),
                "category":       cat,
                "narrative_role": role,
                "reason":         str(m.get("reason", "")),
            }
            if m.get("beat_id") is not None:
                entry["beat_id"] = int(m["beat_id"])
            result.append(entry)
        except (KeyError, ValueError, TypeError):
            continue

    result.sort(key=lambda x: x["score"], reverse=True)

    if cache_key:
        save_cached(cache_key, f"{settings_tag}.json", result)

    return result


# ── SMART BOUNDARY SNAPPING ───────────────────────────────────────────────────

_SENT_END_CHARS   = frozenset('.!?…')
_CLAUSE_END_CHARS = frozenset(',;:')


def _is_sentence_end(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return (t[-1] in _SENT_END_CHARS or
            t.endswith(('...', '!"', '?"', '!»', '?»', '."', '…"')))


def _is_sentence_start(text: str) -> bool:
    t = text.strip()
    return bool(t) and (t[0].isupper() or t[0].isdigit())


def _is_question(text: str) -> bool:
    return bool(_Q_PAT.search(text))


def _is_response_starter(text: str) -> bool:
    t = text.strip().lower()
    starters = (
        "да,", "нет,", "ну,", "вот,", "ладно,", "окей,", "хорошо,",
        "короче,", "смотри,", "слушай,", "понимаешь,", "значит,",
        "well,", "no,", "yes,", "okay,", "actually,", "honestly,",
        "i mean", "so,", "look,", "listen,", "you know,", "right,",
        "basically,",
    )
    return any(t.startswith(s) for s in starters)


def _sentence_breaks(segments: list[dict]) -> tuple[list[tuple], list[tuple]]:
    """
    Returns (starts, ends) of (priority, time) tuples for segment-level boundaries.
    Priority 3: very long gap ≥ 1.5 s + sentence boundary
    Priority 2: gap ≥ MIN_PAUSE + sentence end/start
    Priority 1: gap ≥ MIN_PAUSE (any)
    Priority 0: sentence end with short gap
    """
    starts: list[tuple] = []
    ends:   list[tuple] = []

    for i, seg in enumerate(segments):
        if i > 0:
            gap = seg["start"] - segments[i - 1]["end"]
            sent_boundary = (
                _is_sentence_end(segments[i - 1]["text"]) or
                _is_sentence_start(seg["text"])
            )
            if gap >= 1.5 and sent_boundary:
                starts.append((3, seg["start"]))
            elif gap >= MIN_PAUSE and sent_boundary:
                starts.append((2, seg["start"]))
            elif gap >= MIN_PAUSE:
                starts.append((1, seg["start"]))

        if i < len(segments) - 1:
            gap   = segments[i + 1]["start"] - seg["end"]
            ended = _is_sentence_end(seg["text"])
            if gap >= 1.5 and ended:
                ends.append((3, seg["end"]))
            elif gap >= MIN_PAUSE and ended:
                ends.append((2, seg["end"]))
            elif ended:
                ends.append((1, seg["end"]))
            elif gap >= MIN_PAUSE:
                ends.append((0, seg["end"]))

    if segments:
        starts.insert(0, (3, segments[0]["start"]))
        ends.append((3, segments[-1]["end"]))

    return starts, ends


def _find_setup_start(segments: list[dict], clip_start: float, max_back: float) -> Optional[float]:
    """
    Find the true start of the scene context: last question, response starter,
    or topic shift (long pause ≥ MIN_PAUSE_TOPIC + sentence start).
    """
    cutoff     = clip_start - max_back
    last_q:    Optional[float] = None
    last_intro: Optional[float] = None
    prev_end:  Optional[float] = None

    for seg in segments:
        if seg["start"] < cutoff:
            prev_end = seg["end"]
            continue
        if seg["start"] >= clip_start:
            break

        text = seg["text"].strip()

        if _is_question(text):
            last_q     = seg["start"]
            last_intro = None   # question overrides earlier intro
        elif _is_response_starter(text) and last_q is None:
            last_intro = seg["start"]
        elif (prev_end is not None
              and seg["start"] - prev_end >= MIN_PAUSE_TOPIC
              and _is_sentence_start(text)
              and last_q is None and last_intro is None):
            last_intro = seg["start"]

        prev_end = seg["end"]

    return last_q if last_q is not None else last_intro


# ── WORD-LEVEL PRECISE PAUSE DETECTION ───────────────────────────────────────

def _collect_word_times(
    segments: list[dict], t_from: float, t_to: float
) -> list[tuple[float, float, bool, bool]]:
    """
    Returns sorted (word_start, word_end, is_sent_end, is_clause_end) for words in [t_from, t_to].
    is_sent_end:   word ends a sentence (., !, ?, …) — strongest cut signal
    is_clause_end: word ends a clause (, ; :) — medium cut signal
    """
    out = []
    for seg in segments:
        if seg.get("end", 0) < t_from or seg.get("start", 0) > t_to:
            continue
        words  = seg.get("words", [])
        n      = len(words)
        seg_se = _is_sentence_end(seg.get("text", ""))
        for i, w in enumerate(words):
            ws = w.get("s", 0)
            we = w.get("e", 0)
            if not (ws and we and t_from <= ws and we <= t_to):
                continue
            wtext      = w.get("word", "").strip()
            last_ch    = wtext[-1] if wtext else ""
            sent_end   = (last_ch in _SENT_END_CHARS) or (i == n - 1 and seg_se)
            clause_end = (last_ch in _CLAUSE_END_CHARS) and not sent_end
            out.append((ws, we, sent_end, clause_end))
    out.sort()
    return out


def _build_pauses(
    wt: list[tuple[float, float, bool, bool]]
) -> list[tuple[float, float, float, bool, bool]]:
    """
    Returns (pause_start, pause_end, duration, sent_end, clause_end) for gaps ≥ MIN_PAUSE_RECORD.
    pause_start = end of last word before gap (safe clip-end point)
    pause_end   = start of first word after gap (safe clip-start point)
    """
    result = []
    for i in range(len(wt) - 1):
        gap = wt[i + 1][0] - wt[i][1]
        if gap >= MIN_PAUSE_RECORD:
            se = wt[i][2] if len(wt[i]) > 2 else False
            ce = wt[i][3] if len(wt[i]) > 3 else False
            result.append((wt[i][1], wt[i + 1][0], gap, se, ce))
    return result


def _pause_score(dur: float, sent_end: bool, clause_end: bool) -> int:
    """Score a pause as a cut point. Returns -1 if below minimum threshold."""
    if dur < MIN_PAUSE_PHRASE:
        return -1
    score = 0
    if dur >= MIN_PAUSE_TOPIC:    score += 12
    elif dur >= MIN_PAUSE_NATURAL: score += 6
    else:                          score += 2   # >= MIN_PAUSE_PHRASE
    if sent_end:    score += 10
    elif clause_end: score += 3
    return score


def _latest_cut_before(
    t: float, pauses: list[tuple], max_back: float,
    visual: Optional[dict] = None,
) -> Optional[float]:
    """
    Best pause-end (= start of word after pause) at or before t.
    Combined audio score + visual bonus — sentence boundaries + scene changes win.
    Expands window once if nothing valid found.
    """
    for cap_mult in (1.0, 1.6):
        lo    = t - max_back * cap_mult
        valid = [
            (max(0.0, p[1] - BOUNDARY_START_PAD), _pause_score(p[2], p[3], p[4]) + _visual_score_bonus(p[1], visual))
            for p in pauses
            if lo <= p[1] <= t
        ]
        valid = [(pe, s) for pe, s in valid if s >= 0]
        if valid:
            best = max(s for _, s in valid)
            top  = [pe for pe, s in valid if s == best]
            return max(top)     # latest among best-scored
    return None


def _earliest_cut_after(
    t: float, pauses: list[tuple], max_fwd: float,
    visual: Optional[dict] = None,
) -> Optional[float]:
    """
    Best pause-start (= end of last word before pause) at or after t.
    Combined audio score + visual bonus — sentence boundaries + scene changes win.
    Expands window once if nothing valid found.
    """
    for cap_mult in (1.0, 1.6):
        hi    = t + max_fwd * cap_mult
        valid = [
            (p[0] + BOUNDARY_END_PAD, _pause_score(p[2], p[3], p[4]) + _visual_score_bonus(p[0], visual))
            for p in pauses
            if t <= p[0] <= hi
        ]
        valid = [(ps, s) for ps, s in valid if s >= 0]
        if valid:
            best = max(s for _, s in valid)
            top  = [ps for ps, s in valid if s == best]
            return min(top)     # earliest among best-scored
    return None


def _word_start_at_or_before(t: float, wt: list[tuple]) -> Optional[float]:
    """Last word-start at or before t — prevents starting mid-word."""
    cands = [w[0] for w in wt if w[0] <= t]
    return max(0.0, max(cands) - BOUNDARY_START_PAD) if cands else None


def _word_end_at_or_after(t: float, wt: list[tuple]) -> Optional[float]:
    """First word-end at or after t — prevents ending mid-word."""
    cands = [w[1] for w in wt if w[1] >= t]
    return min(cands) + BOUNDARY_END_PAD if cands else None


def _sentence_start_before(t: float, wt: list[tuple], max_back: float) -> Optional[float]:
    """
    True start of the sentence containing time t.
    Walks backward through sorted word timestamps to find the most recent
    sentence-ending word, then returns the start of the next word.
    Returns None if no sentence end found within max_back seconds.
    """
    lo = t - max_back
    before = [w for w in wt if w[1] <= t and w[0] >= lo]
    if not before:
        return None
    for ws, we, sent_end, _ in reversed(before):
        if sent_end:
            nexts = [w for w in wt if w[0] > we]
            if nexts:
                nw = min(nexts, key=lambda w: w[0])
                return max(0.0, nw[0] - BOUNDARY_START_PAD)
            return None
    return None


def _sentence_end_after(t: float, wt: list[tuple], max_fwd: float) -> Optional[float]:
    """
    End time of the first sentence that ends at or after t (word-text based).
    Does not require a pause — scans for the first sentence-ending word (. ! ?)
    whose end timestamp is >= t. Returns word_end plus a short tail buffer.
    """
    hi = t + max_fwd
    for ws, we, sent_end, _ in wt:
        if ws > hi:
            break
        if sent_end and we >= t:
            return we + SENTENCE_TAIL_PAD
    return None


def _pauses_from_silence(
    silence_spans: list,
    wt: list[tuple[float, float, bool, bool]],
    t_from: float,
    t_to: float,
) -> list[tuple[float, float, float, bool, bool]]:
    """
    Build pause tuples from FFmpeg silencedetect output.
    These are more reliable than word-gap heuristics because they come
    directly from the audio signal rather than Whisper timestamp alignment.

    pause_start = ss (silence begins  → safe clip-end   before silence)
    pause_end   = se (silence ends    → safe clip-start  after silence)
    Sentence/clause boundary inferred from nearest preceding word timestamp.
    """
    result = []
    for span in silence_spans:
        ss, se = float(span[0]), float(span[1])
        if se < t_from or ss > t_to:
            continue
        dur = se - ss
        if dur < MIN_PAUSE_RECORD:
            continue
        # Determine sentence/clause boundary from word-level data:
        # look for the last word ending ≤ 0.5 s before silence starts.
        nearby = [w for w in wt if w[1] <= ss + 0.3 and ss - w[1] <= 0.5]
        sent_end = clause_end = False
        if nearby:
            last_w   = max(nearby, key=lambda w: w[1])
            sent_end   = last_w[2]
            clause_end = last_w[3]
        result.append((ss, se, dur, sent_end, clause_end))
    return result


def snap_boundaries(
    start: float, end: float, segments: list[dict],
    visual: Optional[dict] = None,
) -> tuple[float, float]:
    if not segments:
        return start, end

    clip_dur = max(1.0, end - start)
    back_cap = min(25.0, max(8.0, clip_dur * 0.55))
    fwd_cap  = min(20.0, max(10.0, clip_dur * 0.55))  # more room to find sentence end

    # ── Build cut-candidate pauses (two sources, merged) ─────────────────────
    wt = _collect_word_times(segments, start - back_cap - 5, end + fwd_cap + 5)

    # Source A: FFmpeg silencedetect — ground truth from the audio signal
    silence_spans = (visual or {}).get("silence_spans", [])
    pauses_sil = _pauses_from_silence(silence_spans, wt,
                                      start - back_cap - 5, end + fwd_cap + 5)

    # Source B: word-gap heuristics from Whisper timestamps (fallback / supplement)
    pauses_wrd = _build_pauses(wt) if len(wt) >= 4 else []

    # Merge: keep word-gap pauses only when not already covered by a silence span
    # (avoid double-scoring the same cut point)
    pauses = list(pauses_sil)
    for pw in pauses_wrd:
        covered = any(
            abs(pw[0] - ps[0]) <= 0.25 or abs(pw[1] - ps[1]) <= 0.25
            for ps in pauses_sil
        )
        if not covered:
            pauses.append(pw)
    pauses.sort(key=lambda p: p[0])

    has_words = bool(wt)

    # Segment-level fallback (always cheap to compute)
    good_starts_seg, good_ends_seg = _sentence_breaks(segments)

    # ── Setup / question detection ───────────────────────────────────────────
    setup_start = _find_setup_start(segments, start, min(50.0, back_cap * 2.0))
    use_setup   = (setup_start is not None
                   and setup_start < start
                   and (start - setup_start) <= back_cap)

    # ── START snapping ───────────────────────────────────────────────────────
    anchor    = (setup_start + 0.1) if use_setup else start
    new_start = start

    if pauses:
        snapped = _latest_cut_before(anchor, pauses, back_cap, visual)
        if snapped is not None:
            new_start = snapped
        elif use_setup:
            wb = _word_start_at_or_before(setup_start + 0.05, wt)
            new_start = wb if wb is not None else setup_start
        else:
            wb = _word_start_at_or_before(start + 0.05, wt)
            new_start = wb if wb is not None else start
    elif has_words:
        if use_setup:
            wb = _word_start_at_or_before(setup_start + 0.05, wt)
            new_start = wb if wb is not None else setup_start
        else:
            wb = _word_start_at_or_before(start + 0.05, wt)
            new_start = wb if wb is not None else start
    else:
        if use_setup:
            new_start = setup_start
        else:
            for ms in (3, 2, 1):
                cands = [t for s, t in good_starts_seg if s >= ms and start - back_cap <= t <= start]
                if cands:
                    new_start = max(cands)
                    break

    # ── END snapping ─────────────────────────────────────────────────────────
    new_end = end

    if pauses:
        snapped = _earliest_cut_after(end, pauses, fwd_cap, visual)
        if snapped is not None:
            new_end = snapped
        else:
            wb = _word_end_at_or_after(end - 0.05, wt)
            new_end = wb if wb is not None else end
    elif has_words:
        wb = _word_end_at_or_after(end - 0.05, wt)
        new_end = wb if wb is not None else end
    else:
        for ms in (3, 2, 1, 0):
            cands = [t for s, t in good_ends_seg if s >= ms and end <= t <= end + fwd_cap]
            if cands:
                new_end = min(cands)
                break

    # ── Sentence completion pass ──────────────────────────────────────────────
    # Check word-level: is the last word before new_end a sentence end?
    # This works even when sentences end without a pause (fast speech).
    _nearby_end = [w for w in wt if abs(w[1] - new_end) <= 0.6]
    _at_sent_end = any(w[2] for w in _nearby_end)

    if not _at_sent_end and wt:
        mid_end_seg = next(
            (seg for seg in segments
             if seg["start"] < new_end < seg["end"]
             and not _is_sentence_end(seg.get("text", ""))),
            None,
        )
        # Two strategies: pause-based and word-text-based (no pause needed)
        next_sent_p = next(
            (p for p in pauses
             if p[0] >= new_end and p[3] and p[0] - new_end <= fwd_cap),
            None,
        )
        next_sent_w = _sentence_end_after(new_end, wt, fwd_cap)

        if next_sent_p and next_sent_w:
            new_end = min(next_sent_p[0], next_sent_w)
        elif next_sent_p:
            new_end = next_sent_p[0]
        elif next_sent_w:
            new_end = next_sent_w
        elif mid_end_seg:
            new_end = mid_end_seg["end"]

    # ── Sentence-start verification ───────────────────────────────────────────
    # Check if the word immediately before new_start ends a sentence.
    # If not, we're starting mid-sentence — walk back to true sentence start.
    if wt:
        pre_words = [w for w in wt if w[1] <= new_start + 0.15 and w[0] >= new_start - 2.0]
        if pre_words:
            last_pre = max(pre_words, key=lambda w: w[1])
            if not last_pre[2]:  # not a sentence end → likely mid-sentence
                sent_s = _sentence_start_before(new_start, wt, back_cap)
                if sent_s is not None and sent_s < new_start:
                    new_start = sent_s

    # If we start mid-segment, nudge back to segment start.
    # Whisper segments often represent a full reply; cutting 5-10 seconds into
    # one can sound like a chopped sentence even if word timestamps are valid.
    mid_start_seg = next(
        (seg for seg in segments if seg["start"] < new_start < seg["end"]),
        None,
    )
    if mid_start_seg and new_start - mid_start_seg["start"] <= 12.0:
        new_start = max(0.0, mid_start_seg["start"] - BOUNDARY_START_PAD)

    # ── Mid-word safety check (final pass) ───────────────────────────────────
    # Catches any residual Whisper timestamp drift that puts a boundary inside
    # a spoken phoneme.  Runs after all other adjustments.
    mid_w = next((w for w in wt if w[0] < new_start < w[1]), None)
    if mid_w:
        new_start = max(0.0, mid_w[0] - BOUNDARY_START_PAD)

    mid_w = next((w for w in wt if w[0] < new_end < w[1]), None)
    if mid_w:
        new_end = mid_w[1] + BOUNDARY_END_PAD

    mid_end_seg = next((seg for seg in segments if seg["start"] < new_end < seg["end"]), None)
    if mid_end_seg and mid_end_seg["end"] - new_end <= 12.0:
        new_end = max(new_end, mid_end_seg["end"] + BOUNDARY_END_PAD)

    return min(new_start, start), max(new_end, end)


# ── MOMENT SELECTION ──────────────────────────────────────────────────────────

def _dedup_snapped(clips: list[dict]) -> list[dict]:
    """
    Post-snap deduplication based on clip_start/clip_end.
    Two AI moments that had little overlap on raw timestamps can still snap
    to the same video region — this pass catches and removes those duplicates.
    Keeps the higher-scored clip; merging is already handled by _merge.
    """
    if len(clips) <= 1:
        return clips
    ranked = sorted(clips, key=lambda x: x.get("score", 0), reverse=True)
    kept: list[dict] = []
    for c in ranked:
        cs, ce = c["clip_start"], c["clip_end"]
        cd = max(1.0, ce - cs)
        same_src = [k for k in kept if k.get("source_file") == c.get("source_file")]
        overlap_pct = max(
            (min(ce, k["clip_end"]) - max(cs, k["clip_start"])) / cd
            for k in same_src
        ) if same_src else 0.0
        if overlap_pct < 0.30:  # stricter than pre-snap 50%
            kept.append(c)
    kept.sort(key=lambda x: x["clip_start"])
    return kept


def _moment_text(m: dict, segments: list[dict]) -> str:
    return _clip_text({"clip_start": m["start"], "clip_end": m["end"]}, segments)


def _deduplicate_moments(
    moments: list[dict],
    segments_map: Optional[dict] = None,
) -> list[dict]:
    """
    Remove overlapping/repeated moments per source file, keeping the higher-scored one.

    Two moments are treated as duplicates when either:
      - they overlap by more than 50% of the shorter one's duration, or
      - (when segments_map is given) the spoken text is highly similar even though
        the moments are far apart in time — e.g. the same point made twice in an
        hour-long video. This lets heavily-repetitive sources compress further
        without losing any single idea.
    """
    if not moments:
        return moments
    ranked = sorted(moments, key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []
    texts_by_src: dict = {}
    for m in ranked:
        ms, me = m["start"], m["end"]
        md = max(1.0, me - ms)
        src = m.get("source_file")
        same_src = [k for k in kept if k.get("source_file") == src]
        overlap_pct = max(
            (min(me, k["end"]) - max(ms, k["start"])) / md
            for k in same_src
        ) if same_src else 0.0
        duplicate = overlap_pct >= 0.50
        text = ""
        if not duplicate and segments_map:
            segs = segments_map.get(src) or []
            if segs:
                text = _moment_text(m, segs)
                if text:
                    duplicate = any(
                        _text_similarity(text, t) >= 0.62
                        for t in texts_by_src.get(src, [])
                    )
        if duplicate:
            continue
        kept.append(m)
        if text:
            texts_by_src.setdefault(src, []).append(text)
    return kept


def select_moments(
    moments: list[dict],
    target_seconds: int,
    segments: Optional[list[dict]] = None,
    min_score: float = 7.0,
    unlimited: bool = False,
    segments_map: Optional[dict] = None,
    visual_map: Optional[dict] = None,
    style: str = "highlights",
    story_beats: Optional[list] = None,
) -> list[dict]:
    if not moments:
        return []

    style_clip_caps = {
        "entertainment": 95.0,
        "gaming": 120.0,
        "educational": 150.0,
        "highlights": 120.0,
        "condensed": 210.0,
    }
    max_auto_clip = style_clip_caps.get(style, 120.0)

    source_ids_all = sorted({m.get("source_file") for m in moments}, key=lambda x: str(x))
    source_count = max(1, len(source_ids_all))
    condensed_budget_per_source = (
        target_seconds / source_count
        if style == "condensed" and target_seconds
        else target_seconds
    )

    # Remove overlapping/duplicate moments before selection. Condensed mode keeps
    # overlaps until merge time so setup/context is not discarded by peak scores.
    if style != "condensed":
        moments = _deduplicate_moments(moments, segments_map)

    # ── Style-based + narrative-aware effective score ─────────────────────────
    pref_cats = _STYLE_PREF_CATS.get(style, _STYLE_PREF_CATS["highlights"])

    # For condensed mode the threshold adapts to compression pressure.
    # 5 hours -> 20 minutes must be much stricter than 1 hour -> 40 minutes.
    effective_min = _STYLE_MIN_SCORE.get(style, min_score)
    if style == "condensed" and not unlimited and target_seconds:
        if segments_map:
            source_total = sum((segs[-1]["end"] - segs[0]["start"]) for segs in segments_map.values() if segs)
        elif segments:
            source_total = segments[-1]["end"] - segments[0]["start"]
        else:
            by_src: dict[str, float] = {}
            for m in moments:
                sf = str(m.get("source_file", ""))
                by_src[sf] = max(by_src.get(sf, 0.0), float(m.get("end", 0)))
            source_total = sum(by_src.values())
        ratio = target_seconds / source_total if source_total else 1.0
        if ratio < 0.08:
            effective_min = max(effective_min, 6.8)
        elif ratio < 0.15:
            effective_min = max(effective_min, 6.2)
        elif ratio < 0.30:
            effective_min = max(effective_min, 5.6)

    def _eff_score(m: dict) -> float:
        cat   = m.get("category", "general")
        score = m["score"]
        if style != "highlights" and style != "condensed" and cat not in pref_cats:
            role = m.get("narrative_role", "development")
            if style == "entertainment" and role in ("intro", "setup", "development"):
                score = max(4.5, score * 0.78)
            else:
                score = max(2.0, score * 0.50)
        if style == "entertainment":
            if cat == "funny":
                score += 0.7
            elif cat == "reaction":
                score += 0.4
        role  = m.get("narrative_role", "development")
        score += _NARRATIVE_BONUS.get(role, 0.0)
        return score

    ranked = sorted(moments, key=_eff_score, reverse=True)
    if style == "condensed":
        chronological = sorted(moments, key=lambda m: (str(m.get("source_file", "")), m.get("start", 0)))
        candidates = [
            m for m in chronological
            if _eff_score(m) >= effective_min or m.get("score", 0) >= 5.0
        ]
        if len(candidates) < 5:
            candidates = sorted(ranked[:max(8, len(ranked))], key=lambda m: (str(m.get("source_file", "")), m.get("start", 0)))
    else:
        above  = [m for m in ranked if _eff_score(m) >= effective_min]
        if len(above) < 3:
            below_thresh = [m for m in ranked if _eff_score(m) >= max(4.0, effective_min - 1.5)]
            candidates = below_thresh[:max(5, len(above) * 2 + 2)]
        else:
            candidates = list(above)

    # Add low-scored but structurally important context. Highlight-only scoring
    # often underrates setup/resolution, but removing them makes the edit absurd.
    def _moment_key(moment: dict) -> tuple:
        return (
            moment.get("source_file"),
            round(moment.get("start", 0), 2),
            round(moment.get("end", 0), 2),
        )

    def _add_context_candidate(moment: Optional[dict]) -> None:
        if not moment:
            return
        key = _moment_key(moment)
        if not any(_moment_key(c) == key for c in candidates):
            candidates.append(moment)

    source_ids = source_ids_all
    for sf in source_ids:
        src_moments = [m for m in ranked if m.get("source_file") == sf]
        src_chrono = sorted(src_moments, key=lambda m: m.get("start", 0))
        segs_for_src = (segments_map or {}).get(sf) if (segments_map and sf) else segments
        vid_end = segs_for_src[-1]["end"] if segs_for_src else max((m.get("end", 0) for m in src_moments), default=0)
        early_limit = max(90.0, vid_end * 0.25) if vid_end else 90.0
        late_limit = vid_end * 0.70 if vid_end else 0.0

        intro = next(
            (m for m in src_moments
             if m.get("narrative_role") in ("intro", "setup")
             and m.get("start", 0) <= early_limit
             and _eff_score(m) >= 4.5),
            None,
        )
        if intro is None:
            intro = next(
                (m for m in src_moments
                 if m.get("start", 0) <= early_limit and _eff_score(m) >= 5.0),
                None,
            )
        _add_context_candidate(intro)

        resolution = next(
            (m for m in src_moments
             if m.get("narrative_role") == "resolution"
             and (not vid_end or m.get("start", 0) >= late_limit)
             and _eff_score(m) >= 4.5),
            None,
        )
        _add_context_candidate(resolution)

        if style == "condensed":
            if condensed_budget_per_source <= 180:
                bucket_count = 2
            elif condensed_budget_per_source <= 360:
                bucket_count = 3
            else:
                bucket_count = 5 if vid_end and vid_end >= 900 else 3
            for b in range(bucket_count):
                bs = (vid_end * b / bucket_count) if vid_end else 0.0
                be = (vid_end * (b + 1) / bucket_count) if vid_end else float("inf")
                bucket = [
                    m for m in src_moments
                    if bs <= m.get("start", 0) < be and _eff_score(m) >= 4.5
                ]
                if bucket:
                    _add_context_candidate(max(bucket, key=_eff_score))

            critical_ids = {
                b.get("id") for b in (story_beats or [])
                if b.get("importance") == "critical" and (not b.get("source_file") or b.get("source_file") == sf)
            }
            if critical_ids:
                for m in src_moments:
                    if m.get("beat_id") in critical_ids and _eff_score(m) >= 4.0:
                        _add_context_candidate(m)

        if style == "entertainment":
            humor_peaks = [
                m for m in src_chrono
                if m.get("category") in ("funny", "reaction") and _eff_score(m) >= effective_min
            ][:12]
            for peak in humor_peaks:
                setup = next(
                    (
                        m for m in reversed(src_chrono)
                        if m.get("end", 0) <= peak.get("start", 0)
                        and peak.get("start", 0) - m.get("end", 0) <= 90.0
                        and m.get("narrative_role") in ("intro", "setup", "development")
                        and _eff_score(m) >= 4.5
                    ),
                    None,
                )
                _add_context_candidate(setup)

    # Collect ALL above-threshold candidates first (no early budget cutoff).
    # This guarantees clips from the end of the video aren't starved by
    # high-scoring early clips filling the time budget first.
    clips = []

    for m in candidates:
        s, e = m["start"], m["end"]

        if e - s < MIN_CLIP_SEC:
            mid = (s + e) / 2
            s   = max(0.0, mid - MIN_CLIP_SEC / 2)
            e   = mid + MIN_CLIP_SEC / 2

        sf     = m.get("source_file")
        segs   = (segments_map or {}).get(sf) if (segments_map and sf) else segments
        visual = (visual_map or {}).get(sf)    if (visual_map and sf)   else None

        if segs:
            vid_end = segs[-1]["end"]
            if s >= vid_end - 2.0:
                continue  # starts past/at video end — would be empty
            # Tail buffer: AI timestamps are often tight on the end side and cut off
            # before the scene settles. Adding 1.5 s gives snap_boundaries a later
            # anchor so it can find the first natural sentence-end pause AFTER the
            # moment — resulting in complete, non-abrupt clips.
            dur_hint = max(MIN_CLIP_SEC, e - s)
            role = m.get("narrative_role", "development")
            head = SCENE_HEADROOM + (1.5 if role in ("intro", "setup") else 0.0)
            tail = max(SCENE_TAILROOM, min(7.0, dur_hint * 0.18))
            s = max(0.0, s - head)
            e = min(e + tail, vid_end)

        if segs:
            s, e = snap_boundaries(s, e, segs, visual)
            vid_end = segs[-1]["end"]
            s = max(0.0, min(s, vid_end))
            e = max(s + 0.25, min(e, vid_end))
            if e - s > max_auto_clip:
                raw_s = float(m.get("start", s))
                raw_e = float(m.get("end", e))
                keep_start = max(s, raw_s - 4.0)
                keep_end = min(e, max(raw_e + 8.0, keep_start + MIN_CLIP_SEC), keep_start + max_auto_clip)
                s, e = snap_boundaries(keep_start, keep_end, segs, visual)
                s = max(0.0, min(s, vid_end))
                e = max(s + 0.25, min(e, vid_end, s + max_auto_clip + 12.0))

        dur = e - s
        if dur < 1.0:
            continue
        clips.append({**m, "clip_start": s, "clip_end": e, "clip_duration": dur, "_merge_cap": max_auto_clip + 12.0})

    clips.sort(key=lambda x: x["clip_start"])
    clips = _merge(clips)
    clips = _dedup_snapped(clips)  # remove duplicates that arose after boundary snapping

    if not unlimited:
        # ── Build beat coverage index ─────────────────────────────────────────
        critical_beat_keys: set = set()
        if story_beats:
            for b in story_beats:
                if b.get("importance") == "critical":
                    if b.get("source_file"):
                        critical_beat_keys.add((b.get("source_file"), b.get("id")))
                    else:
                        critical_beat_keys.add((None, b.get("id")))

        def _beat_coverage(idx: int) -> set:
            bid = clips[idx].get("beat_id")
            if not bid:
                return set()
            sf = clips[idx].get("source_file")
            keys = {(sf, bid), (None, bid)}
            return keys & critical_beat_keys

        def _is_last_cover_of_beat(i: int) -> bool:
            covered = _beat_coverage(i)
            if not covered:
                return False
            for beat_id in covered:
                others = [j for j in range(len(clips)) if j != i and beat_id in _beat_coverage(j)]
                if not others:
                    return True  # removing i would leave this critical beat uncovered
            return False

        # ── Trim by time budget ───────────────────────────────────────────────
        # Protected = last intro/setup clip OR last clip covering a critical beat.
        def _condensed_anchor_indices() -> set[int]:
            if style != "condensed":
                return set()
            anchors: set[int] = set()
            grouped: dict[str, list[tuple[int, dict]]] = {}
            for i, clip in enumerate(clips):
                grouped.setdefault(str(clip.get("source_file", "")), []).append((i, clip))

            for sf, items in grouped.items():
                ordered = sorted(items, key=lambda item: item[1].get("clip_start", 0))
                if not ordered:
                    continue
                anchors.add(ordered[0][0])
                anchors.add(ordered[-1][0])

                intro_setup = [
                    item for item in ordered
                    if item[1].get("narrative_role") in ("intro", "setup")
                ]
                if intro_setup:
                    anchors.add(intro_setup[0][0])

                resolution = [
                    item for item in ordered
                    if item[1].get("narrative_role") == "resolution"
                ]
                if resolution:
                    anchors.add(resolution[-1][0])

                segs_for_src = (segments_map or {}).get(ordered[0][1].get("source_file")) if segments_map else segments
                vid_end = (
                    segs_for_src[-1]["end"]
                    if segs_for_src
                    else max((clip.get("clip_end", 0) for _, clip in ordered), default=0)
                )
                if condensed_budget_per_source <= 180:
                    bucket_count = 2
                elif condensed_budget_per_source <= 360:
                    bucket_count = 3
                else:
                    bucket_count = 5 if vid_end and vid_end >= 900 else 3
                for b in range(bucket_count):
                    bs = (vid_end * b / bucket_count) if vid_end else 0.0
                    be = (vid_end * (b + 1) / bucket_count) if vid_end else float("inf")
                    bucket = [
                        item for item in ordered
                        if bs <= item[1].get("clip_start", 0) < be
                    ]
                    if bucket:
                        anchors.add(max(bucket, key=lambda item: _eff_score(item[1]))[0])
            return anchors

        def _is_protected(i: int, condensed_anchors: Optional[set[int]] = None) -> bool:
            if condensed_anchors and i in condensed_anchors:
                return True
            role = clips[i].get("narrative_role", "development")
            if role in ("intro", "setup", "resolution"):
                if sum(1 for c in clips if c.get("narrative_role") == role) <= 1:
                    return True
            if style == "entertainment" and role in ("intro", "setup"):
                setup_end = clips[i].get("clip_end", clips[i].get("end", 0))
                setup_src = clips[i].get("source_file")
                has_near_humor = any(
                    c.get("source_file") == setup_src
                    and c.get("clip_start", c.get("start", 0)) >= setup_end
                    and c.get("clip_start", c.get("start", 0)) - setup_end <= 90.0
                    and c.get("category") in ("funny", "reaction")
                    and _eff_score(c) >= effective_min
                    for c in clips
                )
                if has_near_humor:
                    return True
            return _is_last_cover_of_beat(i)

        while len(clips) > 1 and sum(c["clip_duration"] for c in clips) > target_seconds:
            condensed_anchors = _condensed_anchor_indices()
            candidates_idx = [i for i in range(len(clips)) if not _is_protected(i, condensed_anchors)]
            if not candidates_idx:
                break
            if style == "condensed":
                role_drop_cost = {
                    "intro": 3.0,
                    "setup": 2.5,
                    "development": 1.0,
                    "resolution": 2.5,
                    "climax": 0.4,
                }
                worst = min(
                    candidates_idx,
                    key=lambda i: (
                        _eff_score(clips[i]) + role_drop_cost.get(clips[i].get("narrative_role"), 1.0),
                        -clips[i].get("clip_duration", 0),
                    ),
                )
            else:
                worst = min(candidates_idx, key=lambda i: _eff_score(clips[i]))
            clips.pop(worst)

        # ── Hard cap on clip count (not for condensed — it needs many clips) ──
        if style != "condensed":
            max_clips = max(8, target_seconds // 60)
            while len(clips) > max_clips:
                candidates_idx = [i for i in range(len(clips)) if not _is_protected(i)]
                if not candidates_idx:
                    break
                worst = min(candidates_idx, key=lambda i: _eff_score(clips[i]))
                clips.pop(worst)

    # ── Timeline coverage guarantee ───────────────────────────────────────────
    # Ensure clips cover early, middle, and end of the video.
    # If the first third of the video has no clip, mark the earliest available
    # clip as "intro" so the viewer has context at the start.
    if clips and (segments or segments_map):
        if segments is None and segments_map:
            first_src = clips[0].get("source_file")
            segments = segments_map.get(first_src) or next(iter(segments_map.values()), None)
        if not segments:
            return clips
        vid_total = segments[-1]["end"]
        third = vid_total / 3.0
        has_early = any(c["clip_start"] < third for c in clips)
        if not has_early:
            # The earliest clip is probably mid-video — tag it as intro so it gets
            # narrative protection in future trims and signals context to the viewer
            earliest = min(clips, key=lambda c: c["clip_start"])
            if earliest.get("narrative_role", "development") not in ("intro", "setup"):
                earliest["narrative_role"] = "setup"

    return clips


def _merge(clips: list[dict]) -> list[dict]:
    if not clips:
        return clips
    out = [clips[0].copy()]
    for c in clips[1:]:
        prev = out[-1]
        same = prev.get("source_file") == c.get("source_file")
        merge_cap = min(
            float(prev.get("_merge_cap", 10_000.0) or 10_000.0),
            float(c.get("_merge_cap", 10_000.0) or 10_000.0),
        )
        merged_duration = max(prev["clip_end"], c["clip_end"]) - prev["clip_start"]
        if same and c["clip_start"] - prev["clip_end"] < MERGE_GAP and merged_duration <= merge_cap:
            prev["clip_end"]      = max(prev["clip_end"], c["clip_end"])
            prev["clip_duration"] = prev["clip_end"] - prev["clip_start"]
            prev["score"]         = max(prev["score"], c["score"])
            prev_reason = prev.get("reason", "")
            cur_reason = c.get("reason", "")
            prev["reason"] = " / ".join(r for r in (prev_reason, cur_reason) if r)
        else:
            out.append(c.copy())
    return out


def _segments_overlapping(segments: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in segments if s.get("end", 0) >= start and s.get("start", 0) <= end]


def _clip_text(clip: dict, segments: list[dict]) -> str:
    parts = [
        s.get("text", "").strip()
        for s in _segments_overlapping(segments, clip["clip_start"], clip["clip_end"])
        if s.get("text", "").strip()
    ]
    return " ".join(parts)


def _norm_text_for_dedup(text: str) -> set[str]:
    words = re.findall(r"[\w']{3,}", text.lower(), flags=re.UNICODE)
    stop = {
        "the", "and", "that", "this", "you", "for", "with", "but", "are", "was",
        "что", "это", "как", "вот", "там", "тут", "они", "она", "его", "еще",
    }
    return {w for w in words if w not in stop}


def _text_similarity(a: str, b: str) -> float:
    aw, bw = _norm_text_for_dedup(a), _norm_text_for_dedup(b)
    if not aw or not bw:
        return 0.0
    inter = len(aw & bw)
    return max(inter / len(aw), inter / len(bw), inter / len(aw | bw))


def _qa_expand_clip(
    clip: dict,
    segments: list[dict],
    visual: Optional[dict],
    video_duration: float,
) -> dict:
    """Heuristic QA pass: make clips self-contained before rendering."""
    if not segments:
        return clip

    out = clip.copy()
    start = float(out["clip_start"])
    end = float(out["clip_end"])
    overlapping = _segments_overlapping(segments, start, end)
    if not overlapping:
        return out

    changed: list[str] = []
    first = overlapping[0]
    first_text = first.get("text", "").strip()
    first_lower = first_text.lower()
    if first.get("start", start) < start and start - first.get("start", start) <= 18.0:
        start = max(0.0, first["start"] - BOUNDARY_START_PAD)
        changed.append("segment_start")
    continuation_starters = (
        "and ", "but ", "so ", "because ", "then ", "therefore ",
        "yes", "no", "well", "okay", "right",
        "и ", "а ", "но ", "да", "нет", "ну ", "вот ", "поэтому ", "потому ",
    )
    starts_like_response = (
        _is_response_starter(first_text)
        or first_lower.startswith(continuation_starters)
        or (first.get("start", start) - start <= 0.35 and not _is_sentence_start(first_text))
    )

    if starts_like_response:
        idx = segments.index(first)
        back_limit = max(0.0, start - 45.0)
        new_start: Optional[float] = None
        for prev in reversed(segments[max(0, idx - 8):idx]):
            gap = first["start"] - prev["end"]
            if prev["start"] < back_limit:
                break
            if _is_question(prev.get("text", "")) or gap >= MIN_PAUSE_TOPIC:
                new_start = prev["start"]
                break
            if new_start is None and first["start"] - prev["start"] <= 18.0:
                new_start = prev["start"]
        if new_start is not None and new_start < start:
            start = new_start
            changed.append("context_start")

    last = overlapping[-1]
    last_text = last.get("text", "").strip()
    last_idx = segments.index(last)
    if last.get("end", end) > end and last.get("end", end) - end <= 18.0:
        end = last["end"] + BOUNDARY_END_PAD
        changed.append("segment_end")
    next_seg = segments[last_idx + 1] if last_idx + 1 < len(segments) else None
    incomplete_end = (
        not _is_sentence_end(last_text)
        or (next_seg is not None and next_seg["start"] - last["end"] < MIN_PAUSE_NATURAL)
    )
    if incomplete_end:
        limit = min(video_duration or segments[-1]["end"], end + 14.0)
        new_end: Optional[float] = None
        for nxt in segments[last_idx:last_idx + 8]:
            if nxt["end"] > limit:
                break
            following_idx = segments.index(nxt) + 1
            following = segments[following_idx] if following_idx < len(segments) else None
            gap_after = (following["start"] - nxt["end"]) if following else 99.0
            if _is_sentence_end(nxt.get("text", "")) and gap_after >= MIN_PAUSE_PHRASE:
                new_end = nxt["end"] + SENTENCE_TAIL_PAD
                break
        if new_end is None and next_seg is not None and next_seg["end"] <= limit:
            new_end = next_seg["end"] + SENTENCE_TAIL_PAD
        if new_end is not None and new_end > end:
            end = new_end
            changed.append("complete_end")

    start, end = snap_boundaries(start, end, segments, visual)
    end = min(max(end, start + 0.25), video_duration or segments[-1]["end"])
    out["clip_start"] = max(0.0, start)
    out["clip_end"] = end
    out["clip_duration"] = end - out["clip_start"]
    if changed:
        out.setdefault("qa_flags", [])
        out["qa_flags"] = sorted(set(out["qa_flags"] + changed))
    return out


def qa_refine_clips(
    clips: list[dict],
    segments_map: dict[str, list[dict]],
    visual_map: Optional[dict[str, dict]] = None,
    video_durations: Optional[dict[str, float]] = None,
) -> list[dict]:
    refined: list[dict] = []
    for clip in clips:
        sf = clip.get("source_file", "")
        segs = segments_map.get(sf, [])
        visual = (visual_map or {}).get(sf)
        vdur = (video_durations or {}).get(sf) or (segs[-1]["end"] if segs else 0.0)
        refined.append(_qa_expand_clip(clip, segs, visual, vdur))

    refined.sort(key=lambda c: (c.get("source_file", ""), c["clip_start"]))
    refined = _merge(refined)
    refined = _dedup_snapped(refined)

    ranked = sorted(refined, key=lambda c: (c.get("score", 0), c.get("clip_duration", 0)), reverse=True)
    kept: list[dict] = []
    kept_texts: list[tuple[dict, str]] = []
    for clip in ranked:
        sf = clip.get("source_file", "")
        text = _clip_text(clip, segments_map.get(sf, []))
        duplicate = False
        for kept_clip, kept_text in kept_texts:
            if kept_clip.get("source_file") != sf:
                continue
            if _text_similarity(text, kept_text) >= 0.68:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(clip)
        kept_texts.append((clip, text))

    kept.sort(key=lambda c: (c.get("source_file", ""), c["clip_start"]))
    return kept


def enforce_duration_budget(
    clips: list[dict],
    target_seconds: int,
    style: str = "highlights",
    slack: float = 1.02,
    story_beats: Optional[list[dict]] = None,
    segments_map: Optional[dict] = None,
    visual_map: Optional[dict] = None,
) -> list[dict]:
    """Final hard budget pass after QA expansion.

    QA/refinement intentionally adds context, but on multi-hour inputs that can
    balloon the edit. This pass keeps a few structural anchors, tightens long
    clips around the AI-selected moment, then removes the lowest-value clips
    until the result is close to the requested duration.

    `story_beats` lets this pass refuse to drop the only remaining clip that
    covers a "critical" beat — without it, clips already protected by
    select_moments()'s own beat-coverage logic could still get cut here later.
    `segments_map`/`visual_map` let the tightening step re-snap to a natural
    speech pause instead of leaving a hard, mid-sentence crop.
    """
    if not clips or target_seconds <= 0:
        return clips

    out = [c.copy() for c in clips]
    guard = target_seconds * slack

    critical_beat_keys: set = set()
    for b in story_beats or []:
        if b.get("importance") == "critical":
            critical_beat_keys.add((b.get("source_file"), b.get("id")))
            critical_beat_keys.add((None, b.get("id")))

    def _beat_coverage(c: dict) -> set:
        bid = c.get("beat_id")
        if not bid or not critical_beat_keys:
            return set()
        sf = c.get("source_file")
        return {(sf, bid), (None, bid)} & critical_beat_keys

    def _is_last_cover_of_beat(idx: int) -> bool:
        covered = _beat_coverage(out[idx])
        if not covered:
            return False
        for key in covered:
            others = [j for j in range(len(out)) if j != idx and key in _beat_coverage(out[j])]
            if not others:
                return True  # removing it would leave this critical beat uncovered
        return False

    def _total() -> float:
        return sum(float(c.get("clip_duration", c.get("clip_end", 0) - c.get("clip_start", 0))) for c in out)

    def _keep_score(c: dict) -> float:
        role = c.get("narrative_role", "development")
        role_bonus = {
            "intro": 1.6,
            "setup": 1.3,
            "development": 0.6,
            "resolution": 1.5,
            "climax": 0.2,
        }.get(role, 0.5)
        cat_bonus = 0.4 if c.get("category") in ("funny", "reaction", "emotional", "dramatic") else 0.0
        beat_bonus = 3.0 if _beat_coverage(c) else 0.0
        return float(c.get("score", 5.0)) + role_bonus + cat_bonus + beat_bonus

    def _set_bounds(c: dict, start: float, end: float) -> None:
        c["clip_start"] = max(0.0, start)
        c["clip_end"] = max(c["clip_start"] + 0.25, end)
        c["clip_duration"] = c["clip_end"] - c["clip_start"]

        # Re-snap to the nearest natural speech pause/sentence boundary instead
        # of leaving a hard crop mid-word — the same logic used during initial
        # moment selection, applied again here after the budget-driven trim.
        sf = c.get("source_file")
        segs = (segments_map or {}).get(sf) if segments_map and sf else None
        if segs:
            visual = (visual_map or {}).get(sf) if visual_map else None
            snapped_s, snapped_e = snap_boundaries(c["clip_start"], c["clip_end"], segs, visual)
            vid_end = segs[-1]["end"]
            c["clip_start"] = max(0.0, min(snapped_s, vid_end))
            c["clip_end"] = max(c["clip_start"] + 0.25, min(snapped_e, vid_end))
        c["clip_duration"] = c["clip_end"] - c["clip_start"]

    if _total() > guard:
        avg_budget = max(18.0, target_seconds / max(1, len(out)))
        cap = max(25.0, min(90.0, avg_budget * 1.8))
        for c in out:
            if _total() <= guard:
                break
            cs, ce = float(c.get("clip_start", 0)), float(c.get("clip_end", 0))
            dur = ce - cs
            if dur <= cap:
                continue
            raw_s = float(c.get("start", cs))
            raw_e = float(c.get("end", ce))
            if cs <= raw_s < raw_e <= ce and raw_e - raw_s >= MIN_CLIP_SEC:
                ns, ne = max(cs, raw_s - 2.0), min(ce, raw_e + 4.0)
            elif c.get("narrative_role") in ("intro", "setup"):
                ns, ne = cs, min(ce, cs + cap)
            elif c.get("narrative_role") == "resolution":
                ns, ne = max(cs, ce - cap), ce
            else:
                mid = (cs + ce) / 2
                ns, ne = max(cs, mid - cap / 2), min(ce, mid + cap / 2)
            if ne - ns >= MIN_CLIP_SEC and ne - ns < dur:
                _set_bounds(c, ns, ne)

    def _anchor_indices() -> set[int]:
        if style != "condensed":
            return set()
        grouped: dict[str, list[tuple[int, dict]]] = {}
        for i, c in enumerate(out):
            grouped.setdefault(str(c.get("source_file", "")), []).append((i, c))
        source_count = max(1, len(grouped))
        per_source_budget = target_seconds / source_count
        anchors: set[int] = set()
        for items in grouped.values():
            ordered = sorted(items, key=lambda item: item[1].get("clip_start", 0))
            if not ordered:
                continue
            anchors.add(max(ordered, key=lambda item: _keep_score(item[1]))[0])
            if per_source_budget > 240:
                anchors.add(ordered[0][0])
                anchors.add(ordered[-1][0])
            if per_source_budget > 420:
                mids = ordered[len(ordered) // 3: max(len(ordered) // 3 + 1, len(ordered) * 2 // 3)]
                if mids:
                    anchors.add(max(mids, key=lambda item: _keep_score(item[1]))[0])

        max_anchor_budget = guard * 0.65
        while anchors and sum(out[i].get("clip_duration", 0) for i in anchors) > max_anchor_budget:
            worst_anchor = min(anchors, key=lambda i: (_keep_score(out[i]), -out[i].get("clip_duration", 0)))
            anchors.remove(worst_anchor)
        return anchors

    while len(out) > 1 and _total() > guard:
        anchors = _anchor_indices()
        protected = anchors | {i for i in range(len(out)) if _is_last_cover_of_beat(i)}
        candidates = [i for i in range(len(out)) if i not in protected]
        if not candidates:
            candidates = [i for i in range(len(out)) if i not in anchors] or list(range(len(out)))
        worst = min(candidates, key=lambda i: (_keep_score(out[i]), -out[i].get("clip_duration", 0)))
        out.pop(worst)

    return out


_BRIDGE_GAP_SEC = 35.0  # min skipped time between two clips before adding a context line


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} сек"
    return f"{round(seconds / 60)} мин"


def add_context_bridges(
    clips: list[dict],
    story_beats: Optional[list[dict]] = None,
    gap_threshold: float = _BRIDGE_GAP_SEC,
) -> list[dict]:
    """
    Flag clips that follow a large time-skip with a short on-screen context line
    (rendered by editor.cut_clip via the "bridge_text" field), so the edit can cut
    more footage between scenes without losing the viewer's sense of what happened.
    Uses the nearest pre-analyzed story beat inside the skipped gap when available,
    otherwise falls back to a plain elapsed-time indicator.
    """
    if not clips:
        return clips
    beats_by_src: dict = {}
    for b in story_beats or []:
        beats_by_src.setdefault(b.get("source_file"), []).append(b)
    importance_rank = {"critical": 0, "major": 1, "minor": 2}

    prev_by_src: dict = {}
    for c in clips:
        src = c.get("source_file")
        prev = prev_by_src.get(src)
        prev_by_src[src] = c
        if prev is None:
            continue
        gap = c["clip_start"] - prev["clip_end"]
        if gap < gap_threshold:
            continue
        elapsed = _fmt_elapsed(gap)
        note = ""
        candidates = [
            b for b in beats_by_src.get(src, [])
            if prev["clip_end"] <= float(b.get("time", 0)) <= c["clip_start"]
        ]
        if candidates:
            best = min(candidates, key=lambda b: importance_rank.get(b.get("importance", "major"), 1))
            desc = str(best.get("description", "")).strip()
            note = desc if len(desc) <= 90 else desc[:87].rstrip() + "…"
        c["bridge_text"] = f"⏩ {elapsed} спустя: {note}" if note else f"⏩ {elapsed} спустя"
    return clips


def split_clips_on_long_pauses(
    clips: list[dict],
    segments_map: dict[str, list[dict]],
    visual_map: Optional[dict[str, dict]] = None,
    min_pause: float = 2.0,
    min_piece: float = 5.0,
) -> list[dict]:
    """Remove long internal silence by splitting a clip around it."""
    out: list[dict] = []
    for clip in clips:
        sf = clip.get("source_file", "")
        start, end = float(clip["clip_start"]), float(clip["clip_end"])
        spans = [
            (float(s), float(e))
            for s, e in ((visual_map or {}).get(sf, {}).get("silence_spans", []) or [])
            if e - s >= min_pause and start + min_piece <= s and e <= end - min_piece
        ]
        if not spans:
            segs = segments_map.get(sf, [])
            for a, b in zip(segs, segs[1:]):
                gap = b["start"] - a["end"]
                if gap >= min_pause and start + min_piece <= a["end"] and b["start"] <= end - min_piece:
                    spans.append((a["end"], b["start"]))
        if not spans:
            out.append(clip)
            continue

        cur = start
        part = 1
        for ss, se in sorted(spans):
            if ss - cur >= min_piece:
                c = clip.copy()
                c["clip_start"] = cur
                c["clip_end"] = ss
                c["clip_duration"] = ss - cur
                c["reason"] = f"{clip.get('reason', '')} / pause-clean part {part}".strip(" /")
                out.append(c)
                part += 1
            cur = max(cur, se)
        if end - cur >= min_piece:
            c = clip.copy()
            c["clip_start"] = cur
            c["clip_end"] = end
            c["clip_duration"] = end - cur
            c["reason"] = f"{clip.get('reason', '')} / pause-clean part {part}".strip(" /")
            out.append(c)
    out.sort(key=lambda c: (c.get("source_file", ""), c["clip_start"]))
    return out


async def ai_review_clip_plan(
    clips: list[dict],
    segments_map: dict[str, list[dict]],
    api_key: Optional[str] = None,
) -> list[dict]:
    """Second-pass AI review of the chosen edit plan.

    Returns conservative actions:
      {index, action: keep|drop|extend, extend_start_sec, extend_end_sec, reason}
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or not clips:
        return []

    items = []
    for i, clip in enumerate(clips[:40]):
        sf = clip.get("source_file", "")
        text = _clip_text(clip, segments_map.get(sf, []))
        if len(text) > 700:
            text = text[:330] + " ... " + text[-330:]
        items.append({
            "index": i,
            "start": round(clip.get("clip_start", 0), 2),
            "end": round(clip.get("clip_end", 0), 2),
            "role": clip.get("narrative_role", "development"),
            "score": clip.get("score", 0),
            "text": text,
        })

    prompt = f"""Review this automatic video edit plan. Fix only obvious coherence problems.

Look for:
- a clip starts with an answer/continuation but lacks the question or setup
- a clip ends before the sentence/reaction resolves
- duplicate clips saying the same thing
- a clip is nonsensical without context

Return ONLY valid JSON:
{{"actions":[{{"index":0,"action":"keep|drop|extend","extend_start_sec":0,"extend_end_sec":0,"reason":"short"}}]}}

Rules:
- Prefer keep.
- Use drop only for clear duplicate or meaningless clip.
- For extend, use small values: 2-20 seconds.
- Do not invent new clips.

CLIPS:
{json.dumps(items, ensure_ascii=False)}"""

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a strict video-edit QA reviewer. Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 3000,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
        data = json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception:
        return []

    actions = []
    for a in data.get("actions", []):
        try:
            idx = int(a.get("index"))
            action = str(a.get("action", "keep")).lower().strip()
            if action not in {"keep", "drop", "extend"}:
                action = "keep"
            actions.append({
                "index": idx,
                "action": action,
                "extend_start_sec": max(0.0, min(20.0, float(a.get("extend_start_sec", 0) or 0))),
                "extend_end_sec": max(0.0, min(20.0, float(a.get("extend_end_sec", 0) or 0))),
                "reason": str(a.get("reason", ""))[:160],
            })
        except (TypeError, ValueError):
            continue
    return actions


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _to_text(segments: list[dict]) -> str:
    """
    Converts segments to timestamped transcript with pause/topic-break markers.
    Pause markers help DeepSeek understand the natural rhythm and choose
    timestamps that fall at real speech boundaries.
    """
    lines: list[str] = []
    for i, s in enumerate(segments):
        if i > 0:
            gap = s["start"] - segments[i - 1]["end"]
            if gap >= 2.5:
                lines.append(f"  ── TOPIC BREAK ({gap:.1f}s) ──")
            elif gap >= MIN_PAUSE:
                lines.append(f"  [pause {gap:.1f}s]")
        lines.append(f"[{_fmt(s['start'])}-{_fmt(s['end'])}] {s['text']}")
    return "\n".join(lines)


def _fmt(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


_STYLE_TIPS = {
    "highlights": (
        "Peak reactions, genuine surprise/shock, laugh-out-loud moments, memorable quotes, "
        "emotional climaxes. Include brief setup clips so climax moments aren't confusing."
    ),
    "entertainment": (
        "Jokes that land, banter and roasts, absurd turns, awkward pauses, reaction shots, "
        "callback/payoff scenes, quotable one-liners. Include the setup and aftermath that "
        "make the punchline understandable."
    ),
    "gaming": (
        "Clutch plays, outplays, big fails, hype reactions, trash talk that landed, funny glitches, "
        "close calls with high stakes. Include brief context so the stakes are clear."
    ),
    "educational": (
        "The single clearest explanation of a key insight, counterintuitive facts, memorable "
        "analogies, 'aha moment' reveals. Include the question/problem that the insight answers."
    ),
    "condensed": (
        "Include EVERYTHING that contributes to the story — plot points, character moments, "
        "key information, cause/effect links, emotional beats, reveals, decisions, and payoffs. "
        "Cut ONLY: repetition of the same point, pure small talk, technical difficulties, "
        "long off-topic tangents, and filler phrases with zero narrative value."
    ),
}


_PROFILE_TIPS = {
    "auto": "Use the default rules for this style.",
    "episode_story": (
        "Treat this as an episode recap: preserve A-plot and B-plot, first cause, escalation, "
        "turning points, callbacks, final payoff, and relationship changes. Explain each selected "
        "scene by what it makes understandable later."
    ),
    "comedy_arc": (
        "Treat this as a comedy edit: preserve setup, misdirection, escalation, punchline, reaction, "
        "and callbacks. Do not keep isolated punchlines without the setup that makes them funny."
    ),
    "shorts_pack": (
        "Select independent scenes that can become separate Shorts. Each scene must be self-contained, "
        "fast to understand, and have a strong hook/payoff inside 20-90 seconds."
    ),
    "youtube_recap": (
        "Make a balanced YouTube recap: clear intro, chronological progression, enough context between "
        "highlights, and a satisfying ending. Prefer fewer confusing jumps."
    ),
    "full_condensed": (
        "Make the most complete condensed episode that still fits the requested target. Keep the strongest "
        "story links, character turns, setups, callbacks, and resolution beats; cut minor details first."
    ),
}


def _build_prompt(transcript: str, target_min: int, total_sec: float, style: str, unlimited: bool,
                  story_beats: Optional[list] = None, montage_profile: str = "auto") -> str:
    style_desc = STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["highlights"])
    style_tips = _STYLE_TIPS.get(style, _STYLE_TIPS["highlights"])
    profile_tip = _PROFILE_TIPS.get(montage_profile or "auto", _PROFILE_TIPS["auto"])
    total_min  = total_sec / 60

    # ── Story beats section (adaptation mode) ─────────────────────────────────
    beats_section = ""
    if story_beats:
        imp_labels = {"critical": "CRITICAL — must be covered", "major": "MAJOR", "minor": "minor"}
        lines = [
            "━━ STORY STRUCTURE (pre-analyzed) ━━",
            "These key beats define the complete narrative. MUST cover every CRITICAL beat.",
            "Tag each moment with the beat_id it covers (0 if none apply).", "",
        ]
        for b in story_beats:
            imp = imp_labels.get(b.get("importance", "major"), "MAJOR")
            mm, ss = int(b["time"] // 60), int(b["time"] % 60)
            lines.append(f"  Beat {b['id']} [{imp}] @ {mm:02d}:{ss:02d} — {b['description']}")
        lines.append("")
        beats_section = "\n".join(lines) + "\n\n"

    # ── Condensed mode has a completely different prompt ───────────────────────
    if style == "condensed":
        duration_rule = (
            "No duration target — include ALL story-relevant content. "
            "The result will naturally be 40-70% of the original length."
        ) if unlimited else (
            f"Hard target ~{target_min} min TOTAL. Hit this target by keeping only the strongest "
            "cause/effect story links, representative character turns, and payoffs. Do not include "
            "every minor detail; choose the few scenes that make the condensed story understandable."
        )
        json_template = (
            '{"moments": [{"start": <int_seconds>, "end": <int_seconds>, "score": <1-10>, '
            '"category": "<funny|reaction|emotional|educational|dramatic|general>", '
            '"narrative_role": "<intro|setup|development|climax|resolution>", '
            '"beat_id": <int or 0>, '
            '"reason": "<one sentence: what story value this adds>"}}]}'
        )
        return f"""You are a professional editor creating a CONDENSED STORY VERSION of this video.
Your goal: preserve the complete story in less time. NOT a highlights reel — a coherent episode retelling.
The viewer must understand the situation, conflict, escalation, turning points, and ending without seeing the original.

VIDEO: {total_min:.1f} min total
GOAL: {duration_rule}
MONTAGE PROFILE: {montage_profile} — {profile_tip}

{beats_section}WHAT TO KEEP: {style_tips}

TRANSCRIPT:
{transcript}

━━ CONDENSED SCORING ━━
10 : Absolutely essential — removing this breaks cause/effect, motivation, reveal, or ending
8-9: Important story moment — setup, character development, key decision, plot progression, payoff
6-7: Useful context — include only when it makes a later selected scene understandable
4-5: Minor color — cut when a time target exists
1-3: Pure filler — repetition, "um/uh" talk, off-topic tangent, small talk → NEVER include

━━ CONDENSED RULES ━━
1. INCLUDE only moments that introduce important new information, change relationships, advance the plot, or create/resolve tension
2. INCLUDE setup before payoff, but use the shortest setup that makes the payoff understandable
3. CUT: restatements of what was just said, long pauses, "as I mentioned", redundant examples, technical issues
4. LENGTH: clips can be shorter than normal — 8-75 s each, cut tight around the actual content
5. ORDER: always chronological — this is a condensed version, not a highlights reel
6. COVERAGE: spread clips across the ENTIRE video — beginning, middle, AND end must all be represented
7. NEVER skip the beginning/context or ending/payoff, but keep them short under a tight target

━━ TIMESTAMPS ━━
Output timestamps as INTEGER SECONDS ONLY. [05:23] → 323. NEVER MM:SS strings.

━━ NARRATIVE ROLE ━━
Tag each moment: intro / setup / development / climax / resolution

Return ONLY valid JSON (no markdown, no explanation):
{json_template}"""

    # ── Standard highlight modes ───────────────────────────────────────────────
    duration_rule = (
        "Include ALL moments scoring 7+. No duration cap — quality over quantity."
        if unlimited else
        f"Target ~{target_min} min total. Prefer FEWER higher-quality clips over hitting the target exactly."
    )

    json_template = (
        '{"moments": [{"start": <int_seconds>, "end": <int_seconds>, "score": <1-10>, '
        '"category": "<funny|reaction|emotional|educational|dramatic|general>", '
        '"narrative_role": "<intro|setup|development|climax|resolution>", '
        '"beat_id": <int or 0>, '
        '"reason": "<one sentence: role in story + why great>"}}]}'
    )

    return f"""You are a viral video editor making a highlights reel. Your job: select moments that together form a COHERENT STORY — not just a random list of best bits.

VIDEO: {total_min:.1f} min total | STYLE: {style_desc}
GOAL: {duration_rule}
MONTAGE PROFILE: {montage_profile} — {profile_tip}

{beats_section}WHAT TO PICK ({style}): {style_tips}

For entertainment/funny edits: a punchline without setup is not a funny clip. Include the few seconds or prior exchange that explains the joke, plus the immediate reaction/aftermath when it increases the laugh.

TRANSCRIPT (break markers: [pause Xs] = speech gap, ── TOPIC BREAK (Xs) ── = strong scene change):
{transcript}

━━ SCORING GUIDE (be brutally selective) ━━
10  : Legendary — viewer would immediately share this clip
9   : Unmissable — clearly the highlight of this video
7-8 : Clearly good — worthy inclusion, real entertainment/value
5-6 : Below average — skip unless nothing better exists
1-4 : Filler, repetition, boring — NEVER include

REALITY CHECK: In a typical 1-hour video, only ~5-10 moments deserve score 7+.
Most content is NOT worth including. If in doubt, score lower and skip.

━━ NARRATIVE ARC — MOST IMPORTANT RULE ━━
The final edit must feel like a mini-movie with a clear story, NOT a random clip dump.
A viewer who has NEVER seen this video must understand what is happening throughout.

Required story structure (include at least one clip for each applicable role):
  INTRO (mandatory, early in video):
    Tells WHO is here, WHAT the topic/situation is, WHY it matters.
    Even if it scores 6 it is REQUIRED — the edit is meaningless without it.
  SETUP: context that makes later moments understandable.
  DEVELOPMENT: escalation, how the situation progresses.
  CLIMAX: peak moments, punchlines, key reveals, biggest reactions.
  RESOLUTION (if present): final outcome, conclusion.

CRITICAL: Never select ONLY climax moments — without intro/setup they are incomprehensible.

━━ CLIP RULES ━━
1. SELF-CONTAINED — viewer with zero context must understand and enjoy it fully
2. START correctly: begin at the question/setup, not just the punchline; after a [pause]
3. END correctly: 2-5 s after punchline settles; before a [pause] or TOPIC BREAK
4. BOUNDARIES: within 2 s of a [pause] or TOPIC BREAK
5. LENGTH: 12-150 s is ideal; go longer to include the complete scene
6. NO DUPLICATES: same joke/scene → keep only the best clip
7. SPREAD across the full video — early, middle, and late parts

━━ TIMESTAMPS ━━
Output timestamps as INTEGER SECONDS ONLY. [05:23] → 323. NEVER MM:SS strings.

━━ CATEGORY ━━
funny / reaction / emotional / educational / dramatic / general

━━ NARRATIVE ROLE ━━
intro / setup / development / climax / resolution

Return ONLY valid JSON (no markdown, no explanation):
{json_template}"""
