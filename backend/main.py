import asyncio
import datetime
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware
from local_security import LocalRequestMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyze import (
    analyze_transcript, select_moments, map_story_beats,
    set_cache_dir, file_hash, load_cached, save_cached, extract_visual_data,
    snap_boundaries, qa_refine_clips, enforce_duration_budget,
    split_clips_on_long_pauses, ai_review_clip_plan, add_context_bridges,
)
from downloader import download_video
from editor import (
    get_video_duration, process_video, extract_thumbnail, cut_clip,
    detect_season_episode, prepare_transition, render_quality_report,
    needs_audio_proxy, make_preview_proxy,
)
from transcribe import transcribe_video

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

app = FastAPI(title="ClipForge")
app.add_middleware(LocalRequestMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

BASE      = Path(__file__).parent.parent
ENV_PATH  = BASE / ".env"
UPLOADS   = BASE / "uploads"
OUTPUTS   = BASE / "outputs"
CACHE_DIR    = BASE / "cache"
PROJECTS_DIR = CACHE_DIR / "projects"
PREVIEWS_DIR = CACHE_DIR / "previews"
for d in (UPLOADS, OUTPUTS, CACHE_DIR, PROJECTS_DIR, PREVIEWS_DIR):
    d.mkdir(exist_ok=True)
set_cache_dir(CACHE_DIR)

JOBS: dict[str, dict] = {}


class DeepSeekKeyPayload(BaseModel):
    api_key: str


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _log(job_id: str, msg: str, pct: int):
    j = JOBS.get(job_id)
    if not j:
        return
    j["log"].append({"message": msg, "progress": pct})
    j["progress"] = max(int(j.get("progress", 0) or 0), pct)
    print(f"  [{job_id[:6]}] {pct:3d}% {msg}")


def _fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _duration_recommendation(
    total_seconds: float,
    file_count: int,
    style: str,
    target_minutes: int,
    profile: str = "auto",
) -> dict:
    total_min = max(0.0, total_seconds / 60.0)
    if total_min <= 0:
        return {}

    if style == "condensed" or profile in {"episode_story", "full_condensed", "youtube_recap"}:
        aggressive = max(8.0, total_min * 0.04)
        balanced = max(12.0, total_min * 0.07)
        detailed = max(18.0, total_min * 0.12)
        mode = "story"
    elif style == "entertainment" or profile == "comedy_arc":
        aggressive = max(5.0, total_min * 0.025)
        balanced = max(8.0, total_min * 0.045)
        detailed = max(12.0, total_min * 0.07)
        mode = "comedy"
    else:
        aggressive = max(5.0, total_min * 0.03)
        balanced = max(8.0, total_min * 0.05)
        detailed = max(12.0, total_min * 0.08)
        mode = "highlights"

    if file_count >= 3:
        # Multi-file edits need enough room for at least one clear beat per file.
        aggressive = max(aggressive, file_count * 2.5)
        balanced = max(balanced, file_count * 4.0)
        detailed = max(detailed, file_count * 6.0)

    aggressive = min(aggressive, total_min * 0.35)
    balanced = min(max(balanced, aggressive), total_min * 0.45)
    detailed = min(max(detailed, balanced), total_min * 0.60)

    target = float(target_minutes or 0)
    if target <= 0:
        assessment = "target_missing"
        explanation = "Цель не задана; лучше выбрать balanced."
    elif target < aggressive * 0.85:
        assessment = "too_short"
        explanation = "Цель очень жёсткая: монтаж будет похож на трейлер, часть сюжетных связок потеряется."
    elif target > detailed * 1.25:
        assessment = "too_long"
        explanation = "Цель слишком длинная для выбранного материала: система может оставить лишние сцены."
    else:
        assessment = "ok"
        explanation = "Цель выглядит рабочей для выбранного объёма материала."

    return {
        "source_minutes": round(total_min, 1),
        "file_count": file_count,
        "mode": mode,
        "aggressive_minutes": round(aggressive, 1),
        "balanced_minutes": round(balanced, 1),
        "detailed_minutes": round(detailed, 1),
        "recommended_minutes": round(balanced, 1),
        "target_minutes": target_minutes,
        "target_assessment": assessment,
        "explanation": explanation,
    }


def _thumbs_dir(job_id: str) -> Path:
    p = OUTPUTS / f"{job_id}_thumbs"
    p.mkdir(exist_ok=True)
    return p


def _save_project(job_id: str) -> None:
    j = JOBS.get(job_id)
    if not j:
        return
    data = {
        "job_id":       job_id,
        "created_at":   datetime.datetime.now().isoformat(),
        "video_title":  j.get("video_title") or "",
        "status":       j.get("status", ""),
        "options":      {k: v for k, v in j.get("options", {}).items() if k != "api_key"},
        "moments":      j.get("moments", []),
        "candidate_clips": j.get("candidate_clips", []),
        "story_beats":  j.get("story_beats", []),
        "quality_warnings": j.get("quality_warnings", []),
        "source_stats": j.get("source_stats", {}),
        "duration_recommendation": j.get("duration_recommendation", {}),
        "source_files": j.get("video_paths", []),
        "result":       j.get("result"),
    }
    path = PROJECTS_DIR / f"{job_id}.json"
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  [project] save failed: {e}")
        return
    # keep only recent project files (oldest deleted first)
    all_files = sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for old in all_files[:-20]:
        try:
            old.unlink()
        except OSError:
            pass


def _make_job(options: dict) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":          "queued",
        "progress":        0,
        "log":             [],
        "result":          None,
        "error":           None,
        "moments":         [],
        "candidate_clips": [],
        "story_beats":     [],
        "source_stats":    {},
        "duration_recommendation": {},
        "created_ts":      time.time(),
        "video_paths":     [],
        "video_title":     None,
        "options":         options,
    }
    return job_id


def _restore_projects_on_startup() -> None:
    for path in sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            job_id = data.get("job_id") or path.stem
            if job_id in JOBS:
                continue
            JOBS[job_id] = {
                "status": data.get("status", "reviewing"),
                "progress": 100 if data.get("status") in ("done", "reviewing") else 0,
                "log": [],
                "result": data.get("result"),
                "error": None,
                "moments": data.get("moments", []),
                "candidate_clips": data.get("candidate_clips") or data.get("moments", []),
                "story_beats": data.get("story_beats", []),
                "quality_warnings": data.get("quality_warnings", []),
                "source_stats": data.get("source_stats", {}),
                "duration_recommendation": data.get("duration_recommendation", {}),
                "created_ts": time.time(),
                "video_paths": data.get("source_files", []),
                "video_title": data.get("video_title"),
                "options": data.get("options", {}),
            }
        except Exception:
            continue


_restore_projects_on_startup()


def _get_api_key(header_key: Optional[str]) -> str:
    key = header_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise HTTPException(400, "DeepSeek API key required")
    return key


def _save_env_value(name: str, value: str) -> None:
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    prefix = f"{name}="
    updated = False
    out = []
    for line in lines:
        if line.startswith(prefix):
            out.append(f"{name}={value}")
            updated = True
        else:
            out.append(line)
    if not updated:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{name}={value}")

    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    os.environ[name] = value


def _transcript_cache_valid(segs) -> bool:
    if not isinstance(segs, list) or not segs:
        return False
    word_entries = [
        w
        for seg in segs
        for w in (seg.get("words") or [])
    ]
    # Old caches stored only word start/end. Rebuild them so punctuation-aware
    # boundary snapping can see the actual word text.
    return bool(word_entries) and any(str(w.get("word", "")).strip() for w in word_entries)


def _load_transcript_for_path(video_path: str, model: Optional[str] = None, *, require_words: bool = False):
    key = file_hash(video_path)
    suffixes = []
    if model:
        suffixes.append(f"_transcript_{model}.json")
    suffixes.extend(["_transcript_base.json", "_transcript.json"])
    seen = set()
    for suffix in suffixes:
        if suffix in seen:
            continue
        seen.add(suffix)
        cached = load_cached(key, suffix)
        if cached and (not require_words or _transcript_cache_valid(cached)):
            return cached
    for path in sorted(CACHE_DIR.glob(f"{key}_transcript*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cached and (not require_words or _transcript_cache_valid(cached)):
            return cached
    return None


def _common_opts(
    target_minutes, model, language, style, min_score, unlimited, manual_review,
    speed, flip, color, preset, normalize_audio, subtitles, chapters,
    watermark_text, watermark_auto, watermark_corner, api_key,
    max_scene_duration: int = 0,
    montage_profile: str = "auto",
) -> dict:
    return dict(
        target_minutes=target_minutes,
        model=model,
        language=language or None,
        style=style,
        montage_profile=montage_profile or "auto",
        min_score=min_score,
        unlimited=unlimited,
        manual_review=manual_review,
        speed=max(1.0, min(1.15, float(speed))),
        flip=bool(flip),
        color=color,
        preset=preset,
        normalize_audio=bool(normalize_audio),
        subtitles=bool(subtitles),
        chapters=bool(chapters),
        watermark_text=str(watermark_text).strip(),
        watermark_auto=bool(watermark_auto),
        watermark_corner=watermark_corner,
        max_scene_duration=max(0, int(max_scene_duration or 0)),
        auto_render=False,
        auto_clean_pauses=True,
        ai_plan_review=True,
        api_key=api_key,
    )


def _quality_warnings(clips: list[dict], segments_map: dict[str, list[dict]], style: str) -> list[dict]:
    warnings: list[dict] = []
    if not clips:
        return warnings

    grouped: dict[str, list[dict]] = {}
    for c in clips:
        grouped.setdefault(c.get("source_file", ""), []).append(c)

    for sf, items in grouped.items():
        items = sorted(items, key=lambda c: c.get("clip_start", 0))
        segs = segments_map.get(sf, [])
        vid_end = segs[-1]["end"] if segs else max((c.get("clip_end", 0) for c in items), default=0)
        if not vid_end:
            continue
        has_early = any(c.get("clip_start", 0) <= vid_end * 0.25 for c in items)
        has_mid = any(vid_end * 0.25 < c.get("clip_start", 0) < vid_end * 0.75 for c in items)
        has_late = any(c.get("clip_start", 0) >= vid_end * 0.75 for c in items)
        if style == "condensed":
            if not has_early:
                warnings.append({"type": "missing_intro", "message": "Condensed edit has no early context/intro."})
            if not has_mid:
                warnings.append({"type": "missing_middle", "message": "Condensed edit may skip the middle progression."})
            if not has_late:
                warnings.append({"type": "missing_resolution", "message": "Condensed edit has no late resolution/payoff."})

        roles = {c.get("narrative_role", "development") for c in items}
        if style == "condensed" and not (roles & {"intro", "setup"}):
            warnings.append({"type": "missing_setup_role", "message": "No clip is tagged as intro/setup."})
        if style == "condensed" and "resolution" not in roles:
            warnings.append({"type": "missing_resolution_role", "message": "No clip is tagged as resolution."})

    return warnings


# ── PIPELINE ──────────────────────────────────────────────────────────────────

async def _pipeline(
    job_id:      str,
    file_paths:  list[str],
    source_urls: list[str],
    options:     dict,
):
    tmp = tempfile.mkdtemp(prefix=f"cf_{job_id}_")
    try:
        JOBS[job_id]["status"] = "running"
        loop = asyncio.get_event_loop()
        all_paths: list[str] = list(file_paths)

        # ── STEP 0: download URLs ───────────────────────────────────────
        for ui, url in enumerate(source_urls):
            _log(job_id, f"Downloading {ui+1}/{len(source_urls)}…", 2 + ui * 2)

            def _dl(u=url):
                def cb(msg, pct):
                    _log(job_id, msg, 2 + int(pct * 0.14))
                return download_video(u, tmp, cb)

            info = await loop.run_in_executor(None, _dl)
            all_paths.append(info["path"])
            JOBS[job_id]["video_paths"].append(info["path"])
            JOBS[job_id]["video_title"] = JOBS[job_id].get("video_title") or info.get("title", "")
            _log(job_id, f"Downloaded: {info.get('title','')[:50]}", 16)

        if not all_paths:
            raise ValueError("No video files to process")

        n = len(all_paths)

        # ── STEP 1: transcribe ──────────────────────────────────────────
        all_segs_map: dict[str, list[dict]] = {}
        video_durations: dict[str, float] = {}

        model_size    = options.get("model", "small")
        force_retrans = options.get("force_reanalyze", False)

        _log(job_id, f"Probing durations for {n} file(s)…", 18)
        for vp in all_paths:
            dur = await loop.run_in_executor(None, lambda p=vp: get_video_duration(p))
            video_durations[vp] = dur
        total_source_sec = sum(video_durations.values())
        JOBS[job_id]["source_stats"] = {
            "file_count": n,
            "total_seconds": round(total_source_sec, 2),
            "total_minutes": round(total_source_sec / 60, 1),
            "files": [
                {"file": Path(vp).name, "seconds": round(video_durations.get(vp, 0), 2)}
                for vp in all_paths
            ],
        }
        JOBS[job_id]["duration_recommendation"] = _duration_recommendation(
            total_source_sec,
            n,
            options.get("style", "highlights"),
            int(options.get("target_minutes", 20) or 20),
            options.get("montage_profile", "auto"),
        )
        rec = JOBS[job_id]["duration_recommendation"]
        if rec:
            _log(
                job_id,
                f"Source total {rec['source_minutes']} min; recommended final {rec['aggressive_minutes']}–{rec['detailed_minutes']} min (best ~{rec['recommended_minutes']} min)",
                19,
            )

        for fi, vp in enumerate(all_paths):
            base_pct   = 20 + fi * (20 // n)
            cache_k    = file_hash(vp)
            # Cache key includes model name so switching models always re-transcribes
            tr_suffix  = f"_transcript_{model_size}.json"
            cached     = None if force_retrans else load_cached(cache_k, tr_suffix)
            if cached and not _transcript_cache_valid(cached):
                cached = None

            dur = video_durations.get(vp, 0.0)

            if cached:
                _log(job_id, f"Transcript cached ({model_size}): {Path(vp).name}", base_pct + 10)
                all_segs_map[vp] = cached
                continue

            _log(job_id, f"Transcribing {fi+1}/{n} [{model_size}]: {Path(vp).name}…", base_pct)

            def _on_tr(msg: str, pct: int, base=base_pct):
                _log(job_id, msg, base + int(pct * 0.2))

            segs = await loop.run_in_executor(None, lambda p=vp: transcribe_video(
                p, model_size=model_size,
                language=options.get("language"),
                on_progress=_on_tr, duration_hint=dur,
            ))
            if segs:
                save_cached(cache_k, tr_suffix, segs)
            all_segs_map[vp] = segs or []
            _log(job_id, f"Transcribed {len(segs or [])} segments [{model_size}]", base_pct + 18)

        # ── STEP 2: AI analysis ────────────────────────────────────────
        _log(job_id, "Analyzing with DeepSeek AI…", 45)
        all_moments: list[dict] = []
        all_story_beats: list[dict] = []
        adaptation = options.get("adaptation", False)
        story_map_required = adaptation or options.get("style") == "condensed"

        for fi, vp in enumerate(all_paths):
            segs = all_segs_map.get(vp, [])
            if not segs:
                continue
            cache_k = file_hash(vp)

            # Story-first modes: map narrative beats before clip selection.
            beats: list[dict] = []
            if story_map_required:
                label = "Adaptation" if adaptation else "Condensed"
                _log(job_id, f"{label}: mapping story beats in {Path(vp).name}…", 46)
                beats = await map_story_beats(
                    segs, api_key=options.get("api_key"), cache_key=cache_k,
                )
                if beats:
                    for b in beats:
                        b["source_file"] = vp
                    _log(job_id, f"{label}: {len(beats)} story beats found", 47)
                    all_story_beats.extend(beats)

            moments = await analyze_transcript(
                segs,
                target_minutes=options.get("target_minutes", 20),
                unlimited=options.get("unlimited", False),
                style=options.get("style", "highlights"),
                api_key=options.get("api_key"),
                cache_key=cache_k,
                force_reanalyze=options.get("force_reanalyze", False),
                story_beats=beats if beats else None,
                montage_profile=options.get("montage_profile", "auto"),
            )
            for m in moments:
                m["source_file"] = vp
            all_moments.extend(moments)

        if all_story_beats:
            JOBS[job_id]["story_beats"] = all_story_beats

        if not all_moments:
            raise ValueError("AI found no suitable moments")
        _log(job_id, f"Found {len(all_moments)} candidate moments", 60)

        # ── STEP 2.5: visual scene analysis (two-pass) ────────────────
        _log(job_id, "Visual scene analysis…", 61)  # header always shown
        all_visual_map: dict[str, dict] = {}
        n_files = len(all_paths)
        for fi, vp in enumerate(all_paths):
            cache_k  = file_hash(vp)
            cached_v = load_cached(cache_k, "_visual.json")
            # Require fps1_scores key — old caches lack 1fps pass data
            cache_valid = (cached_v is not None
                           and cached_v.get("kf_scores")
                           and "fps1_scores" in cached_v)
            if cache_valid:
                n_cuts = len(cached_v.get("scene_cuts", []))
                n_diss = len(cached_v.get("dissolves", []))
                n_sil  = len(cached_v.get("silence_spans", []))
                _log(job_id,
                     f"Visual [{fi+1}/{n_files}]: cached — {n_cuts} cuts, {n_diss} dissolves, {n_sil} silences in {Path(vp).name}",
                     61 + fi)
                all_visual_map[vp] = cached_v
            else:
                _log(job_id, f"Visual [{fi+1}/{n_files}]: 2-pass scan in {Path(vp).name}…", 61 + fi)
                vis = await loop.run_in_executor(None, lambda p=vp: extract_visual_data(p))
                n_kf   = len(vis.get("kf_scores", []))
                n_fp1  = len(vis.get("fps1_scores", []))
                n_cuts = len(vis.get("scene_cuts", []))
                n_diss = len(vis.get("dissolves", []))
                n_sil  = len(vis.get("silence_spans", []))
                _log(job_id,
                     f"Visual [{fi+1}/{n_files}]: {n_kf} keyframes + {n_fp1} 1fps frames → {n_cuts} cuts, {n_diss} dissolves, {n_sil} silences in {Path(vp).name}",
                     61 + fi)
                if n_kf > 0:
                    try:
                        save_cached(cache_k, "_visual.json", vis)
                        _log(job_id, f"Visual [{fi+1}/{n_files}]: cache saved", 62 + fi)
                    except Exception as exc:
                        _log(job_id, f"Visual [{fi+1}/{n_files}]: cache save skipped ({exc})", 62 + fi)
                all_visual_map[vp] = vis
        _VISUAL_CACHE.update(all_visual_map)

        # ── STEP 3: select + snap ──────────────────────────────────────
        _log(job_id, "Selecting moments and snapping speech boundaries...", 63)
        selected = select_moments(
            all_moments,
            target_seconds=options.get("target_minutes", 20) * 60,
            segments_map=all_segs_map,
            visual_map=all_visual_map,
            min_score=options.get("min_score", 7.0),
            unlimited=options.get("unlimited", False),
            style=options.get("style", "highlights"),
            story_beats=all_story_beats if all_story_beats else None,
        )
        if not selected:
            raise ValueError("Could not select any clips")
        _log(job_id, f"Selected draft {len(selected)} clips before QA", 64)

        _log(job_id, "QA: fixing scene context and endings...", 64)
        selected = qa_refine_clips(selected, all_segs_map, all_visual_map, video_durations)

        # Apply max scene duration cap
        max_scene_dur = options.get("max_scene_duration", 0)
        if max_scene_dur and max_scene_dur > 0:
            for clip in selected:
                if clip["clip_duration"] > max_scene_dur:
                    target_end = clip["clip_start"] + max_scene_dur
                    sf = clip.get("source_file")
                    segs = all_segs_map.get(sf, [])
                    visual = all_visual_map.get(sf)
                    if segs:
                        _, natural_end = snap_boundaries(clip["clip_start"], target_end, segs, visual)
                        vdur = video_durations.get(sf, segs[-1]["end"])
                        clip["clip_end"] = min(natural_end, target_end + 8.0, vdur)
                    else:
                        clip["clip_end"] = target_end
                    clip["clip_duration"] = clip["clip_end"] - clip["clip_start"]

        if options.get("ai_plan_review", True):
            _log(job_id, "QA: AI review of selected edit plan...", 65)
            actions = await ai_review_clip_plan(selected, all_segs_map, api_key=options.get("api_key"))
            if actions:
                by_index = {a["index"]: a for a in actions}
                reviewed: list[dict] = []
                dropped = extended = 0
                for i, clip in enumerate(selected):
                    action = by_index.get(i)
                    if action and action["action"] == "drop":
                        dropped += 1
                        continue
                    if action and action["action"] == "extend":
                        sf = clip.get("source_file", "")
                        segs = all_segs_map.get(sf, [])
                        visual = all_visual_map.get(sf)
                        vdur = video_durations.get(sf, segs[-1]["end"] if segs else 0.0)
                        clip = clip.copy()
                        clip["clip_start"] = max(0.0, clip["clip_start"] - action["extend_start_sec"])
                        clip["clip_end"] = min(vdur, clip["clip_end"] + action["extend_end_sec"])
                        if segs:
                            clip["clip_start"], clip["clip_end"] = snap_boundaries(
                                clip["clip_start"], clip["clip_end"], segs, visual,
                            )
                        clip["clip_duration"] = clip["clip_end"] - clip["clip_start"]
                        clip.setdefault("qa_flags", [])
                        clip["qa_flags"] = sorted(set(clip["qa_flags"] + ["ai_review"]))
                        extended += 1
                    reviewed.append(clip)
                selected = reviewed or selected
                _log(job_id, f"QA: AI review applied ({extended} extended, {dropped} dropped)", 66)

        if options.get("auto_clean_pauses", True):
            before = len(selected)
            selected = split_clips_on_long_pauses(selected, all_segs_map, all_visual_map)
            added = len(selected) - before
            if added:
                _log(job_id, f"QA: removed long pauses by splitting {added} extra clips", 67)

        selected = qa_refine_clips(selected, all_segs_map, all_visual_map, video_durations)

        if not options.get("unlimited", False):
            before_dur = sum(c.get("clip_duration", 0) for c in selected)
            selected = enforce_duration_budget(
                selected,
                target_seconds=options.get("target_minutes", 20) * 60,
                style=options.get("style", "highlights"),
                story_beats=all_story_beats,
                segments_map=all_segs_map,
                visual_map=all_visual_map,
            )
            after_dur = sum(c.get("clip_duration", 0) for c in selected)
            if after_dur < before_dur - 1:
                _log(job_id, f"Budget: tightened final edit {before_dur/60:.1f} → {after_dur/60:.1f} min", 68)
            polished_before = after_dur
            selected = qa_refine_clips(selected, all_segs_map, all_visual_map, video_durations)
            polished_after = sum(c.get("clip_duration", 0) for c in selected)
            if polished_after > polished_before + 1:
                _log(job_id, f"QA: polished final speech boundaries (+{polished_after-polished_before:.1f}s)", 68)

        path_order = {p: i for i, p in enumerate(all_paths)}
        selected.sort(key=lambda c: (path_order.get(c.get("source_file", ""), 0), c["clip_start"]))
        selected = add_context_bridges(selected, all_story_beats)

        # auto-watermark
        if options.get("watermark_auto") and not options.get("watermark_text"):
            for vp in all_paths:
                tag = detect_season_episode(vp)
                if tag:
                    options["watermark_text"] = tag
                    break

        # add trim bounds and source index to each clip
        path_to_idx = {p: i for i, p in enumerate(all_paths)}
        for c in selected:
            sf   = c.get("source_file", "")
            vdur = video_durations.get(sf, 0)
            c["trim_min"]   = max(0.0, c["clip_start"] - 45.0)
            c["trim_max"]   = min(vdur, c["clip_end"] + 45.0)
            c["source_idx"] = path_to_idx.get(sf, 0)

        total_dur = sum(c["clip_duration"] for c in selected)
        _log(job_id, f"Selected {len(selected)} clips ({total_dur/60:.1f} min)", 68)

        warnings = _quality_warnings(selected, all_segs_map, options.get("style", "highlights"))
        if warnings:
            JOBS[job_id]["quality_warnings"] = warnings
            _log(job_id, f"QA warnings: {len(warnings)} issue(s) to review", 69)

        JOBS[job_id]["candidate_clips"] = list(selected)

        if options.get("auto_render"):
            JOBS[job_id]["moments"] = list(selected)
            _save_project(job_id)
            _log(job_id, "Auto-render: rendering approved QA plan...", 64)
            await _render(job_id, selected, all_paths, all_segs_map, options, tmp)
            return

        # thumbnails — extracted in parallel
        td = _thumbs_dir(job_id)
        def _thumb(args):
            i, clip = args
            mid  = (clip["clip_start"] + clip["clip_end"]) / 2
            extract_thumbnail(clip.get("source_file", all_paths[0]), mid, str(td / f"{i:04d}.jpg"))
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=4) as ex:
            list(ex.map(_thumb, enumerate(selected)))

        # Always open the editor — it is the primary workflow
        JOBS[job_id]["status"] = "reviewing"
        JOBS[job_id]["moments"] = list(selected)
        _save_project(job_id)

    except Exception as exc:
        import traceback; traceback.print_exc()
        _log(job_id, f"Error: {exc}", JOBS[job_id].get("progress", 0))
        JOBS[job_id].update(status="error", error=str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _render(
    job_id: str,
    clips: list[dict],
    all_paths: list[str],
    segs_map: dict,
    options: dict,
    tmp: str,
    finalize: bool = True,
) -> dict:
    loop     = asyncio.get_event_loop()
    out_name = f"{job_id}.mp4"
    out_path = str(OUTPUTS / out_name)

    # prepare transition
    trans_src = options.get("transition_path")
    if trans_src and Path(trans_src).exists():
        trans_out = os.path.join(tmp, "transition.mp4")
        await loop.run_in_executor(None, lambda: prepare_transition(trans_src, trans_out, options))
        options = {**options, "transition_path": trans_out}
    else:
        options = {k: v for k, v in options.items() if k != "transition_path"}

    def _on_prog(msg: str, pct: int):
        _log(job_id, msg, pct)

    await loop.run_in_executor(
        None, lambda: process_video(clips, out_path, tmp, options, segs_map, _on_prog)
    )

    total_dur = sum(c["clip_duration"] for c in clips)
    qa_report = await loop.run_in_executor(None, lambda: render_quality_report(out_path, clips))
    if qa_report.get("issues"):
        _log(job_id, f"Render QA: {len(qa_report['issues'])} warning(s)", 99)
        for issue in qa_report["issues"][:3]:
            _log(job_id, f"Render QA: {issue.get('message', issue.get('type'))}", 99)
    else:
        _log(job_id, "Render QA: passed", 99)

    result = {
        "filename":     out_name,
        "download_url": f"/download/{out_name}",
        "duration_min": round(total_dur / 60, 1),
        "clips":        len(clips),
        "avg_score":    round(sum(c.get("score", 0) for c in clips) / max(1, len(clips)), 1),
        "qa":           qa_report,
    }
    if finalize:
        JOBS[job_id].update(status="done", moments=list(clips), result=result)
        _save_project(job_id)
    return result


# ── ROUTES: CONFIG ────────────────────────────────────────────────────────────

@app.get("/config")
async def config():
    return {"has_api_key": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())}


@app.post("/config/deepseek-key")
async def save_deepseek_key(payload: DeepSeekKeyPayload):
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(400, "DeepSeek API key is empty")
    _save_env_value("DEEPSEEK_API_KEY", api_key)
    return {"ok": True, "has_api_key": True}


@app.post("/cache/clear-analysis")
async def clear_analysis_cache():
    patterns = ("*_ai_*.json", "*_story_beats*.json", "*_visual.json")
    removed = 0
    for pat in patterns:
        for p in CACHE_DIR.glob(pat):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return {"ok": True, "removed": removed}


# ── ROUTES: UPLOAD ────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    target_minutes: int = Form(20), model: str = Form("small"),
    language: Optional[str] = Form(None), style: str = Form("highlights"),
    montage_profile: str = Form("auto"),
    min_score: float = Form(7.0), unlimited: bool = Form(False),
    manual_review: bool = Form(False),
    speed: float = Form(1.0), flip: bool = Form(False),
    color: str = Form("none"), preset: str = Form("youtube"),
    normalize_audio: bool = Form(False), subtitles: bool = Form(False),
    chapters: bool = Form(False), watermark_text: str = Form(""),
    watermark_auto: bool = Form(True), watermark_corner: str = Form("br"),
    max_scene_duration: int = Form(0),
    force_reanalyze: bool = Form(False),
    auto_render: bool = Form(False),
    auto_clean_pauses: bool = Form(True),
    ai_plan_review: bool = Form(True),
    transition: Optional[UploadFile] = File(None),
    x_deepseek_key: Optional[str] = Header(None),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}:
        raise HTTPException(400, f"Unsupported format: {ext}")

    api_key = _get_api_key(x_deepseek_key)
    opts    = _common_opts(target_minutes, model, language, style, min_score, unlimited,
                           manual_review, speed, flip, color, preset, normalize_audio,
                           subtitles, chapters, watermark_text, watermark_auto, watermark_corner, api_key,
                           max_scene_duration, montage_profile)
    opts["force_reanalyze"] = bool(force_reanalyze)
    opts["auto_render"] = bool(auto_render)
    opts["manual_review"] = not bool(auto_render)
    opts["auto_clean_pauses"] = bool(auto_clean_pauses)
    opts["ai_plan_review"] = bool(ai_plan_review)
    job_id  = _make_job(opts)
    vpath   = str(UPLOADS / f"{job_id}{ext}")
    with open(vpath, "wb") as f:
        f.write(await file.read())
    JOBS[job_id]["video_paths"].append(vpath)

    if transition and transition.filename:
        te = Path(transition.filename).suffix.lower()
        tp = str(UPLOADS / f"{job_id}_trans{te}")
        with open(tp, "wb") as f:
            f.write(await transition.read())
        opts["transition_path"] = tp

    bg.add_task(_pipeline, job_id, [vpath], [], opts)
    return {"job_id": job_id}


@app.post("/upload-multi")
async def upload_multi(
    bg: BackgroundTasks,
    files: list[UploadFile] = File(...),
    target_minutes: int = Form(20), model: str = Form("small"),
    language: Optional[str] = Form(None), style: str = Form("highlights"),
    montage_profile: str = Form("auto"),
    min_score: float = Form(7.0), unlimited: bool = Form(False),
    manual_review: bool = Form(False),
    speed: float = Form(1.0), flip: bool = Form(False),
    color: str = Form("none"), preset: str = Form("youtube"),
    normalize_audio: bool = Form(False), subtitles: bool = Form(False),
    chapters: bool = Form(False), watermark_text: str = Form(""),
    watermark_auto: bool = Form(True), watermark_corner: str = Form("br"),
    max_scene_duration: int = Form(0),
    force_reanalyze: bool = Form(False),
    auto_render: bool = Form(False),
    auto_clean_pauses: bool = Form(True),
    ai_plan_review: bool = Form(True),
    transition: Optional[UploadFile] = File(None),
    x_deepseek_key: Optional[str] = Header(None),
):
    api_key  = _get_api_key(x_deepseek_key)
    opts     = _common_opts(target_minutes, model, language, style, min_score, unlimited,
                             manual_review, speed, flip, color, preset, normalize_audio,
                             subtitles, chapters, watermark_text, watermark_auto, watermark_corner, api_key,
                             max_scene_duration, montage_profile)
    opts["force_reanalyze"] = bool(force_reanalyze)
    opts["auto_render"] = bool(auto_render)
    opts["manual_review"] = not bool(auto_render)
    opts["auto_clean_pauses"] = bool(auto_clean_pauses)
    opts["ai_plan_review"] = bool(ai_plan_review)
    job_id   = _make_job(opts)
    vpaths   = []
    for i, f in enumerate(files):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}:
            continue
        p = str(UPLOADS / f"{job_id}_{i:02d}{ext}")
        with open(p, "wb") as fh:
            fh.write(await f.read())
        vpaths.append(p)
        JOBS[job_id]["video_paths"].append(p)

    if not vpaths:
        raise HTTPException(400, "No valid video files")

    if transition and transition.filename:
        te = Path(transition.filename).suffix.lower()
        tp = str(UPLOADS / f"{job_id}_trans{te}")
        with open(tp, "wb") as fh:
            fh.write(await transition.read())
        opts["transition_path"] = tp

    bg.add_task(_pipeline, job_id, vpaths, [], opts)
    return {"job_id": job_id}


@app.post("/from-url")
async def from_url(
    bg: BackgroundTasks,
    url: str = Form(...),
    target_minutes: int = Form(20), model: str = Form("small"),
    language: Optional[str] = Form(None), style: str = Form("highlights"),
    montage_profile: str = Form("auto"),
    min_score: float = Form(7.0), unlimited: bool = Form(False),
    manual_review: bool = Form(False),
    speed: float = Form(1.0), flip: bool = Form(False),
    color: str = Form("none"), preset: str = Form("youtube"),
    normalize_audio: bool = Form(False), subtitles: bool = Form(False),
    chapters: bool = Form(False), watermark_text: str = Form(""),
    watermark_auto: bool = Form(True), watermark_corner: str = Form("br"),
    max_scene_duration: int = Form(0),
    force_reanalyze: bool = Form(False),
    auto_render: bool = Form(False),
    auto_clean_pauses: bool = Form(True),
    ai_plan_review: bool = Form(True),
    x_deepseek_key: Optional[str] = Header(None),
):
    urls = [u.strip() for u in url.splitlines() if u.strip().startswith("http")]
    if not urls:
        raise HTTPException(400, "Invalid URL")
    api_key = _get_api_key(x_deepseek_key)
    opts    = _common_opts(target_minutes, model, language, style, min_score, unlimited,
                           manual_review, speed, flip, color, preset, normalize_audio,
                           subtitles, chapters, watermark_text, watermark_auto, watermark_corner, api_key,
                           max_scene_duration, montage_profile)
    opts["force_reanalyze"] = bool(force_reanalyze)
    opts["auto_render"] = bool(auto_render)
    opts["manual_review"] = not bool(auto_render)
    opts["auto_clean_pauses"] = bool(auto_clean_pauses)
    opts["ai_plan_review"] = bool(ai_plan_review)
    job_id  = _make_job(opts)
    bg.add_task(_pipeline, job_id, [], urls, opts)
    return {"job_id": job_id}


# ── ROUTES: STATUS ────────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
async def status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")

    def _enrich(clips):
        return [{**c, "thumb_url": f"/preview/{job_id}/{i}"} for i, c in enumerate(clips)]

    return {
        "job_id":          job_id,
        "status":          j["status"],
        "progress":        j["progress"],
        "log":             j["log"],
        "result":          j["result"],
        "error":           j["error"],
        "moments":         _enrich(j.get("moments", [])),
        "candidate_clips": _enrich(j.get("candidate_clips", [])),
        "story_beats":     j.get("story_beats", []),
        "quality_warnings": j.get("quality_warnings", []),
        "source_stats":    j.get("source_stats", {}),
        "duration_recommendation": j.get("duration_recommendation", {}),
        "processing_elapsed_sec": round(time.time() - float(j.get("created_ts", time.time())), 1),
        "video_title":     j.get("video_title"),
        "options":         {k: v for k, v in j.get("options", {}).items() if k != "api_key"},
    }


# ── ROUTES: REVIEW RENDER ─────────────────────────────────────────────────────

class RenderRequest(BaseModel):
    approved: list[int]
    clip_adjustments: dict = {}   # str(idx) -> {clip_start, clip_end}
    clip_options: dict = {}       # str(idx) -> {vol, fade_in, fade_out, speed, color, flip, brightness, contrast, saturation, watermark_text}
    options_patch: dict = {}      # global effects overrides
    virtual_clips: list[dict] = [] # exact frontend timeline clips, supports split/duplicate/reorder


def _selected_from_render_body(j: dict, body: RenderRequest) -> list[dict]:
    if body.virtual_clips:
        selected = [dict(c) for c in body.virtual_clips]
        for clip in selected:
            clip["clip_duration"] = float(clip["clip_end"]) - float(clip["clip_start"])
        return selected

    all_clips = j.get("candidate_clips") or j.get("moments", [])
    selected: list[dict] = []
    for orig_i in body.approved:
        if not (0 <= orig_i < len(all_clips)):
            continue
        clip = dict(all_clips[orig_i])
        adj = body.clip_adjustments.get(str(orig_i), {})
        if adj:
            clip["clip_start"] = float(adj.get("clip_start", clip["clip_start"]))
            clip["clip_end"] = float(adj.get("clip_end", clip["clip_end"]))
            clip["clip_duration"] = clip["clip_end"] - clip["clip_start"]
        co = body.clip_options.get(str(orig_i))
        if co:
            clip["_opts"] = co
        selected.append(clip)
    return selected


def _split_shorts_parts(clips: list[dict], part_seconds: float = 58.0, min_part: float = 8.0) -> list[dict]:
    out: list[dict] = []
    part_no = 1
    for clip in clips:
        start = float(clip["clip_start"])
        end = float(clip["clip_end"])
        dur = max(0.0, end - start)
        if dur <= part_seconds + min_part:
            c = dict(clip)
            c["short_part"] = part_no
            c["short_label"] = f"Part {part_no}"
            out.append(c)
            part_no += 1
            continue
        cur = start
        while cur < end - min_part:
            nxt = min(end, cur + part_seconds)
            if end - nxt < min_part:
                nxt = end
            c = dict(clip)
            c["clip_start"] = cur
            c["clip_end"] = nxt
            c["clip_duration"] = nxt - cur
            c["short_part"] = part_no
            c["short_label"] = f"Part {part_no}"
            out.append(c)
            part_no += 1
            cur = nxt
    return out


def _cleanup_render_selection(clips: list[dict]) -> tuple[list[dict], int]:
    cleaned: list[dict] = []
    dropped = 0
    for clip in clips:
        start = float(clip.get("clip_start", 0.0))
        end = float(clip.get("clip_end", start))
        dur = max(0.0, end - start)
        if dur <= 0.15:
            dropped += 1
            continue

        src = clip.get("source_file") or clip.get("src", "")
        duplicate = False
        for kept in cleaned:
            if src != (kept.get("source_file") or kept.get("src", "")):
                continue
            ks = float(kept.get("clip_start", 0.0))
            ke = float(kept.get("clip_end", ks))
            overlap = max(0.0, min(end, ke) - max(start, ks))
            smaller = max(0.01, min(dur, max(0.0, ke - ks)))
            same_bounds = abs(start - ks) <= 1.0 and abs(end - ke) <= 1.5
            if same_bounds or overlap / smaller >= 0.82:
                duplicate = True
                break
        if duplicate:
            dropped += 1
            continue
        c = dict(clip)
        c["clip_start"] = start
        c["clip_end"] = end
        c["clip_duration"] = dur
        cleaned.append(c)
    return cleaned, dropped


@app.post("/job/{job_id}/render")
async def render_approved(job_id: str, bg: BackgroundTasks, body: RenderRequest):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    if j["status"] not in ("reviewing", "done", "error"):
        raise HTTPException(400, "Job is still running")

    selected = _selected_from_render_body(j, body)
    selected, dropped_repeats = _cleanup_render_selection(selected)

    if not selected:
        raise HTTPException(400, "No clips selected")

    opts = {**j["options"], **body.options_patch}
    j.update(status="running", moments=[], options=opts, progress=60, log=[])
    if dropped_repeats:
        _log(job_id, f"Render QA: removed {dropped_repeats} repeated clip(s)", 60)

    all_paths = j["video_paths"]
    segs_map  = {}
    model_size = opts.get("model", "small")
    for vp in all_paths:
        cached = _load_transcript_for_path(vp, model_size)
        if cached:
            segs_map[vp] = cached

    tmp = tempfile.mkdtemp(prefix=f"cf_{job_id}_render_")

    async def _do():
        try:
            await _render(job_id, selected, all_paths, segs_map, opts, tmp)
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(job_id, f"Error: {exc}", j.get("progress", 0))
            j.update(status="error", error=str(exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    bg.add_task(_do)
    return {"ok": True}


@app.post("/job/{job_id}/export-shorts")
async def export_shorts(job_id: str, bg: BackgroundTasks, body: RenderRequest):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    selected = _selected_from_render_body(j, body)
    selected, dropped_repeats = _cleanup_render_selection(selected)
    if not selected:
        raise HTTPException(400, "No clips selected")

    opts = {**j.get("options", {}), **body.options_patch, "preset": "shorts"}
    shorts_clips = _split_shorts_parts(selected)
    j.update(status="running", progress=60, log=[])
    if dropped_repeats:
        _log(job_id, f"Shorts QA: removed {dropped_repeats} repeated clip(s)", 60)

    async def _do():
        loop = asyncio.get_event_loop()
        try:
            def _render_pack():
                out_files: list[dict] = []
                total = len(shorts_clips)
                for i, clip in enumerate(shorts_clips, 1):
                    src = clip.get("source_file") or clip.get("src", "")
                    out_path = OUTPUTS / f"{job_id}_short_part_{i:03d}.mp4"
                    clip_opts = {**opts, "_opts": clip.get("_opts") or {}}
                    cut_clip(src, clip["clip_start"], clip["clip_end"], str(out_path), clip_opts)
                    out_files.append({
                        "filename": out_path.name,
                        "download_url": f"/download/{out_path.name}",
                        "duration": round(clip["clip_end"] - clip["clip_start"], 2),
                        "part": i,
                        "label": f"Part {i}",
                    })
                    _log(job_id, f"Exporting short {i}/{total}", 60 + int(i / total * 35))
                return out_files

            files = await loop.run_in_executor(None, _render_pack)
            j.update(status="done", progress=100, result={
                "filename": f"{job_id}_shorts",
                "download_url": files[0]["download_url"] if files else "",
                "duration_min": round(sum(f["duration"] for f in files) / 60, 2),
                "clips": len(files),
                "avg_score": "shorts",
                "shorts": files,
            }, moments=shorts_clips)
            _save_project(job_id)
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(job_id, f"Error: {exc}", j.get("progress", 0))
            j.update(status="error", error=str(exc))

    bg.add_task(_do)
    return {"ok": True}


@app.post("/job/{job_id}/render-both")
async def render_full_and_shorts(job_id: str, bg: BackgroundTasks, body: RenderRequest):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    if j["status"] not in ("reviewing", "done", "error"):
        raise HTTPException(400, "Job is still running")

    selected = _selected_from_render_body(j, body)
    selected, dropped_repeats = _cleanup_render_selection(selected)
    if not selected:
        raise HTTPException(400, "No clips selected")

    opts = {**j.get("options", {}), **body.options_patch}
    shorts_opts = {**opts, "preset": "shorts", "normalize_audio": True}
    shorts_clips = _split_shorts_parts(selected)
    j.update(status="running", moments=[], options=opts, progress=60, log=[])
    if dropped_repeats:
        _log(job_id, f"Render QA: removed {dropped_repeats} repeated clip(s)", 60)

    all_paths = j["video_paths"]
    segs_map = {}
    model_size = opts.get("model", "small")
    for vp in all_paths:
        cached = _load_transcript_for_path(vp, model_size)
        if cached:
            segs_map[vp] = cached

    tmp = tempfile.mkdtemp(prefix=f"cf_{job_id}_both_")

    async def _do():
        loop = asyncio.get_event_loop()
        try:
            _log(job_id, "Rendering full video...", 60)
            full_result = await _render(job_id, selected, all_paths, segs_map, opts, tmp, finalize=False)
            _log(job_id, "Rendering Shorts parts...", 88)

            def _render_pack():
                out_files: list[dict] = []
                total = len(shorts_clips)
                for i, clip in enumerate(shorts_clips, 1):
                    src = clip.get("source_file") or clip.get("src", "")
                    out_path = OUTPUTS / f"{job_id}_short_part_{i:03d}.mp4"
                    clip_opts = {**shorts_opts, "_opts": clip.get("_opts") or {}}
                    cut_clip(src, clip["clip_start"], clip["clip_end"], str(out_path), clip_opts)
                    out_files.append({
                        "filename": out_path.name,
                        "download_url": f"/download/{out_path.name}",
                        "duration": round(clip["clip_end"] - clip["clip_start"], 2),
                        "part": i,
                        "label": f"Part {i}",
                    })
                    _log(job_id, f"Rendering short part {i}/{total}", 70 + int(i / max(1, total) * 25))
                return out_files

            files = await loop.run_in_executor(None, _render_pack)
            combined = {
                **full_result,
                "full": full_result,
                "shorts": files,
                "outputs": {
                    "full": full_result,
                    "shorts": files,
                },
            }
            j.update(status="done", progress=100, result=combined, moments=selected)
            _save_project(job_id)
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(job_id, f"Error: {exc}", j.get("progress", 0))
            j.update(status="error", error=str(exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    bg.add_task(_do)
    return {"ok": True}


# ── ROUTES: RE-RENDER (post-montage effects change) ───────────────────────────

@app.post("/job/{job_id}/rerender")
async def rerender(job_id: str, bg: BackgroundTasks, options_patch: dict = Body(...)):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    if j["status"] not in ("done", "error"):
        raise HTTPException(400, "Job must be completed first")

    clips = j.get("moments", [])
    if not clips:
        raise HTTPException(400, "No clips to re-render")

    new_opts = {**j["options"], **options_patch}
    j.update(status="running", progress=60, log=[], options=new_opts)

    all_paths = j["video_paths"]
    segs_map  = {}
    model_size = new_opts.get("model", "small")
    for vp in all_paths:
        cached = _load_transcript_for_path(vp, model_size)
        if cached:
            segs_map[vp] = cached

    tmp = tempfile.mkdtemp(prefix=f"cf_{job_id}_rr_")

    async def _do():
        try:
            await _render(job_id, clips, all_paths, segs_map, new_opts, tmp)
        except Exception as exc:
            import traceback; traceback.print_exc()
            j.update(status="error", error=str(exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    bg.add_task(_do)
    return {"ok": True}


# ── ROUTES: PREVIEW / SOURCE VIDEO ───────────────────────────────────────────

@app.get("/preview/{job_id}/{idx}")
async def preview(job_id: str, idx: int):
    td   = _thumbs_dir(job_id)
    path = td / f"{idx:04d}.jpg"
    if not path.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/job/{job_id}/segments")
async def get_segments(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    result = {}
    model_size = (j.get("options") or {}).get("model", "small")
    for idx, vp in enumerate(j.get("video_paths", [])):
        cached = _load_transcript_for_path(vp, model_size)
        result[str(idx)] = cached or []
    return result


def _extract_peaks(path: str, max_peaks: int = 3000) -> dict:
    """Extract mono waveform peaks via ffmpeg. Returns {peaks:[0-255], duration:float}."""
    from editor import get_video_duration
    dur = get_video_duration(path)
    if dur <= 0:
        return {"peaks": [], "duration": 0.0}
    # Resample to ~200 Hz mono, enough resolution for any timeline zoom
    sr = 200
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
        data = proc.stdout
        if not data:
            return {"peaks": [], "duration": dur}
        n = len(data) // 2
        samples = struct.unpack(f"<{n}h", data[:n * 2])
        # Downsample to max_peaks by taking max of each chunk
        if n > max_peaks:
            chunk = n // max_peaks
            peaks = [
                int(max(abs(v) for v in samples[i:i+chunk]) / 328)
                for i in range(0, n - chunk + 1, chunk)
            ]
        else:
            peaks = [int(abs(v) / 328) for v in samples]
        # Clamp
        peaks = [min(100, p) for p in peaks]
        return {"peaks": peaks, "duration": dur}
    except Exception:
        return {"peaks": [], "duration": dur}


_WF_CACHE:     dict[str, dict] = {}   # path -> waveform data
_VISUAL_CACHE: dict[str, dict] = {}   # path -> visual analysis data


@app.get("/job/{job_id}/waveform/{source_idx}")
async def get_waveform(job_id: str, source_idx: int):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    paths = j.get("video_paths", [])
    if not (0 <= source_idx < len(paths)):
        raise HTTPException(404, "Source index out of range")
    vp = paths[source_idx]
    if vp not in _WF_CACHE:
        loop = asyncio.get_event_loop()
        _WF_CACHE[vp] = await loop.run_in_executor(None, lambda: _extract_peaks(vp))
    return JSONResponse(_WF_CACHE[vp])


@app.get("/source-video/{job_id}/{file_idx}")
async def source_video(job_id: str, file_idx: int):
    """Serve source video for in-browser preview with seeking.

    Browsers can't decode AC3/E-AC3 audio, which many ripped source files
    use, so the original would play silently. If that's the case, lazily
    build a cached copy with video stream-copied (fast) and audio
    transcoded to AAC, and serve that instead.
    """
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    paths = j.get("video_paths", [])
    if not (0 <= file_idx < len(paths)):
        raise HTTPException(404, "File index out of range")
    p = Path(paths[file_idx])
    if not p.exists():
        raise HTTPException(404, "Source file no longer available")

    serve_path = p
    loop = asyncio.get_event_loop()
    if await loop.run_in_executor(None, needs_audio_proxy, str(p)):
        proxy_path = PREVIEWS_DIR / f"{job_id}_{file_idx}.mp4"
        if not proxy_path.exists():
            await loop.run_in_executor(None, make_preview_proxy, str(p), str(proxy_path))
        serve_path = proxy_path

    return FileResponse(str(serve_path), media_type="video/mp4",
                        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"})


# ── ROUTES: TIMESTAMPS / DOWNLOAD / DELETE ────────────────────────────────────

@app.get("/job/{job_id}/timestamps")
async def timestamps(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Not found")
    clips = j.get("moments", []) or j.get("candidate_clips", [])
    lines = []
    for i, c in enumerate(clips):
        s  = _fmt_time(c.get("clip_start", c.get("start", 0)))
        e  = _fmt_time(c.get("clip_end",   c.get("end",   0)))
        sc = c.get("score", 0)
        r  = c.get("reason", "")
        sf = Path(c.get("source_file", "")).name
        src = f"  [{sf}]" if sf else ""
        lines.append(f"{i+1:2d}. {s} – {e}  score:{sc:.1f}{src}  {r}")
    return PlainTextResponse(
        "\n".join(lines),
        headers={"Content-Disposition": "attachment; filename=timestamps.txt"},
    )


@app.get("/download/{filename}")
async def download(filename: str):
    p = (OUTPUTS / filename).resolve()
    if p.parent != OUTPUTS.resolve() or not p.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), media_type="video/mp4", filename=filename)


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    j = JOBS.pop(job_id, None)
    if not j:
        raise HTTPException(404, "Not found")
    for vp in j.get("video_paths", []):
        try:
            Path(vp).unlink(missing_ok=True)
        except OSError:
            pass
    for f in [(OUTPUTS / f"{job_id}.mp4"), (OUTPUTS / f"{job_id}_thumbs")]:
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
    return {"ok": True}


# ── ROUTES: PROJECTS ─────────────────────────────────────────────────────────

@app.get("/projects")
async def list_projects():
    files = sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            moments = data.get("moments", [])
            total_dur = sum(m.get("clip_duration", 0) for m in moments)
            src_files = data.get("source_files", [])
            result.append({
                "project_id":   data["job_id"],
                "created_at":   data.get("created_at", ""),
                "video_title":  data.get("video_title", ""),
                "clip_count":   len(moments),
                "duration_min": round(total_dur / 60, 1),
                "videos_ok":    bool(src_files) and all(Path(p).exists() for p in src_files),
                "status":       data.get("status", ""),
            })
        except Exception:
            continue
    return result


@app.post("/projects/{project_id}/load")
async def load_project(project_id: str):
    try:
        uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(400, "Invalid project ID")
    path = PROJECTS_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(404, "Project not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    job_id = data["job_id"]
    if job_id not in JOBS:
        JOBS[job_id] = {
            "status":          data.get("status", "reviewing"),
            "progress":        100,
            "log":             [],
            "result":          data.get("result"),
            "error":           None,
            "moments":         data.get("moments", []),
            "candidate_clips": data.get("candidate_clips") or data.get("moments", []),
            "story_beats":     data.get("story_beats", []),
            "quality_warnings": data.get("quality_warnings", []),
            "video_paths":     data.get("source_files", []),
            "video_title":     data.get("video_title", ""),
            "options":         data.get("options", {}),
        }
        # Regenerate thumbnails asynchronously if source files exist
        src_files = data.get("source_files", [])
        moments   = data.get("moments", [])
        if src_files and moments:
            td = _thumbs_dir(job_id)
            def _regen():
                for i, clip in enumerate(moments):
                    src = clip.get("source_file", src_files[0])
                    if not Path(src).exists():
                        continue
                    thumb_path = str(td / f"{i:04d}.jpg")
                    if Path(thumb_path).exists():
                        continue
                    try:
                        mid = (clip.get("clip_start", 0) + clip.get("clip_end", 0)) / 2
                        extract_thumbnail(src, mid, thumb_path)
                    except Exception:
                        pass
            import threading
            threading.Thread(target=_regen, daemon=True).start()
    return {"job_id": job_id, "status": JOBS[job_id]["status"]}


# ── STATIC ────────────────────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse

_frontend = BASE / "frontend"

if _frontend.exists():
    @app.get("/", response_class=HTMLResponse)
    async def root():
        html = (_frontend / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
