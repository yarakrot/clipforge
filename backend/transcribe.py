from typing import Optional

from faster_whisper import WhisperModel

_cache: dict = {}
_device_cache: Optional[tuple] = None


def _pick_device() -> tuple:
    """Detect whether a CUDA GPU is usable by ctranslate2; fall back to CPU.

    GPU + float16 gives both faster *and* more accurate transcription than
    CPU + int8, since int8 quantization trades some recognition quality for
    speed. We only fall back to CPU when CUDA truly isn't usable.
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    try:
        WhisperModel("tiny", device="cuda", compute_type="float16")
        _device_cache = ("cuda", "float16")
        print("[Whisper] CUDA GPU detected — using float16.")
    except Exception:
        _device_cache = ("cpu", "int8")
        print("[Whisper] No usable CUDA GPU — using CPU int8.")
    return _device_cache


def get_model(size: str) -> WhisperModel:
    if size not in _cache:
        device, compute_type = _pick_device()
        print(f"[Whisper] Loading model '{size}' on {device}/{compute_type}...")
        try:
            _cache[size] = WhisperModel(size, device=device, compute_type=compute_type)
        except Exception:
            _cache[size] = WhisperModel(size, device="cpu", compute_type="int8")
        print(f"[Whisper] Model ready.")
    return _cache[size]


def transcribe_video(
    video_path: str,
    model_size: str = "small",
    language: str = None,
    on_progress=None,           # callback(msg: str, pct: int)  pct = 0..100
    duration_hint: float = 0,   # total video seconds, for progress calc
) -> list[dict]:
    model = get_model(model_size)
    beam = 5 if model_size in {"small", "medium", "large-v2", "large-v3"} else 3
    opts = {
        "beam_size": beam,
        "vad_filter": True,
        # Shorter min-silence than the 2000 ms default so segments break at real
        # speech pauses instead of merging several sentences into one block —
        # downstream boundary-snapping relies on segment edges as a fallback.
        "vad_parameters": {"min_silence_duration_ms": 500},
        "word_timestamps": True,
        "condition_on_previous_text": True,
        # Suppresses Whisper's known repeated-phrase hallucination loops on
        # noisy/unclear audio without affecting normal speech.
        "no_repeat_ngram_size": 3,
    }
    if language:
        opts["language"] = language

    segments_iter, info = model.transcribe(video_path, **opts)

    # use detected duration if not provided
    total = duration_hint or (info.duration if hasattr(info, "duration") and info.duration else 0)

    segments = []
    last_pct = -1

    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            seg_dict = {
                "start": round(seg.start, 3),
                "end":   round(seg.end,   3),
                "text":  text,
            }
            # Compact word timestamps with text for punctuation-aware boundaries.
            if seg.words:
                seg_dict["words"] = [
                    {
                        "s": round(w.start, 3),
                        "e": round(w.end, 3),
                        "word": w.word.strip(),
                    }
                    for w in seg.words
                    if w.word.strip()
                ]
            segments.append(seg_dict)

        if on_progress and total > 0:
            pct = min(99, int(seg.end / total * 100))
            if pct != last_pct:
                last_pct = pct
                elapsed_min = int(seg.end // 60)
                total_min   = int(total   // 60)
                on_progress(
                    f"Transcribing... {elapsed_min}/{total_min} min",
                    pct,
                )

    if on_progress:
        on_progress(f"Transcription complete — {len(segments)} segments", 100)

    return segments
