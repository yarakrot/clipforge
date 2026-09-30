import os
import re
import yt_dlp


def download_video(url: str, out_dir: str, on_progress=None) -> dict:
    """
    Downloads video from URL using yt-dlp.
    Returns {"path": str, "title": str, "duration": float, "thumbnail": str}
    """
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    result = {}

    def progress_hook(d):
        if d["status"] == "downloading" and on_progress:
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed") or 0
            if total > 0:
                pct = int(downloaded / total * 100)
                speed_mb = speed / 1_048_576
                on_progress(f"Downloading... {pct}% ({speed_mb:.1f} MB/s)", pct)
        elif d["status"] == "finished":
            result["path"] = d["filename"]
            if on_progress:
                on_progress("Download complete, processing...", 100)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "writeinfojson": False,
        "writethumbnail": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # use prepared filename from yt-dlp
        path = ydl.prepare_filename(info)
        # yt-dlp may change extension after merge
        if not os.path.exists(path):
            base = os.path.splitext(path)[0]
            for ext in ("mp4", "mkv", "webm"):
                candidate = f"{base}.{ext}"
                if os.path.exists(candidate):
                    path = candidate
                    break

        return {
            "path": path,
            "title": info.get("title", "video"),
            "duration": float(info.get("duration") or 0),
            "uploader": info.get("uploader") or info.get("channel", ""),
            "thumbnail": info.get("thumbnail", ""),
        }


def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))
