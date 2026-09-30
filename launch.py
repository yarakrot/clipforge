#!/usr/bin/env python3
"""
ClipForge launcher
Checks deps → warms up ffmpeg → starts server → opens browser
"""
import os
import sys
import time
import socket
import tempfile
import threading
import webbrowser
import subprocess
from pathlib import Path

BASE = Path(__file__).parent

# System TEMP can fill up the C: drive (uploads/ffmpeg spill large files there).
# Redirect TEMP/TMP to a folder on this project's own drive before anything
# (pip, ffmpeg, Starlette's multipart parser) touches the temp dir.
TMP_DIR = BASE / "tmp"
TMP_DIR.mkdir(exist_ok=True)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BANNER = r"""
  ╔══════════════════════════════════════╗
  ║                                      ║
  ║   ✂  ClipForge  —  Auto Montage     ║
  ║                                      ║
  ╚══════════════════════════════════════╝
"""

REQUIRED = {
    "av": "av==16.1.0",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "faster_whisper": "faster-whisper",
    "httpx": "httpx",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "multipart": "python-multipart",
    "dotenv": "python-dotenv",
    "yt_dlp": "yt-dlp",
}


def check_and_install():
    missing = []
    for module, pkg in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=True
        )
        print("  Done.\n")


def load_env():
    env_file = BASE / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            key = os.environ.get("DEEPSEEK_API_KEY", "")
            if key:
                print("  API key loaded from .env")
        except Exception:
            pass


def warm_ffmpeg():
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    name = os.path.basename(exe)
    print(f"  ffmpeg: {name}")


def find_free_port(preferred: int = 8000, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in {preferred}-{preferred + attempts - 1}")


def open_browser_delayed(port: int):
    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    print(BANNER)

    print("  Checking dependencies...")
    check_and_install()

    print("  Checking ffmpeg...", end=" ", flush=True)
    warm_ffmpeg()

    load_env()

    port = find_free_port(8000)
    if port != 8000:
        print(f"  Port 8000 is busy, using {port}")

    threading.Thread(target=open_browser_delayed, args=(port,), daemon=True).start()

    print()
    print(f"  ► http://localhost:{port}")
    print("  ► Ctrl+C to stop")
    print()

    backend = BASE / "backend"
    os.chdir(str(backend))
    sys.path.insert(0, str(backend))

    import asyncio
    import uvicorn

    # Suppress spurious Windows ProactorEventLoop "connection reset" noise
    # that fires when a browser closes a connection mid-stream.
    def _silence_conn_reset(loop, ctx):
        exc = ctx.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return
        loop.default_exception_handler(ctx)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_silence_conn_reset)
        asyncio.set_event_loop(loop)
        config = uvicorn.Config("main:app", host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n  Stopped. Goodbye!")


if __name__ == "__main__":
    main()
