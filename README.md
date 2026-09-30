# ClipForge

Turn long recordings into editable highlights and condensed stories with local speech recognition, AI-assisted clip selection, and FFmpeg rendering.

ClipForge is a single-user desktop workflow with a browser interface. Built with Python, FastAPI, faster-whisper, FFmpeg, and vanilla JavaScript. The interface is currently in Russian.

## Features

- Import one or more local videos, or download from supported URLs where permitted.
- Transcribe recordings locally with faster-whisper, using CUDA when available and CPU otherwise.
- Use DeepSeek to suggest highlights, story beats, and clip plans.
- Review clips in a timeline, adjust boundaries, and undo editing actions.
- Render full edits and vertical short clips with optional subtitles, transitions, watermarks, and audio normalization.
- Save recent projects locally and export timestamps.

## Requirements

- Python 3.12 is recommended.
- Internet access for dependency installation, initial model downloads, and DeepSeek requests.
- A DeepSeek API key for AI analysis. API usage may incur charges.
- Enough free disk space for source videos, temporary files, caches, and rendered output.

FFmpeg is provided through imageio-ffmpeg. Some URL downloads may need a separate FFmpeg installation to merge video and audio streams. On Windows:

```powershell
winget install --id Gyan.FFmpeg --exact
```

Open a new terminal after installing FFmpeg. CUDA acceleration requires compatible hardware and the runtime libraries required by faster-whisper/CTranslate2.

## Quick Start on Windows

```powershell
git clone https://github.com/yarakrot/clipforge.git
cd clipforge
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Replace the placeholder in `.env` with your own DeepSeek key, then run:

```powershell
.\.venv\Scripts\python.exe launch.py
```

You can also double-click `launch.bat` after installation. The launcher opens the local browser interface and chooses a free port starting at 8000. Stop the application with Ctrl+C in its terminal.

Alternatively, save your key in the interface's DeepSeek API Key panel. This writes it to the local `.env` file. The launcher can install missing dependencies into its current Python environment; using the virtual environment above is recommended.

## Workflow

1. Add local recordings that you own or have permission to process.
2. Choose the transcription model, language, edit style, and target duration.
3. Start analysis and review the suggested clips and story structure.
4. Adjust the timeline and render an edit or shorts.
5. Download results from the interface. Processing files remain in local project folders.

## Privacy and Local Storage

Speech recognition runs locally. AI analysis sends transcript text and editing prompts to the DeepSeek API; do not process confidential recordings without permission to send that information to the provider. URL imports contact the source service.

- `.env` stores your DeepSeek key in plaintext on your computer.
- `uploads/` stores imported recordings.
- `outputs/` stores rendered results and thumbnails.
- `cache/` stores transcripts, analysis, previews, and recent project metadata, including local source paths.
- `tmp/` stores temporary processing files.
- The browser stores editing preferences and session state in local storage.

These runtime folders and secret configuration files are excluded from Git. Do not upload them manually or include them in public screenshots. Deleting a job can delete its imported source copies and rendered results; it does not guarantee removal of every cached item.

## Security and Intended Use

The launcher binds to `127.0.0.1`. Host validation and browser request checks reject unexpected hosts, cross-site requests, and writes without the application's request header. These controls are not user authentication: other processes on your computer can still access the local service.

Use this as a local, single-user tool. Do not expose it with a tunnel or deploy it as a public service without adding authentication, request and storage limits, and isolated processing. Keep dependencies up to date.

URL downloading and video editing do not grant rights to the source material. Follow source-service terms and copyright permissions. Changing speed, color, or orientation does not remove those obligations.

## Development and Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q backend launch.py
```

The security tests cover allowed local requests, cross-site blocking, host validation, and protection against simple cross-site writes. A complete video workflow also requires a model, an API key, and suitable recordings.

## Project Structure

```text
backend/           API, transcription, AI analysis, and rendering
frontend/          Browser interface
tests/             Local request security tests
launch.py          Local server and browser launcher
launch.bat         Windows launcher
requirements.txt   Python dependencies
.env.example       Safe configuration template
```

## License

MIT — see [LICENSE](LICENSE). Dependencies and any downloaded model weights have their own licenses and terms.
