# Viral Clipper

Turn a long YouTube video (or any video file) into short vertical 9:16 clips with the most viral potential. It adds a camera that follows the speaker, animated captions, and a hook title, then saves everything to your PC.

Everything runs locally for free. No paid API key is needed.

> Only process videos you own or have permission to use. Downloading other people's videos breaks YouTube's Terms of Service and reposting them can get your accounts copyright strikes.

---

## 1. Install (Windows)

**Step 1 — Python.** Install Python **3.11** from <https://www.python.org/downloads/>. On the first installer screen, tick **"Add python.exe to PATH"**.

**Step 2 — FFmpeg.** Open *Command Prompt* and run:
```
winget install Gyan.FFmpeg
```
Close the window and open a new one so the PATH updates. Check it with `ffmpeg -version`.

**Step 3 — The free AI (Ollama).** Install Ollama from <https://ollama.com/download>. Then download a model:
```
ollama pull qwen2.5:7b
```
If your PC has less than 16 GB RAM, use `qwen2.5:3b` instead, and set that name in `config.yaml` or in the app.

**Step 4 — This tool.** Double-click **`install.bat`**. It creates a private Python environment and installs everything.

**Step 5 — Run.** Double-click **`run.bat`**. The app opens in your browser at `http://127.0.0.1:7860`. Keep the black window open while you use it.

On macOS or Linux, install `ffmpeg` with your package manager, then run:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 2. Use it

The page works top to bottom:

1. **Video** — paste one or more YouTube links (one per line), or drop video files. A single link shows a preview, and you can download the whole video to your PC at 720p, 1080p, 2K or 4K (only the qualities that video really has are offered).
2. **Cut the part you want** — drag the two handles to keep only a part of the video. Clips are taken only from that part. This appears when you have one video loaded.
3. **Settings** — number of clips, length, caption style and language, post-caption language, speaker size, vertical framing, captions and hook title on or off. The AI and speech options are under "AI and speech settings".
4. **Find clips** — the button sits at the bottom, after everything is set.

### Projects

Every video becomes its own **project**. Projects run one at a time in a queue, and you can keep adding videos while one is working. Clips appear in the list as soon as each one is finished, so you don't have to wait for the whole batch.

| Status | What you can do |
|---|---|
| Working | Watch live progress, open the clips made so far, or **Stop** |
| In queue | **Remove** it from the queue |
| Ready | **Open**, **Remake** with the current settings, or **Delete** |
| Failed / Stopped / Interrupted | **Try again** or **Delete** |

Projects are saved in `outputs/projects.json` and survive restarts. Clips made with older versions show up automatically.

### Each clip

Opening a project shows every clip fully expanded with:
- the vertical video, its viral score, length, layout and how much of the time the speaker stayed in frame,
- an **AI-written caption with 5 hashtags**, written for search (SEO), with a **Copy** button,
- **Framing** controls: nudge the frame left, right, up, down, or zoom in and out, then **Apply and re-render**. This uses the saved framing plan, so it re-renders in seconds without analysing the video again,
- the transcript, a download button, and "Show in folder".

Clips are saved in `outputs/<video name>_<project id>/` as `clip_01_<title>.mp4` with a matching `.txt` file holding the caption, hashtags, score and transcript.

## 3. Framing, captions and languages

**Vertical framing**

| Layout | What it does |
|---|---|
| `auto` | Chooses per shot (default) |
| `track` | The crop follows the active speaker, with smooth pans and hard cuts on speaker changes |
| `split` | Two speakers stacked top and bottom, each tracked |
| `face_screen` | Speaker on top, the screen or slide below |
| `fit` | The whole 16:9 frame over a blurred background |

**Speaker size (zoom):** `auto`, `wide` (full body), `medium` (chest up), `close` (face).

**Caption styles:** Clean, Bold box, Neon glow, Gold, Minimal, Karaoke. Each has its own font, colours and animation.

**Caption language:** `auto` (as spoken), English, Urdu (Urdu script) or Roman Urdu. English captions for an Urdu video use Whisper's own translation. Roman Urdu is transliterated word by word, so every word keeps its timing and the word-by-word highlight still works. Urdu captions use a proper Urdu font, which the tool downloads by itself.

**Post caption:** the AI writes the caption and 5 hashtags for each clip in `auto` (English video → English, Urdu video → Roman Urdu), or in the language you pick.

## 4. Settings (`config.yaml`)

The most useful ones:

- `transcription.model`: `small` is a good balance. `medium` or `large-v3` are more accurate, especially for Urdu and Hindi, but slower.
- `transcription.language`: set `"ur"`, `"hi"` or `"en"` to skip auto-detect.
- `llm.provider`: `ollama` (free, local), `openai` (any OpenAI-compatible API, including Groq, OpenRouter, and Gemini), `anthropic` (Claude), or `none` (built-in heuristic, no AI).
- `reframe.dead_zone`, `max_pan_speed`, `smoothing_seconds`: control how the virtual camera moves.
- `captions.style` / `captions.language`, and the overrides below them (font, size, colours) if you want to change a style.
- `reframe.zoom`: `auto`, `wide`, `medium`, `close`.
- `post_caption.language`: language of the AI-written caption and hashtags.
- `output.crf` / `preset`: file quality and render speed on the CPU. `veryfast` is recommended.
- `output.encoder`: `auto` uses your graphics card's video encoder (NVIDIA, Intel or AMD) when it works, which is much faster; otherwise it uses the CPU. You can force `nvenc`, `qsv`, `amf` or `cpu`.

To use a paid API, put the key in `config.yaml`, or set it as an environment variable (`LLM_API_KEY` or `ANTHROPIC_API_KEY`).

## 5. How it works

```
source ─► downloader.py   yt-dlp download (or local file)
       ─► transcriber.py  faster-whisper, word-level timestamps (cached)
       ─► selector.py     AI reads transcript in chunks → scored clips → snapped to sentences
       ─► per clip:
            reframer.py   scene cuts · face detection (YuNet/Haar, quick ROI scans) ·
                          tracking · mouth-movement speaker detection · layout choice ·
                          zoom level · virtual camera (dead zone + speed limit + smoothing)
                          → the plan is saved, so framing fixes need no re-analysis
            language.py   Roman Urdu transliteration, caption language planning
            captions.py   caption styles + word-by-word highlighted .ass subtitles
            renderer.py   FFmpeg-only fast path for single-layout clips, otherwise
                          per-frame in Python (H.264 + AAC, captions burned in)
```

```
viral_clipper/
├── app.py            web app server (open http://127.0.0.1:7860)
├── static/           the web interface (index.html)
├── clipper.py        command-line interface
├── config.yaml       settings
├── core/             the engine (files above)
├── downloads/        source videos
├── outputs/          finished clips
└── .cache/           transcripts and models
```

## 6. Troubleshooting

- **"ffmpeg was not found"**: install FFmpeg (step 2) and open a new terminal.
- **"AI request failed"**: check that Ollama is running (its icon is in the system tray) and that you pulled the model named in the settings. The tool automatically falls back to the built-in scorer, so you still get clips.
- **Making clips is slow**: set `preset: veryfast` in `config.yaml`, and check the progress text. It says `Rendering (nvenc, fast)` or `(cpu, fast)`, which shows which encoder is used. Clips that are entirely "follow speaker" or "full frame" render fastest; split-screen clips take longer.
- **Transcription is very slow**: use the `base` or `tiny` model. On an NVIDIA GPU, Whisper uses CUDA automatically if the CUDA 12 and cuDNN libraries are installed. Otherwise it runs on CPU.
- **Download errors**: YouTube changes often. Update the downloader with `pip install -U yt-dlp`.
- **Captions missing**: your FFmpeg build lacks `libass`. The Gyan build from winget includes it.
- **Urdu captions look broken**: update to this version. Two bugs caused it: letter spacing in the caption style (Urdu letters cannot join when it is on) and Whisper splitting Urdu words into pieces. Both are fixed, and Urdu fonts download automatically.
- **Wrong person followed**: set the layout to `split` for two-person videos, or lower `speaker_hold_seconds` so the camera switches faster.
