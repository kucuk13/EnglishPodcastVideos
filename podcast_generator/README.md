# English Podcast Video Generator

Generate a full podcast-style video from a topic, English level, and word count.

**Pipeline:** Claude API → Coqui XTTS v2 → pydub → moviepy → MP4

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- Anthropic API key (set `ANTHROPIC_API_KEY` environment variable)

## Setup

```bash
cd podcast_generator
pip install -r requirements.txt
```

### Voice Reference Files

Place two short (5–15 s) clean speech `.wav` samples in the `voices/` folder:

```
voices/
├── voice_alex.wav
└── voice_jordan.wav
```

These are used by XTTS v2 for voice cloning — each speaker will sound like the corresponding reference.

## Usage

```bash
python main.py --topic "Meeting People" --level B1 --words 2000
```

### CLI Arguments

| Argument    | Default              | Description                          |
|-------------|----------------------|--------------------------------------|
| `--topic`   | *(required)*         | Podcast topic                        |
| `--level`   | `B1`                 | CEFR English level (A1–C2)          |
| `--words`   | `2000`               | Approximate word count for dialogue  |
| `--output`  | `output_podcast.mp4` | Output video filename                |

## Output

A 1280×720 MP4 video with:
- Speaker name and dialogue text on screen
- Subtitle bar at the bottom
- Animated pulsing circle for visual flair
- Cloned voices for both speakers
