"""
main.py — Entry point for the English Podcast Video Generator.

Usage:
    python main.py                          (interactive prompts)
    python main.py --topic "Meeting People" (CLI flags also supported)
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (one level up from this script)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("podcast")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
TEMP_DIR = BASE_DIR / "temp"


def _clean_temp():
    """Remove and recreate the temp directory."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _get_settings_interactive():
    """Prompt the user for settings interactively."""
    print()
    print("═" * 60)
    print("  🎙  ENGLISH PODCAST VIDEO GENERATOR")
    print("═" * 60)
    print()

    topic = input("  📌 Enter podcast topic: ").strip()
    while not topic:
        topic = input("  ⚠  Topic cannot be empty. Try again: ").strip()

    level = input("  📊 Enter CEFR level (A1/A2/B1/B2/C1/C2) [B1]: ").strip().upper()
    if not level:
        level = "B1"

    words_input = input("  📝 Approximate word count [2000]: ").strip()
    words = int(words_input) if words_input.isdigit() else 2000

    output = input("  💾 Output filename [output_podcast.mp4]: ").strip()
    if not output:
        output = "output_podcast.mp4"

    return topic, level, words, output


def main():
    # Support both CLI flags and interactive mode
    parser = argparse.ArgumentParser(
        description="Generate a full English podcast video from a topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  python main.py --topic "Meeting People" --level B1 --words 2000\n'
            "  python main.py          (interactive mode)\n"
        ),
    )
    parser.add_argument("--topic", default=None, help="Podcast topic")
    parser.add_argument(
        "--level", default=None, help="CEFR English level (default: B1)"
    )
    parser.add_argument(
        "--words", type=int, default=None,
        help="Approximate word count (default: 2000)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video filename (default: output_podcast.mp4)",
    )

    args = parser.parse_args()

    # If --topic was not provided, switch to interactive mode
    if args.topic is None:
        topic, level, words, output = _get_settings_interactive()
    else:
        topic = args.topic
        level = args.level or "B1"
        words = args.words or 2000
        output = args.output or "output_podcast.mp4"

    t_start = time.time()

    print()
    print("═" * 60)
    print("  🎙  ENGLISH PODCAST VIDEO GENERATOR")
    print("═" * 60)
    print(f"  Topic : {topic}")
    print(f"  Level : {level}")
    print(f"  Words : ~{words}")
    print(f"  Output: {output}")
    print("═" * 60)
    print()

    # ── Step 1: Generate script ──────────────────────────────────────
    logger.info("━━━ STEP 1/4: Generating podcast script via Claude API ━━━")
    from script_generator import generate_script

    script = generate_script(topic, level, words)
    turns = script["turns"]
    total_words = sum(len(t["text"].split()) for t in turns)

    logger.info("Title: \"%s\"", script["title"])
    logger.info("Turns: %d  |  Words: ~%d", len(turns), total_words)

    # Save script to temp for reference
    _clean_temp()
    script_path = TEMP_DIR / "script.json"
    script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Script saved: %s", script_path)

    # ── Step 2: Text-to-Speech ───────────────────────────────────────
    logger.info("━━━ STEP 2/4: Synthesizing speech with XTTS v2 ━━━")
    from tts_engine import synthesize_turns

    audio_paths = synthesize_turns(turns, VOICES_DIR, TEMP_DIR)
    logger.info("Generated %d audio segments.", len(audio_paths))

    # ── Step 3: Audio mixing ─────────────────────────────────────────
    logger.info("━━━ STEP 3/4: Mixing audio ━━━")
    from audio_mixer import concatenate_audio

    combined_audio = TEMP_DIR / "combined.wav"
    concatenate_audio(audio_paths, combined_audio, gap_ms=400)

    # ── Step 4: Video generation ─────────────────────────────────────
    logger.info("━━━ STEP 4/4: Building video ━━━")
    from video_builder import build_video

    output_path = Path(output)
    build_video(turns, audio_paths, combined_audio, output_path, gap_ms=400)

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print("═" * 60)
    print(f"  ✅  DONE in {_format_elapsed(elapsed)}")
    print(f"  📹  Output: {output_path.resolve()}")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
