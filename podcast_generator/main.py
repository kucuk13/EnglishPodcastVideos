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

    topic = input("  📌 Enter podcast topic [Meeting New People]: ").strip()
    if not topic:
        topic = "Meeting New People"

    level = input("  📊 Enter CEFR level (A1/A2/B1/B2/C1/C2) [B1]: ").strip().upper()
    if not level:
        level = "B1"

    words_input = input("  📝 Approximate word count [2000]: ").strip()
    words = int(words_input) if words_input.isdigit() else 2000

    output = input("  💾 Output filename [output_podcast.mp4]: ").strip()
    if not output:
        output = "output_podcast.mp4"

    tts_input = input("  🔊 TTS engine — 1: edge-tts (free)  2: OpenAI TTS [1]: ").strip()
    tts_type = 2 if tts_input == "2" else 1

    return topic, level, words, output, tts_type


def step1_generate_script(topic: str, level: str, words: int) -> dict:
    """Generate podcast script via Claude API. Saves to temp/script.json."""
    logger.info("━━━ STEP 1/5: Generating podcast script via Claude API ━━━")
    from script_generator import generate_script

    _clean_temp()
    script = generate_script(topic, level, words)
    turns = script["turns"]
    total_words = sum(len(t["text"].split()) for t in turns)

    logger.info("Title: \"%s\"", script["title"])
    logger.info("Turns: %d  |  Words: ~%d", len(turns), total_words)

    (TEMP_DIR / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Script saved: %s", TEMP_DIR / "script.json")
    return script


def step2_synthesize_speech(tts_type: int = 1) -> list[Path]:
    """Synthesize TTS audio. Reads temp/script.json, saves temp/audio_paths.json."""
    engine_name = "OpenAI TTS" if tts_type == 2 else "edge-tts"
    logger.info("━━━ STEP 2/5: Synthesizing speech with %s ━━━", engine_name)
    from tts_engine import synthesize_turns

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    audio_paths = synthesize_turns(script["turns"], VOICES_DIR, TEMP_DIR, tts_type=tts_type)

    (TEMP_DIR / "audio_paths.json").write_text(
        json.dumps([str(p) for p in audio_paths], indent=2), encoding="utf-8"
    )
    logger.info("Generated %d audio segments.", len(audio_paths))
    return audio_paths


def step3_mix_audio() -> Path:
    """Concatenate per-turn audio. Reads temp/audio_paths.json, writes temp/combined.wav."""
    logger.info("━━━ STEP 3/5: Mixing audio ━━━")
    from audio_mixer import concatenate_audio

    audio_paths = [
        Path(p)
        for p in json.loads((TEMP_DIR / "audio_paths.json").read_text(encoding="utf-8"))
    ]
    combined_audio = TEMP_DIR / "combined.wav"
    concatenate_audio(audio_paths, combined_audio, gap_ms=400)
    return combined_audio


def step4_generate_background() -> Path | None:
    """Generate AI background image via OpenAI GPT-IMAGE-1. Saves to temp/background.png.

    Returns the image path, or None if OPENAI_API_KEY is not set (graceful fallback).
    """
    logger.info("━━━ STEP 4/5: Generating background image via GPT-IMAGE-1 ━━━")
    from image_generator import generate_background_image

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    bg_path = TEMP_DIR / "background.png"

    try:
        generate_background_image(script["title"], script.get("topic", ""), bg_path)
        return bg_path
    except RuntimeError as exc:
        logger.warning("Background image skipped: %s", exc)
        return None


def step5_build_video(output: str | Path) -> Path:
    """Render the final video. Reads all inputs from temp/. Returns output path."""
    logger.info("━━━ STEP 5/5: Building video ━━━")
    from video_builder import build_video

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    audio_paths = [
        Path(p)
        for p in json.loads((TEMP_DIR / "audio_paths.json").read_text(encoding="utf-8"))
    ]
    combined_audio = TEMP_DIR / "combined.wav"
    background_path = TEMP_DIR / "background.png"

    output_path = Path(output)
    build_video(
        script["turns"],
        audio_paths,
        combined_audio,
        output_path,
        gap_ms=400,
        background_image_path=background_path if background_path.exists() else None,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a full English podcast video from a topic. (default: Meeting New People)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  python main.py --topic "Meeting New People" --level B1 --words 2000\n'
            "  python main.py          (interactive mode)\n"
        ),
    )
    parser.add_argument("--topic", default=None, help="Podcast topic")
    parser.add_argument("--level", default=None, help="CEFR English level (default: B1)")
    parser.add_argument("--words", type=int, default=None, help="Approximate word count (default: 2000)")
    parser.add_argument("--output", default=None, help="Output video filename (default: output_podcast.mp4)")
    parser.add_argument("--tts", type=int, choices=[1, 2], default=1, help="TTS engine: 1=edge-tts (default), 2=OpenAI TTS")

    args = parser.parse_args()

    if args.topic is None:
        topic, level, words, output, tts_type = _get_settings_interactive()
    else:
        topic = args.topic
        level = args.level or "B1"
        words = args.words or 2000
        output = args.output or "output_podcast.mp4"
        tts_type = args.tts

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

    step1_generate_script(topic, level, words)
    step2_synthesize_speech(tts_type=tts_type)
    step3_mix_audio()
    step4_generate_background()
    output_path = step5_build_video(output)

    elapsed = time.time() - t_start
    print()
    print("═" * 60)
    print(f"  ✅  DONE in {_format_elapsed(elapsed)}")
    print(f"  📹  Output: {output_path.resolve()}")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
