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

    level = input("  📊 Enter CEFR level (A1/A2/B1/B2/C1/C2) [A2]: ").strip().upper()
    if not level:
        level = "A2"

    words_input = input("  📝 Approximate word count [2000]: ").strip()
    words = int(words_input) if words_input.isdigit() else 2000

    output = input("  💾 Output filename [output_podcast.mp4]: ").strip()
    if not output:
        output = "output_podcast.mp4"

    llm_input = input("  🤖 LLM engine  — 1: Claude         2: OpenAI       [1]: ").strip()
    llm_type = 2 if llm_input == "2" else 1

    tts_input = input("  🔊 TTS engine — 1: edge-tts (free)  2: OpenAI TTS [2]: ").strip()
    tts_type = 1 if tts_input == "1" else 2

    mode_input = input("  ⚙  Mode         — 1: Auto (no prompts)  2: Step-by-step [1]: ").strip()
    mode = 2 if mode_input == "2" else 1

    return topic, level, words, output, tts_type, llm_type, mode


def _run_step(step_fn, *args, mode: int = 1, **kwargs):
    while True:
        result = step_fn(*args, **kwargs)
        if mode == 1:
            return result
        answer = input("  ➡  Devam edilsin mi? (y/n) [y]: ").strip().lower()
        if answer != "n":
            return result
        print("  🔄  Adım tekrar yapılıyor...")


def step1_generate_script(topic: str, level: str, words: int, llm_type: int) -> dict:
    """Generate podcast script via Claude API. Saves to temp/script.json."""
    logger.info("━━━ STEP 1/8: Generating podcast script via API ━━━")
    from script_generator import generate_script

    _clean_temp()
    script = generate_script(topic, level, words, llm_type)
    turns = script["turns"]
    total_words = sum(len(t["text"].split()) for t in turns)

    logger.info("Title: \"%s\"", script["title"])
    logger.info("Turns: %d  |  Words: ~%d", len(turns), total_words)

    (TEMP_DIR / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Script saved: %s", TEMP_DIR / "script.json")
    return script


def step2_generate_youtube_description(llm_type: int = 1) -> str:
    """Generate a short YouTube video description via LLM from the full script."""
    logger.info("━━━ STEP 2/8: Generating YouTube video description ━━━")
    from script_generator import generate_youtube_description

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    description = generate_youtube_description(script, llm_type)

    if len(description) > 250:
        description = description[:247].rstrip(" .,!?;") + "..."

    description_path = TEMP_DIR / "youtube_description.txt"
    description_path.write_text(description, encoding="utf-8")
    logger.info("YouTube description: %s", description)
    logger.info("Saved: %s", description_path)
    return description


def step3_generate_youtube_keywords(llm_type: int = 1) -> str:
    """Generate SEO-friendly YouTube keywords via LLM from the full script."""
    logger.info("━━━ STEP 3/8: Generating SEO-friendly YouTube keywords ━━━")
    from script_generator import generate_youtube_keywords

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    keywords_text = generate_youtube_keywords(script, llm_type)

    (TEMP_DIR / "youtube_keywords.txt").write_text(keywords_text, encoding="utf-8")
    logger.info("YouTube keywords: %s", keywords_text)
    logger.info("Saved: %s", TEMP_DIR / "youtube_keywords.txt")
    return keywords_text


def step4_synthesize_speech(tts_type: int = 1) -> list[Path]:
    """Synthesize TTS audio. Reads temp/script.json, saves temp/audio_paths.json."""
    engine_name = "OpenAI TTS" if tts_type == 2 else "edge-tts"
    logger.info("━━━ STEP 4/8: Synthesizing speech with %s ━━━", engine_name)
    from tts_engine import synthesize_turns

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    audio_paths = synthesize_turns(script["turns"], VOICES_DIR, TEMP_DIR, tts_type=tts_type)

    (TEMP_DIR / "audio_paths.json").write_text(
        json.dumps([str(p) for p in audio_paths], indent=2), encoding="utf-8"
    )
    logger.info("Generated %d audio segments.", len(audio_paths))
    return audio_paths


def step5_mix_audio() -> Path:
    """Concatenate per-turn audio. Reads temp/audio_paths.json, writes temp/combined.wav."""
    logger.info("━━━ STEP 5/8: Mixing audio ━━━")
    from audio_mixer import concatenate_audio

    audio_paths = [
        Path(p)
        for p in json.loads((TEMP_DIR / "audio_paths.json").read_text(encoding="utf-8"))
    ]
    combined_audio = TEMP_DIR / "combined.wav"
    concatenate_audio(audio_paths, combined_audio, gap_ms=200)
    return combined_audio


def step6_generate_background() -> Path | None:
    """Generate AI background image via OpenAI GPT-IMAGE-1. Saves to temp/background.png.

    Returns the image path, or None if OPENAI_API_KEY is not set (graceful fallback).
    """
    logger.info("━━━ STEP 6/8: Generating background image via GPT-IMAGE-1 ━━━")
    from image_generator import generate_background_image

    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    bg_path = TEMP_DIR / "background.png"

    try:
        generate_background_image(script["title"], script.get("topic", ""), bg_path)
        return bg_path
    except RuntimeError as exc:
        logger.warning("Background image skipped: %s", exc)
        return None


def step7_generate_thumbnail(level: str, llm_type: int = 1) -> Path | None:
    """Generate a click-worthy thumbnail over the step6 background image."""
    logger.info("━━━ STEP 7/8: Generating clickable thumbnail ━━━")
    script = json.loads((TEMP_DIR / "script.json").read_text(encoding="utf-8"))
    bg_path = TEMP_DIR / "background.png"
    thumbnail_path = TEMP_DIR / "thumbnail.png"

    if not bg_path.exists():
        logger.warning("Thumbnail skipped: background image not available.")
        return None

    from PIL import Image, ImageDraw, ImageFont
    from script_generator import generate_thumbnail_headline

    try:
        image = Image.open(bg_path).convert("RGB")
    except Exception as exc:
        logger.warning("Thumbnail creation failed: %s", exc)
        return None

    # LLM-generated click-worthy headline, max 5 words, all caps
    try:
        headline = generate_thumbnail_headline(script, llm_type)
        headline_words = headline.split()
        if len(headline_words) > 5:
            headline = " ".join(headline_words[:5])
        headline = headline.upper().strip('"\'"')
    except Exception:
        headline = " ".join(script.get("title", "English Podcast").upper().split()[:5])

    width, height = image.size  # 1280x720

    # Dark gradient overlay on bottom half (keeps background visible on top)
    gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_grad = ImageDraw.Draw(gradient)
    for y in range(height // 2, height):
        alpha = int(220 * (y - height // 2) / max(height // 2 - 1, 1))
        draw_grad.line([(0, y), (width - 1, y)], fill=(0, 0, 0, alpha))
    image = Image.alpha_composite(image.convert("RGBA"), gradient).convert("RGB")

    # Find the boldest available font
    font_path = None
    for name in ["impact.ttf", "arialbd.ttf", "arial.ttf"]:
        candidate = Path("C:/Windows/Fonts") / name
        if candidate.exists():
            font_path = str(candidate)
            break

    try:
        title_font = ImageFont.truetype(font_path or "arial", 90)
        level_font = ImageFont.truetype(font_path or "arial", 42)
    except Exception:
        title_font = ImageFont.load_default()
        level_font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    # Headline with thick outline for contrast
    text_x = 50
    text_y = height - 190
    for dx, dy in [(-4, -4), (-4, 4), (4, -4), (4, 4), (-4, 0), (4, 0), (0, -4), (0, 4)]:
        draw.text((text_x + dx, text_y + dy), headline, font=title_font, fill=(0, 0, 0))
    draw.text((text_x, text_y), headline, font=title_font, fill=(255, 255, 255))

    # Level badge — top-right corner, color-coded by CEFR level
    level_colors = {
        "A1": (76, 175, 80),
        "A2": (139, 195, 74),
        "B1": (255, 193, 7),
        "B2": (255, 152, 0),
        "C1": (244, 67, 54),
        "C2": (156, 39, 176),
    }
    badge_color = level_colors.get(level.upper(), (255, 193, 7))
    badge_text = f"LEVEL {level.upper()}"
    padding = 18
    try:
        bbox = draw.textbbox((0, 0), badge_text, font=level_font)
        bw = bbox[2] - bbox[0] + padding * 2
        bh = bbox[3] - bbox[1] + padding * 2
    except AttributeError:
        bw, bh = 160, 60  # fallback for old Pillow
    bx = width - bw - 30
    by = 30
    try:
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14, fill=badge_color)
    except AttributeError:
        draw.rectangle([bx, by, bx + bw, by + bh], fill=badge_color)
    draw.text((bx + padding, by + padding), badge_text, font=level_font, fill=(255, 255, 255))

    image.save(thumbnail_path, format="PNG")
    logger.info("Thumbnail saved: %s", thumbnail_path)
    return thumbnail_path


def step8_build_video(output: str | Path) -> Path:
    """Render the final video. Reads all inputs from temp/. Returns output path."""
    logger.info("━━━ STEP 8/8: Building video ━━━")
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
        gap_ms=200,
        background_image_path=background_path if background_path.exists() else None,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a full English podcast video from a topic. (default: Meeting New People)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  python main.py --topic "Meeting New People" --level A2 --words 2000\n'
            "  python main.py          (interactive mode)\n"
        ),
    )
    parser.add_argument("--topic", default=None, help="Podcast topic")
    parser.add_argument("--level", default=None, help="CEFR English level (default: A2)")
    parser.add_argument("--words", type=int, default=None, help="Approximate word count (default: 2000)")
    parser.add_argument("--output", default=None, help="Output video filename (default: output_podcast.mp4)")
    parser.add_argument("--llm", type=int, choices=[1, 2], default=1, help="LLM engine: 1=Claude (default), 2=OpenAI")
    parser.add_argument("--tts", type=int, choices=[1, 2], default=2, help="TTS engine: 1=edge-tts, 2=OpenAI TTS (default)")
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1, help="Run mode: 1=auto (default), 2=step-by-step")

    args = parser.parse_args()

    if args.topic is None:
        topic, level, words, output, tts_type, llm_type, mode = _get_settings_interactive()
    else:
        topic = args.topic
        level = args.level or "A2"
        words = args.words or 2000
        output = args.output or "output_podcast.mp4"
        llm_type = args.llm
        tts_type = args.tts
        mode = args.mode

    t_start = time.time()

    print()
    print("═" * 60)
    print("  🎙  ENGLISH PODCAST VIDEO GENERATOR")
    print("═" * 60)
    print(f"  Topic : {topic}")
    print(f"  Level : {level}")
    print(f"  Words : ~{words}")
    print(f"  Output: {output}")
    print(f"  LLLM Type: {llm_type}")
    print(f"  TTS Type : {tts_type}")
    print("═" * 60)
    print()

    _run_step(step1_generate_script, topic, level, words, llm_type, mode=mode)
    _run_step(step2_generate_youtube_description, llm_type, mode=mode)
    _run_step(step3_generate_youtube_keywords, llm_type, mode=mode)
    _run_step(step4_synthesize_speech, tts_type=tts_type, mode=mode)
    _run_step(step5_mix_audio, mode=mode)
    _run_step(step6_generate_background, mode=mode)
    _run_step(step7_generate_thumbnail, level, llm_type, mode=mode)
    output_path = _run_step(step8_build_video, output, mode=mode)

    elapsed = time.time() - t_start
    print()
    print("═" * 60)
    print(f"  ✅  DONE in {_format_elapsed(elapsed)}")
    print(f"  📹  Output: {output_path.resolve()}")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
