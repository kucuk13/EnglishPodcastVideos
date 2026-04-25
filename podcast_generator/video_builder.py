"""
video_builder.py — moviepy video assembly for podcast.

Creates a 1280x720 video with:
- Speaker name and word-wrapped dialogue text
- Animated pulsing circle near speaker label
- Subtitle bar at the bottom
"""

import logging
import math
import textwrap
from pathlib import Path

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Video dimensions
WIDTH = 1280
HEIGHT = 720
FPS = 24

# Colors
BG_COLOR = (18, 18, 24)
SPEAKER_COLORS = {
    "Alex": (100, 200, 255),      # Soft blue
    "Jordan": (255, 160, 100),    # Warm orange
}
TEXT_COLOR = (230, 230, 240)
SUBTITLE_BG = (0, 0, 0)
SUBTITLE_TEXT_COLOR = (255, 255, 255)


_WINDOWS_FONT_DIR = Path("C:/Windows/Fonts")
_FONT_CANDIDATES = {
    "regular": ["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"],
    "bold":    ["arialbd.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}


def _resolve_font(bold: bool = False) -> str:
    """Return an absolute font path that Pillow can open, or a bare name as fallback."""
    candidates = _FONT_CANDIDATES["bold" if bold else "regular"]
    for name in candidates:
        path = _WINDOWS_FONT_DIR / name
        if path.exists():
            return str(path)
    # Last resort: let moviepy/pillow try by name (may still fail)
    return "arial"


FONT_REGULAR = _resolve_font(bold=False)
FONT_BOLD    = _resolve_font(bold=True)


def _word_wrap(text: str, width: int = 50) -> str:
    """Wrap text to the given character width."""
    return "\n".join(textwrap.wrap(text, width=width))


def _create_pulsing_circle_frame(
    t: float,
    color: tuple[int, int, int],
    base_radius: int = 18,
    pulse_amplitude: int = 6,
    pulse_speed: float = 2.5,
    size: int = 60,
) -> np.ndarray:
    """Create a single frame of an animated pulsing circle (RGBA, uint8)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    radius = base_radius + pulse_amplitude * math.sin(2 * math.pi * pulse_speed * t)
    radius = max(4, int(radius))
    center = size // 2

    # Outer glow
    glow_radius = radius + 6
    draw.ellipse(
        [center - glow_radius, center - glow_radius,
         center + glow_radius, center + glow_radius],
        fill=(*color, 60),
    )

    # Main circle
    draw.ellipse(
        [center - radius, center - radius,
         center + radius, center + radius],
        fill=(*color, 220),
    )

    return np.array(img)  # (H, W, 4) RGBA — transparent background


def _build_turn_clip(
    turn: dict,
    duration: float,
    gap_s: float,
    background_image_path: "Path | None" = None,
) -> CompositeVideoClip:
    """Build a single clip for one dialogue turn.

    Args:
        turn: Dict with 'speaker' and 'text'.
        duration: Duration of this turn's audio in seconds.
        gap_s: Inter-turn gap in seconds (added after audio).
        background_image_path: Optional path to a 1280x720 background image.
    """
    speaker = turn["speaker"]
    text = turn["text"]
    total_duration = duration + gap_s

    color = SPEAKER_COLORS.get(speaker, (200, 200, 200))

    # --- Background ---
    if background_image_path is not None:
        bg = ImageClip(str(background_image_path)).with_duration(total_duration)
        # Semi-transparent dark overlay so text stays readable
        overlay = (
            ColorClip(size=(WIDTH, HEIGHT), color=(0, 0, 0))
            .with_opacity(0.55)
            .with_duration(total_duration)
        )
        base_layers = [bg, overlay]
    else:
        base_layers = [ColorClip(size=(WIDTH, HEIGHT), color=BG_COLOR).with_duration(total_duration)]

    # --- Speaker label ---
    speaker_label = TextClip(
        text=f"  {speaker}",
        font_size=42,
        color=f"rgb({color[0]},{color[1]},{color[2]})",
        font=FONT_BOLD,
        size=(WIDTH - 200, None),
        method="caption",
    ).with_duration(total_duration).with_position(("center", 60))

    # --- Pulsing circle (animated, transparent background) ---
    circle_size = 60

    def make_circle_rgb(t):
        return _create_pulsing_circle_frame(t, color, size=circle_size)[..., :3]

    def make_circle_mask(t):
        return _create_pulsing_circle_frame(t, color, size=circle_size)[..., 3] / 255.0

    pulsing_circle = (
        VideoClip(make_circle_rgb, duration=total_duration)
        .with_mask(VideoClip(make_circle_mask, is_mask=True, duration=total_duration))
        .with_position((100, 60))
    )

    # --- Main dialogue text (word-wrapped, centered) ---
    wrapped_text = _word_wrap(text, width=55)
    main_text = TextClip(
        text=wrapped_text,
        font_size=30,
        color=f"rgb({TEXT_COLOR[0]},{TEXT_COLOR[1]},{TEXT_COLOR[2]})",
        font=FONT_REGULAR,
        size=(WIDTH - 160, None),
        method="caption",
        text_align="center",
    ).with_duration(total_duration).with_position(("center", 180))

    # --- Subtitle bar at bottom ---
    sub_bg_color = tuple(int(c * 0.25) for c in SUBTITLE_BG)
    subtitle_bg = ColorClip(
        size=(WIDTH, 90), color=sub_bg_color
    ).with_duration(total_duration).with_position(("center", HEIGHT - 90))

    subtitle_text_str = text if len(text) <= 120 else text[:117] + "…"
    subtitle_text = TextClip(
        text=subtitle_text_str,
        font_size=22,
        color=f"rgb({SUBTITLE_TEXT_COLOR[0]},{SUBTITLE_TEXT_COLOR[1]},{SUBTITLE_TEXT_COLOR[2]})",
        font=FONT_REGULAR,
        size=(WIDTH - 80, None),
        method="caption",
        text_align="center",
    ).with_duration(total_duration).with_position(("center", HEIGHT - 75))

    layers = [*base_layers, speaker_label, main_text, subtitle_bg, subtitle_text]

    try:
        layers.insert(len(base_layers), pulsing_circle)
    except Exception:
        logger.debug("Pulsing circle skipped (rendering issue).")

    return CompositeVideoClip(layers, size=(WIDTH, HEIGHT)).with_duration(total_duration)



def build_video(
    turns: list[dict],
    audio_paths: list[Path],
    combined_audio_path: str | Path,
    output_path: str | Path,
    gap_ms: int = 400,
    background_image_path: "Path | None" = None,
) -> Path:
    """Assemble the final podcast video.

    Args:
        turns: List of dialogue turn dicts.
        audio_paths: Per-turn wav paths (used for measuring durations).
        combined_audio_path: Path to the full concatenated audio.
        output_path: Path for the output mp4 file.
        gap_ms: Inter-turn silence gap in milliseconds.
        background_image_path: Optional 1280x720 image used as video background.

    Returns:
        Path to the output video file.
    """
    from audio_mixer import get_segment_durations

    output_path = Path(output_path)
    combined_audio_path = Path(combined_audio_path)
    gap_s = gap_ms / 1000.0

    if background_image_path:
        logger.info("Using background image: %s", background_image_path)

    logger.info("Measuring segment durations…")
    durations = get_segment_durations(audio_paths)

    logger.info("Building video clips for %d turns…", len(turns))
    clips = []
    for idx, (turn, dur) in enumerate(zip(turns, durations)):
        logger.debug("Clip %d: %s — %.1f s", idx, turn["speaker"], dur)
        clip = _build_turn_clip(
            turn,
            dur,
            gap_s if idx < len(turns) - 1 else 0,
            background_image_path=background_image_path,
        )
        clips.append(clip)

    logger.info("Concatenating video clips…")
    video = concatenate_videoclips(clips, method="compose")

    # Attach audio
    logger.info("Attaching audio track…")
    audio = AudioFileClip(str(combined_audio_path))

    # If there's a duration mismatch, trim to the shorter one
    final_duration = min(video.duration, audio.duration)
    video = video.subclipped(0, final_duration)
    audio = audio.subclipped(0, final_duration)
    video = video.with_audio(audio)

    total_min = final_duration / 60.0
    logger.info("Final video: %.1f s (%.1f min)", final_duration, total_min)

    logger.info("Exporting video to: %s", output_path)
    video.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        logger="bar",
    )

    logger.info("✅ Video exported: %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("video_builder.py — run via main.py for full pipeline.")
