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
    "Jack": (100, 200, 255),      # Soft blue
    "Amy": (255, 160, 100),    # Warm orange
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
    pulse_speed: float = 1.0,
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
        base_layers = [bg]
    else:
        base_layers = [ColorClip(size=(WIDTH, HEIGHT), color=BG_COLOR).with_duration(total_duration)]

    # --- Bottom bar ---
    BAR_HEIGHT = 210
    bar_y = HEIGHT - BAR_HEIGHT  # 510

    bar_bg = (
        ColorClip(size=(WIDTH, BAR_HEIGHT), color=(0, 0, 0))
        .with_opacity(0.72)
        .with_duration(total_duration)
        .with_position(("left", bar_y))
    )

    # --- Pulsing dot (left of bar, vertically centered with speaker name) ---
    circle_size = 72
    dot_x = 44
    LABEL_FONT_SIZE = 32
    HEADER_CENTER_Y = bar_y + 46   # shared vertical center for dot and label
    dot_y = HEADER_CENTER_Y - circle_size // 2   # top of 72px canvas → center lands on HEADER_CENTER_Y

    def make_circle_rgb(t):
        return _create_pulsing_circle_frame(t, color, size=circle_size)[..., :3]

    def make_circle_mask(t):
        return _create_pulsing_circle_frame(t, color, size=circle_size)[..., 3] / 255.0

    pulsing_circle = (
        VideoClip(make_circle_rgb, duration=total_duration)
        .with_mask(VideoClip(make_circle_mask, is_mask=True, duration=total_duration))
        .with_position((dot_x, dot_y))
    )

    # --- Speaker name (next to dot, top aligned so text center ≈ HEADER_CENTER_Y) ---
    speaker_label = TextClip(
        text=speaker,
        font_size=LABEL_FONT_SIZE,
        color=f"rgb({color[0]},{color[1]},{color[2]})",
        font=FONT_BOLD,
        size=(400, None),
        method="caption",
    ).with_duration(total_duration).with_position((dot_x + circle_size + 12, HEADER_CENTER_Y - LABEL_FONT_SIZE // 2))

    # --- Subtitle text ---
    subtitle_text = TextClip(
        text=text,
        font_size=26,
        color=f"rgb({SUBTITLE_TEXT_COLOR[0]},{SUBTITLE_TEXT_COLOR[1]},{SUBTITLE_TEXT_COLOR[2]})",
        font=FONT_REGULAR,
        size=(WIDTH - 80, 130),
        method="caption",
        text_align="left",
        vertical_align="top",
    ).with_duration(total_duration).with_position((40, bar_y + 82))

    layers = [*base_layers, bar_bg, speaker_label, subtitle_text]
    try:
        layers.insert(len(base_layers) + 1, pulsing_circle)
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
    from audio_mixer import get_segment_durations

    output_path = Path(output_path)
    combined_audio_path = Path(combined_audio_path)
    gap_s = gap_ms / 1000.0

    if background_image_path:
        logger.info("Using background image: %s", background_image_path)

    logger.info("Measuring segment durations…")
    durations = get_segment_durations(audio_paths)

    if len(durations) != len(turns):
        raise ValueError(f"turns={len(turns)} but durations={len(durations)}")

    clips = []
    try:
        logger.info("Building video clips for %d turns…", len(turns))

        for idx, (turn, dur) in enumerate(zip(turns, durations)):
            clip = _build_turn_clip(
                turn,
                dur,
                gap_s if idx < len(turns) - 1 else 0,
                background_image_path=background_image_path,
            )
            clips.append(clip)

        logger.info("Concatenating video clips…")
        video = concatenate_videoclips(clips, method="chain")

        # Attach audio
        logger.info("Attaching audio track…")
        audio = AudioFileClip(str(combined_audio_path))

        final_duration = video.duration

        video = video.subclipped(0, final_duration)

        logger.info("Exporting video to: %s", output_path)

        video.write_videofile(
            str(output_path),
            fps=FPS,
            codec="h264_nvenc", #libx264
            audio_codec="aac",
            preset="p1",
            audio=str(combined_audio_path),
            ffmpeg_params=[
                "-rc", "vbr",
                "-cq", "28",
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
            logger="bar",
        )

        logger.info("✅ Video exported: %s", output_path)
        return output_path

    finally:
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

        try:
            video.close()
        except Exception:
            pass

        try:
            audio.close()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("video_builder.py — run via main.py for full pipeline.")
