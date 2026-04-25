"""
audio_mixer.py — Concatenate TTS audio segments with pydub.

Joins all per-turn wav files into a single audio track, inserting
configurable silence gaps between turns.
"""

import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def concatenate_audio(
    wav_paths: list[Path],
    output_path: str | Path,
    gap_ms: int = 400,
) -> Path:
    """Concatenate wav files into a single audio file.

    Args:
        wav_paths: Ordered list of wav file paths to join.
        output_path: Path for the combined output audio file.
        gap_ms: Silence gap (in milliseconds) to insert between turns.

    Returns:
        Path to the combined audio file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    silence = AudioSegment.silent(duration=gap_ms)
    combined = AudioSegment.empty()

    logger.info("Concatenating %d audio segments (gap=%d ms)…", len(wav_paths), gap_ms)

    for idx, wav_path in enumerate(wav_paths):
        segment = AudioSegment.from_file(str(wav_path))
        logger.debug(
            "Segment %d: %s — %.1f s",
            idx, wav_path.name, len(segment) / 1000.0,
        )

        if idx > 0:
            combined += silence
        combined += segment

    total_duration = len(combined) / 1000.0
    logger.info("Combined audio: %.1f s (%.1f min)", total_duration, total_duration / 60)

    combined.export(str(output_path), format="wav")
    logger.info("Combined audio saved to: %s", output_path)

    return output_path


def get_segment_durations(wav_paths: list[Path]) -> list[float]:
    """Get the duration (in seconds) of each wav segment.

    Args:
        wav_paths: List of wav file paths.

    Returns:
        List of durations in seconds.
    """
    durations = []
    for wav_path in wav_paths:
        segment = AudioSegment.from_file(str(wav_path))
        durations.append(len(segment) / 1000.0)
    return durations


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python audio_mixer.py output.wav segment1.wav segment2.wav ...")
        sys.exit(1)

    out = sys.argv[1]
    inputs = [Path(p) for p in sys.argv[2:]]
    concatenate_audio(inputs, out)
