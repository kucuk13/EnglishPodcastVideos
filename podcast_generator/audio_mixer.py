"""
audio_mixer.py — Concatenate TTS audio segments with pydub.

Joins all per-turn wav files into a single audio track, inserting
configurable silence gaps between turns.

Also normalizes each segment for more consistent YouTube voice volume.
"""

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

logger = logging.getLogger(__name__)


TARGET_DBFS = -18.0
SEGMENT_PEAK_HEADROOM_DB = -1.5


def _prepare_segment(segment: AudioSegment) -> AudioSegment:
    """Normalize one TTS segment to a comfortable voice level."""
    segment = normalize(segment, headroom=SEGMENT_PEAK_HEADROOM_DB)

    if segment.dBFS != float("-inf"):
        change_db = TARGET_DBFS - segment.dBFS
        segment = segment.apply_gain(change_db)

    return segment


def concatenate_audio(
    wav_paths: list[Path],
    output_path: str | Path,
    gap_ms: int = 350,
) -> Path:
    """Concatenate wav files into a single audio file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    silence = AudioSegment.silent(duration=gap_ms)
    combined = AudioSegment.empty()

    logger.info("Concatenating %d audio segments (gap=%d ms)…", len(wav_paths), gap_ms)

    for idx, wav_path in enumerate(wav_paths):
        segment = AudioSegment.from_file(str(wav_path))
        segment = _prepare_segment(segment)

        logger.debug(
            "Segment %d: %s — %.1f s — %.1f dBFS",
            idx,
            wav_path.name,
            len(segment) / 1000.0,
            segment.dBFS,
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
    """Get the duration in seconds of each wav segment."""
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
