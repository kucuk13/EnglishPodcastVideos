"""
tts_engine.py — Coqui XTTS v2 voice synthesis per dialogue turn.

Loads the XTTS v2 model once and synthesizes each turn using the
matching speaker's reference voice file.
"""

import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Map speaker names to voice reference filenames
VOICE_MAP = {
    "Alex": "voice_alex.wav",
    "Jordan": "voice_jordan.wav",
}


def synthesize_turns(
    turns: list[dict],
    voices_dir: str | Path,
    temp_dir: str | Path,
    language: str = "en",
) -> list[Path]:
    """Synthesize each dialogue turn to a wav file using XTTS v2.

    Args:
        turns: List of dicts with 'speaker' and 'text' keys.
        voices_dir: Path to directory containing reference voice wav files.
        temp_dir: Path to directory for temporary output wav files.
        language: Language code for TTS (default: "en").

    Returns:
        Ordered list of Path objects pointing to generated wav files.
    """
    from TTS.api import TTS

    voices_dir = Path(voices_dir)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Validate voice files exist
    for speaker, filename in VOICE_MAP.items():
        voice_path = voices_dir / filename
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Reference voice file not found: {voice_path}\n"
                f"Please place a short .wav sample for {speaker} at this path."
            )

    logger.info("Loading XTTS v2 model (this may take a moment on first run)…")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    # Move to GPU if available
    try:
        tts.to("cuda")
        logger.info("Using GPU for TTS inference.")
    except Exception:
        logger.info("GPU not available, using CPU (this will be slower).")

    output_paths: list[Path] = []

    logger.info("Synthesizing %d dialogue turns…", len(turns))
    for idx, turn in enumerate(tqdm(turns, desc="TTS synthesis", unit="turn")):
        speaker = turn["speaker"]
        text = turn["text"]

        voice_filename = VOICE_MAP.get(speaker)
        if voice_filename is None:
            logger.warning(
                "Unknown speaker '%s' — falling back to Alex's voice.", speaker
            )
            voice_filename = VOICE_MAP["Alex"]

        reference_wav = str(voices_dir / voice_filename)
        output_path = temp_dir / f"turn_{idx:04d}_{speaker.lower()}.wav"

        logger.debug(
            "Turn %d: %s — %d chars, ref=%s",
            idx, speaker, len(text), voice_filename,
        )

        tts.tts_to_file(
            text=text,
            speaker_wav=reference_wav,
            language=language,
            file_path=str(output_path),
        )

        output_paths.append(output_path)

    logger.info("All %d turns synthesized successfully.", len(output_paths))
    return output_paths


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    sample_turns = [
        {"speaker": "Alex", "text": "Hello and welcome to our podcast!"},
        {"speaker": "Jordan", "text": "Thanks for having me, it's great to be here."},
    ]
    paths = synthesize_turns(sample_turns, "voices", "temp")
    for p in paths:
        print(f"  → {p}")
