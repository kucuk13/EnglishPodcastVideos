"""
tts_engine.py — Microsoft Edge TTS voice synthesis per dialogue turn.

Uses edge-tts (free, Python 3.12 compatible) with distinct neural voices
per speaker. No reference audio or GPU required.
"""

import asyncio
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Preset neural voices per speaker
VOICE_MAP = {
    "Alex": "en-US-GuyNeural",
    "Jordan": "en-US-AriaNeural",
}


async def _synthesize_all(turns: list[dict], temp_dir: Path) -> list[Path]:
    import edge_tts

    output_paths: list[Path] = []
    for idx, turn in enumerate(tqdm(turns, desc="TTS synthesis", unit="turn")):
        speaker = turn["speaker"]
        voice = VOICE_MAP.get(speaker, VOICE_MAP["Alex"])
        if speaker not in VOICE_MAP:
            logger.warning("Unknown speaker '%s' — falling back to Alex's voice.", speaker)

        output_path = temp_dir / f"turn_{idx:04d}_{speaker.lower()}.mp3"
        communicate = edge_tts.Communicate(turn["text"], voice)
        await communicate.save(str(output_path))
        output_paths.append(output_path)

    return output_paths


def synthesize_turns(
    turns: list[dict],
    _voices_dir: str | Path,
    temp_dir: str | Path,
    _language: str = "en",
) -> list[Path]:
    """Synthesize each dialogue turn to an mp3 file using edge-tts.

    Args:
        turns: List of dicts with 'speaker' and 'text' keys.
        voices_dir: Unused — kept for API compatibility with main.py.
        temp_dir: Path to directory for temporary output files.
        language: Unused — edge-tts voices are language-specific by name.

    Returns:
        Ordered list of Path objects pointing to generated mp3 files.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Synthesizing %d dialogue turns with edge-tts…", len(turns))
    output_paths = asyncio.run(_synthesize_all(turns, temp_dir))
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
