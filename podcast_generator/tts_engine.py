"""
tts_engine.py — TTS voice synthesis per dialogue turn.

Supports two backends:
  1 — edge-tts (free, no API key required)
  2 — OpenAI TTS API (requires OPENAI_API_KEY)
"""

import asyncio
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# edge-tts neural voices per speaker
VOICE_MAP = {
    "Jack": "en-US-GuyNeural",
    "Amy": "en-US-AriaNeural",
}

# OpenAI TTS voices per speaker
OPENAI_VOICE_MAP = {
    "Jack": "onyx",
    "Amy": "nova",
}


async def _synthesize_all(turns: list[dict], temp_dir: Path) -> list[Path]:
    import edge_tts

    output_paths: list[Path] = []
    for idx, turn in enumerate(tqdm(turns, desc="TTS synthesis", unit="turn")):
        speaker = turn["speaker"]
        voice = VOICE_MAP.get(speaker, VOICE_MAP["Jack"])
        if speaker not in VOICE_MAP:
            logger.warning("Unknown speaker '%s' — falling back to Jack's voice.", speaker)

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
    tts_type: int = 1,
) -> list[Path]:
    """Synthesize each dialogue turn to an mp3 file.

    Args:
        turns: List of dicts with 'speaker' and 'text' keys.
        _voices_dir: Unused — kept for API compatibility.
        temp_dir: Path to directory for temporary output files.
        _language: Unused — voices are language-specific by name.
        tts_type: 1 = edge-tts (free), 2 = OpenAI TTS API.

    Returns:
        Ordered list of Path objects pointing to generated mp3 files.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if tts_type == 2:
        logger.info("Synthesizing %d dialogue turns with OpenAI TTS…", len(turns))
        output_paths = _synthesize_all_openai(turns, temp_dir)
    else:
        logger.info("Synthesizing %d dialogue turns with edge-tts…", len(turns))
        output_paths = asyncio.run(_synthesize_all(turns, temp_dir))

    logger.info("All %d turns synthesized successfully.", len(output_paths))
    return output_paths


def _synthesize_all_openai(turns: list[dict], temp_dir: Path) -> list[Path]:
    import os
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot use OpenAI TTS (type 2).")

    client = openai.OpenAI(api_key=api_key)
    output_paths: list[Path] = []

    for idx, turn in enumerate(tqdm(turns, desc="OpenAI TTS", unit="turn")):
        speaker = turn["speaker"]
        voice = OPENAI_VOICE_MAP.get(speaker, OPENAI_VOICE_MAP["Jack"])
        if speaker not in OPENAI_VOICE_MAP:
            logger.warning("Unknown speaker '%s' — falling back to Jack's voice.", speaker)

        output_path = temp_dir / f"turn_{idx:04d}_{speaker.lower()}.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=turn["text"],
        )
        response.stream_to_file(str(output_path))
        output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    sample_turns = [
        {"speaker": "Jack", "text": "Hello and welcome to our podcast!"},
        {"speaker": "Amy", "text": "Thanks for having me, it's great to be here."},
    ]
    paths = synthesize_turns(sample_turns, "voices", "temp")
    for p in paths:
        print(f"  → {p}")
