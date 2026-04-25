"""
image_generator.py — OpenAI DALL-E 3 background image generation.

Generates a thematic 1280x720 background image for the podcast video.
"""

import io
import logging
import os
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def generate_background_image(title: str, topic: str, output_path: Path) -> Path:
    """Generate a thematic background image via OpenAI DALL-E 3.

    Args:
        title: Episode title used to craft the generation prompt.
        topic: Podcast topic.
        output_path: Destination path for the saved PNG (resized to 1280x720).

    Returns:
        Path to the saved PNG file.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it in your .env file to enable background image generation."
        )

    client = openai.OpenAI(api_key=api_key)

    prompt = (
        f"A visually stunning, wide landscape background image for an English-learning podcast episode. "
        f"Episode title: '{title}'. Topic: '{topic}'. "
        "Style: cinematic, modern, slightly dark/moody so overlaid white text remains readable. "
        "Abstract or thematic illustration — no text, no letters, no people, no faces. "
        "Rich colors, depth, and visual interest. 16:9 aspect ratio."
    )

    logger.info("Calling DALL-E 3 to generate background image…")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    logger.info("Downloading generated image from OpenAI…")
    img_bytes = requests.get(image_url, timeout=60)
    img_bytes.raise_for_status()

    img = Image.open(io.BytesIO(img_bytes.content)).convert("RGB")
    img = img.resize((1280, 720), Image.LANCZOS)
    img.save(output_path, "PNG")

    logger.info("Background image saved: %s", output_path)
    return output_path
