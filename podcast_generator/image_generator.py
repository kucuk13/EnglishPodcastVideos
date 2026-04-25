"""
image_generator.py — OpenAI gpt-image-1 background image generation.

Generates a thematic 1280x720 background image for the podcast video.
"""

import base64
import io
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def generate_background_image(title: str, topic: str, output_path: Path) -> Path:
    """Generate a thematic background image via OpenAI gpt-image-1.

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
        f"Two friendly characters (a man and a woman) having a simple everyday conversation about {topic}."
        "Scene includes a clear background related to the topic (cafe, airport, park, shop, etc)."
        "Simple educational ESL comic illustration style."
        "No speech bubbles."
        "Cartoon style, 16:9 aspect ratio."
    )

    logger.info("Calling gpt-image-1 to generate background image…")
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1536x1024",
        quality="medium",
        n=1,
    )

    image_data = base64.b64decode(response.data[0].b64_json)
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((1280, 720), Image.LANCZOS)
    img.save(output_path, "PNG")

    logger.info("Background image saved: %s", output_path)
    return output_path
