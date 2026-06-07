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
        f"Two friendly characters having a simple everyday conversation about {topic}. "

        "Amy is positioned on the far LEFT side of the frame. "
        "Jack is positioned on the far RIGHT side of the frame. "
        "Never swap their positions. "
        
        "Both characters are visible from the waist up. "
        "They are facing each other naturally. "

        "The background clearly represents the topic "
        "(cafe, airport, park, office, restaurant, shop, hotel, etc). "

        "Bright vibrant colors. "
        "Modern educational illustration. "
        "High contrast. "
        "Clean composition. "
        "Professional YouTube educational style. "

        "No speech bubbles. "
        "No text. "
        "No captions. "
        "16:9 aspect ratio."
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

def generate_thumbnail_image(
    title: str,
    topic: str,
    level: str,
    background_path: Path,
    output_path: Path,
) -> Path:
    """
    Generate a clickable YouTube thumbnail using the existing background image.

    Uses GPT-IMAGE-1 image editing to transform the background into
    a high-CTR thumbnail.
    """

    import openai

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = openai.OpenAI(api_key=api_key)

    thumbnail_prompt = f"""
Create a HIGH-CTR YouTube thumbnail using the uploaded image as the base background.

VIDEO INFO:
- Topic: {topic}
- Title: {title}
- Level: {level}

STYLE:
- Cartoon English-learning YouTube channel
- Modern clickable YouTube thumbnail
- Warm cozy lighting
- Bold thick text
- Mobile-friendly readability
- Strong contrast
- Expressive characters
- Fun and energetic

THUMBNAIL RULES:
- Maximum 3-5 words
- HUGE readable text
- Black brush background behind text
- White + yellow + red text colors
- Large expressive faces
- Slight zoom-in on characters
- Keep background visible but slightly blurred
- Add YouTube-style energy
- Keep composition clean
- Avoid clutter
- Avoid tiny details

IMPORTANT:
- Make it look like a viral English learning thumbnail
- Similar to modern YouTube educational thumbnails
- 16:9 composition
"""

    logger.info("Calling gpt-image-1 to generate thumbnail…")

    with open(background_path, "rb") as image_file:

        response = client.images.edit(
            model="gpt-image-1",
            image=image_file,
            prompt=thumbnail_prompt,
            size="1536x1024",
        )

    image_data = base64.b64decode(response.data[0].b64_json)

    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((1280, 720), Image.LANCZOS)
    img.save(output_path, "PNG")

    logger.info("Thumbnail saved: %s", output_path)

    return output_path