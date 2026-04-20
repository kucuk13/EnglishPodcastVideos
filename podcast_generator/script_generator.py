"""
script_generator.py — Claude API dialogue generation.

Calls claude-sonnet-4-20250514 to produce a two-person podcast dialogue
in strict JSON format.
"""

import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional podcast scriptwriter for an English-learning podcast.
Your task is to write a natural, engaging two-person dialogue between two hosts:
Alex and Jordan.

RULES:
- The dialogue MUST be educational and appropriate for the given CEFR level.
- Use everyday vocabulary and natural conversational patterns.
- Include useful expressions, idioms (if level-appropriate), and varied sentence structures.
- Both speakers should contribute equally.
- The dialogue should flow naturally — include filler words, reactions, and follow-up questions.
- Aim for the specified approximate word count.

OUTPUT FORMAT — you MUST respond with ONLY raw JSON, no markdown fences, no preamble:
{
  "title": "<episode title>",
  "turns": [
    {"speaker": "Alex", "text": "..."},
    {"speaker": "Jordan", "text": "..."}
  ]
}
"""


def generate_script(
    topic: str,
    level: str = "B1",
    words: int = 2000,
    max_retries: int = 2,
) -> dict:
    """Generate a podcast script via Claude API.

    Args:
        topic: The podcast topic.
        level: CEFR English level (A1-C2).
        words: Approximate target word count.
        max_retries: Number of retries on JSON parse failure.

    Returns:
        A dict with keys "title" and "turns".

    Raises:
        ValueError: If the response cannot be parsed after retries.
        RuntimeError: If the API key is missing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before running the script generator."
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = (
        f"Write a podcast dialogue about the topic: \"{topic}\".\n"
        f"Target CEFR level: {level}.\n"
        f"Approximate word count: {words} words.\n\n"
        f"Remember: output ONLY raw JSON, no markdown, no extra text."
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        logger.info("Claude API call attempt %d/%d …", attempt, max_retries)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = message.content[0].text.strip()
        logger.debug("Raw response length: %d chars", len(raw_text))

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("JSON parse failed (attempt %d): %s", attempt, exc)
            # Try to extract JSON from the response if wrapped in markdown
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    continue
            else:
                continue

        # Validate schema
        if "title" not in data or "turns" not in data:
            last_error = ValueError("Missing 'title' or 'turns' in response")
            logger.warning("Schema validation failed (attempt %d)", attempt)
            continue

        if not isinstance(data["turns"], list) or len(data["turns"]) == 0:
            last_error = ValueError("'turns' must be a non-empty list")
            logger.warning("Empty turns (attempt %d)", attempt)
            continue

        for i, turn in enumerate(data["turns"]):
            if "speaker" not in turn or "text" not in turn:
                last_error = ValueError(f"Turn {i} missing 'speaker' or 'text'")
                logger.warning("Turn %d invalid (attempt %d)", i, attempt)
                break
        else:
            # All turns valid
            total_words = sum(len(t["text"].split()) for t in data["turns"])
            logger.info(
                "Script generated: \"%s\" — %d turns, ~%d words",
                data["title"],
                len(data["turns"]),
                total_words,
            )
            return data

    raise ValueError(
        f"Failed to get valid JSON from Claude after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    script = generate_script("Meeting People", "B1", 2000)
    print(json.dumps(script, indent=2, ensure_ascii=False))
