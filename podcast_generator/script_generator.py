import json
import logging
import os
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert scriptwriter for a YouTube English-learning channel.

CHANNEL POSITIONING:
- A2/B1 English learners
- Real-life English conversations
- Natural speaking practice
- Everyday situations: coffee shop, first meeting, airport, shopping, work, small talk
- The viewer understands some English but freezes when speaking

SCRIPT RULES:
- The dialogue MUST be educational and appropriate for the given CEFR level.
- Make the situation clear in the first 3 seconds.
- Start with a strong hook, not a slow introduction.
- Avoid generic podcast openings like "Welcome back to our podcast".
- Use everyday vocabulary and natural conversational patterns.
- Include useful expressions, fillers, reactions, and follow-up questions.
- Both speakers should contribute equally.
- Keep the conversation emotionally realistic: nervous, awkward, curious, friendly, relieved, confident.
- Add light pacing cues through natural reactions, but do not write camera directions.
- Aim for the specified approximate word count.

OUTPUT FORMAT — respond with ONLY raw JSON, no markdown fences, no preamble:
{
  "title": "<episode title>",
  "topic": "<input topic>",
  "level": "<CEFR level>",
  "hook": "<first 1-2 lines that immediately create curiosity>",
  "key_expressions": ["...", "...", "..."],
  "turns": [
    {"speaker": "Jack", "text": "..."},
    {"speaker": "Amy", "text": "..."}
  ]
}
"""

PACKAGING_SYSTEM_PROMPT = """\
You are a YouTube growth expert specialized in English-learning channels.

Generate clickable YouTube packaging for A2/B1 ESL learners.

IMPORTANT STRATEGY:
- Sell the real-life result, not a boring lesson.
- Make the viewer feel: "I need this in real life."
- Prefer specific scenarios over generic wording.
- Avoid generic phrases as the main hook: English Podcast, Learn English, Conversation Practice.
- A2/B1 can be included near the end of titles, not as the main hook.

THUMBNAIL RULES:
- 2-4 words only.
- One idea only.
- Very readable on mobile.
- Emotional, clear, practical.
- Good words: REAL, SPEAK, NATURAL, AWKWARD, FAST, CONFIDENT, FIRST, STOP, WHY, SHY, SMALL TALK, CAN'T SPEAK.

TITLE RULES:
- Under 70 characters if possible.
- Curiosity + practical value.
- Human and emotional.
- Do not sound like a textbook.

Return ONLY raw JSON:
{
  "recommended_title": "...",
  "thumbnail_text": "...",
  "alternate_titles": ["...", "...", "..."],
  "alternate_thumbnail_texts": ["...", "...", "..."],
  "video_angle": "...",
  "first_5_seconds_hook": "...",
  "description": "...",
  "keywords": ["...", "..."],
  "pinned_comment": "..."
}
"""


def _extract_json(raw_text: str) -> dict[str, Any] | None:
    """Parse raw JSON; fall back to the first JSON-looking object."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None


def _parse_and_validate(raw_text: str, attempt: int) -> dict[str, Any] | None:
    """Parse JSON from raw LLM response and validate script schema."""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("JSON parse failed (attempt %d)", attempt)
        return None

    required_keys = ["title", "turns"]
    if any(key not in data for key in required_keys):
        logger.warning("Schema validation failed (attempt %d): missing title/turns", attempt)
        return None

    if not isinstance(data["turns"], list) or len(data["turns"]) == 0:
        logger.warning("Empty turns (attempt %d)", attempt)
        return None

    valid_speakers = {"Jack", "Amy"}
    for i, turn in enumerate(data["turns"]):
        if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
            logger.warning("Turn %d invalid (attempt %d)", i, attempt)
            return None
        if turn["speaker"] not in valid_speakers:
            logger.warning("Unexpected speaker %r in turn %d", turn["speaker"], i)
            return None
        if not isinstance(turn["text"], str) or not turn["text"].strip():
            logger.warning("Empty text in turn %d", i)
            return None

    total_words = sum(len(t["text"].split()) for t in data["turns"])
    logger.info(
        'Script generated: "%s" — %d turns, ~%d words',
        data["title"],
        len(data["turns"]),
        total_words,
    )
    return data


def _parse_packaging(raw_text: str, attempt: int) -> dict[str, Any] | None:
    """Parse and lightly validate packaging JSON."""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("Packaging JSON parse failed (attempt %d)", attempt)
        return None

    required = [
        "recommended_title",
        "thumbnail_text",
        "alternate_titles",
        "alternate_thumbnail_texts",
        "video_angle",
        "first_5_seconds_hook",
        "description",
        "keywords",
        "pinned_comment",
    ]
    if any(key not in data for key in required):
        logger.warning("Packaging schema validation failed (attempt %d)", attempt)
        return None

    # Enforce mobile-friendly thumbnail text. If the model gives too many words,
    # keep the first 4 words instead of failing the whole generation.
    thumbnail_words = str(data["thumbnail_text"]).split()
    if len(thumbnail_words) > 4:
        data["thumbnail_text"] = " ".join(thumbnail_words[:4]).upper()

    data["thumbnail_text"] = str(data["thumbnail_text"]).strip().upper()
    return data


def generate_script(
    topic: str,
    level: str = "A2",
    words: int = 2000,
    llm_type: int = 1,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate a real-life English conversation script."""
    user_prompt = (
        f'Topic: "{topic}"\n'
        f"Target CEFR level: {level}\n"
        f"Approximate word count: {words} words\n\n"
        "INSTRUCTIONS FOR HIGH RETENTION:\n"
        "1. THE HOOK: Do NOT start with 'Hello' or 'Is this seat free'. Start with a specific, intriguing line "
        "that mentions the most interesting part of the story (e.g., a trip to Japan, a mystery novel plot, or a career change). \n"
        "2. PATTERN INTERRUPT: Break the 'textbook' flow. Use natural interruptions, 'oh' sounds, and specific details "
        "like 'Caramel Macchiato' or 'La Bella Vita' instead of just 'coffee' or 'restaurant' .\n"
        "3. EMOTIONAL CONNECTION: Include personal stories early on, like moving from a big city or missing family, "
        "to build a bond with the viewer.\n"
        "4. STRUCTURE: Use a 'In Media Res' start (starting in the middle of a thought) and then loop back to the introduction.\n\n"
        "Write a natural, real-life English conversation based on these rules. "
        "Output ONLY raw JSON with keys: 'hook', 'dialogue' (list of speaker/text), 'vocabulary_highlight'."
    )

    if llm_type == 2:
        return _generate_via_openai(user_prompt, max_retries)
    return _generate_via_claude(user_prompt, max_retries)


def _generate_via_claude(user_prompt: str, max_retries: int) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        logger.info("Claude API call attempt %d/%d …", attempt, max_retries)
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = message.content[0].text.strip()
        logger.debug("Raw response length: %d chars", len(raw_text))
        data = _parse_and_validate(raw_text, attempt)
        if data:
            return data

    raise ValueError(f"Failed to get valid JSON from Claude after {max_retries} attempts.")


def _generate_via_openai(user_prompt: str, max_retries: int) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        logger.info("OpenAI API call attempt %d/%d …", attempt, max_retries)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=8192,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug("Raw response length: %d chars", len(raw_text))
        data = _parse_and_validate(raw_text, attempt)
        if data:
            return data

    raise ValueError(f"Failed to get valid JSON from OpenAI after {max_retries} attempts.")


def _call_claude_text(system: str, user_prompt: str, max_tokens: int = 512) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def _call_openai_text(system: str, user_prompt: str, max_tokens: int = 512) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _llm_text(system: str, user_prompt: str, llm_type: int, max_tokens: int = 512) -> str:
    if llm_type == 2:
        return _call_openai_text(system, user_prompt, max_tokens=max_tokens)
    return _call_claude_text(system, user_prompt, max_tokens=max_tokens)


def generate_youtube_packaging(
    script: dict[str, Any],
    llm_type: int = 1,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate CTR-focused title, thumbnail text, hook, description, tags, and pinned comment."""
    full_text = " ".join(t["text"] for t in script.get("turns", []))
    user_prompt = (
        f'Original script title: "{script.get("title", "")}"\n'
        f'Topic: "{script.get("topic", "")}"\n'
        f'Level: "{script.get("level", "A2/B1")}"\n'
        f'Hook: "{script.get("hook", "")}"\n\n'
        f"Transcript excerpt:\n{full_text[:4000]}\n\n"
        "Create YouTube packaging for this video. Return ONLY raw JSON."
    )

    for attempt in range(1, max_retries + 1):
        raw_text = _llm_text(PACKAGING_SYSTEM_PROMPT, user_prompt, llm_type, max_tokens=1200)
        data = _parse_packaging(raw_text, attempt)
        if data:
            return data

    raise ValueError(f"Failed to get valid packaging JSON after {max_retries} attempts.")


def generate_full_episode_package(
    topic: str,
    level: str = "A2",
    words: int = 2000,
    llm_type: int = 1,
) -> dict[str, Any]:
    """Generate script + YouTube packaging in one call."""
    script = generate_script(topic=topic, level=level, words=words, llm_type=llm_type)
    packaging = generate_youtube_packaging(script=script, llm_type=llm_type)
    return {
        "script": script,
        "youtube_packaging": packaging,
    }


def generate_youtube_description(script: dict[str, Any], llm_type: int = 1) -> str:
    """Generate a 1-2 sentence YouTube description from the full script via LLM."""
    return generate_youtube_packaging(script, llm_type)["description"]


def generate_youtube_keywords(script: dict[str, Any], llm_type: int = 1) -> str:
    """Generate SEO-friendly comma-separated YouTube keywords via LLM."""
    keywords = generate_youtube_packaging(script, llm_type)["keywords"]
    if isinstance(keywords, list):
        return ", ".join(str(k) for k in keywords)
    return str(keywords)


def generate_thumbnail_headline(script: dict[str, Any], llm_type: int = 1) -> str:
    """Generate a clickable 2-4 word thumbnail headline via LLM."""
    return generate_youtube_packaging(script, llm_type)["thumbnail_text"]


def generate_title_ideas(topic: str, level: str = "A2", llm_type: int = 1) -> dict[str, Any]:
    """Generate title/thumbnail ideas before writing the full script."""
    mini_script = {
        "title": topic,
        "topic": topic,
        "level": level,
        "hook": "",
        "turns": [
            {
                "speaker": "Jack",
                "text": f"A real-life English conversation about {topic} for {level} learners.",
            }
        ],
    }
    return generate_youtube_packaging(mini_script, llm_type=llm_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    package = generate_full_episode_package(
        topic="Coffee or Tea",
        level="A2",
        words=1200,
        llm_type=1,  # 1 = Claude, 2 = OpenAI
    )

    print(json.dumps(package, indent=2, ensure_ascii=False))
