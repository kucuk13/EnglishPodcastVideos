import json
import logging
import os
import re

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional podcast scriptwriter for an English-learning podcast.
Your task is to write a natural, engaging two-person dialogue between two hosts:

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
    {"speaker": "Jack", "text": "..."},
    {"speaker": "Amy", "text": "..."}
  ]
}
"""


def _parse_and_validate(raw_text: str, attempt: int) -> dict | None:
    """Parse JSON from raw LLM response and validate schema. Returns dict or None."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (attempt %d): %s", attempt, exc)
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            return None
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

    if "title" not in data or "turns" not in data:
        logger.warning("Schema validation failed (attempt %d)", attempt)
        return None
    if not isinstance(data["turns"], list) or len(data["turns"]) == 0:
        logger.warning("Empty turns (attempt %d)", attempt)
        return None
    for i, turn in enumerate(data["turns"]):
        if "speaker" not in turn or "text" not in turn:
            logger.warning("Turn %d invalid (attempt %d)", i, attempt)
            return None

    total_words = sum(len(t["text"].split()) for t in data["turns"])
    logger.info(
        "Script generated: \"%s\" — %d turns, ~%d words",
        data["title"], len(data["turns"]), total_words,
    )
    return data


def generate_script(
    topic: str,
    level: str = "A2",
    words: int = 2000,
    llm_type: int = 1,
    max_retries: int = 2,
) -> dict:
    user_prompt = (
        f"Write a podcast dialogue about the topic: \"{topic}\".\n"
        f"Target CEFR level: {level}.\n"
        f"Approximate word count: {words} words.\n\n"
        f"Remember: output ONLY raw JSON, no markdown, no extra text."
    )

    if llm_type == 2:
        return _generate_via_openai(user_prompt, max_retries)
    return _generate_via_claude(user_prompt, max_retries)


def _generate_via_claude(user_prompt: str, max_retries: int) -> dict:
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


def _generate_via_openai(user_prompt: str, max_retries: int) -> dict:
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
        raw_text = response.choices[0].message.content.strip()
        logger.debug("Raw response length: %d chars", len(raw_text))
        data = _parse_and_validate(raw_text, attempt)
        if data:
            return data

    raise ValueError(f"Failed to get valid JSON from OpenAI after {max_retries} attempts.")


def _call_claude_text(system: str, user_prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def _call_openai_text(system: str, user_prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    from openai import OpenAI as _OpenAI
    client = _OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _llm_text(system: str, user_prompt: str, llm_type: int) -> str:
    if llm_type == 2:
        return _call_openai_text(system, user_prompt)
    return _call_claude_text(system, user_prompt)


def generate_youtube_description(script: dict, llm_type: int = 1) -> str:
    """Generate a 1-2 sentence YouTube description from the full script via LLM."""
    full_text = " ".join(t["text"] for t in script["turns"])
    system = "You are a YouTube SEO expert. Write concise, engaging video descriptions."
    user_prompt = (
        f"Podcast title: \"{script.get('title', '')}\"\n\n"
        f"Transcript:\n{full_text[:4000]}\n\n"
        "Write a 1-2 sentence YouTube video description that is engaging and informative. "
        "Return ONLY the description text, no labels or extra text."
    )
    return _llm_text(system, user_prompt, llm_type)


def generate_youtube_keywords(script: dict, llm_type: int = 1) -> str:
    """Generate SEO-friendly comma-separated YouTube keywords via LLM."""
    full_text = " ".join(t["text"] for t in script["turns"])
    system = "You are a YouTube SEO expert. Generate relevant, searchable keywords."
    user_prompt = (
        f"Podcast title: \"{script.get('title', '')}\"\n"
        f"Topic: \"{script.get('topic', '')}\"\n\n"
        f"Transcript:\n{full_text[:4000]}\n\n"
        "Generate 10-15 SEO-friendly YouTube keywords/tags for this English learning podcast. "
        "Return ONLY a comma-separated list, no labels or extra text."
    )
    return _llm_text(system, user_prompt, llm_type)


def generate_thumbnail_headline(script: dict, llm_type: int = 1) -> str:
    """Generate a catchy max-5-word thumbnail headline via LLM."""
    system = "You are a YouTube thumbnail copywriter. Write punchy, click-worthy headlines."
    user_prompt = (
        f"Podcast title: \"{script.get('title', '')}\"\n"
        f"Topic: \"{script.get('topic', '')}\"\n\n"
        "Write a catchy YouTube thumbnail headline in MAXIMUM 5 WORDS. "
        "Make it intriguing and click-worthy for English learners. "
        "Return ONLY the headline text, no quotes or extra text."
    )
    return _llm_text(system, user_prompt, llm_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    script = generate_script("Meeting People", "A2", 2000)
    print(json.dumps(script, indent=2, ensure_ascii=False))
