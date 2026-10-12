import json
import os
import re
import unicodedata

from fastapi import HTTPException
from groq import Groq, GroqError

from app.schemas import SummaryRequest, SummaryResponse
from app.storage.prompt_store import PROMPT_KEY, get_prompt


def make_story_key(story_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", story_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    key = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    if not key:
        raise HTTPException(
            status_code=422, detail="story_name must contain at least one letter or number"
        )
    return key


SUMMARY_OUTPUT_CONTRACT = """Return exactly one valid JSON object with exactly two string fields:
{"key":"story-name","content":"formatted summary"}
Set key to the supplied story name in lowercase with words separated by hyphens.
Put the complete Markdown summary in content. Do not wrap the JSON in a code fence."""


def generate_summary(request: SummaryRequest) -> SummaryResponse:
    story_key = make_story_key(request.story_name)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    try:
        system_prompt = f"{get_prompt(PROMPT_KEY)}\n\n{SUMMARY_OUTPUT_CONTRACT}"
    except (LookupError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Story name: {request.story_name}\n\nStory:\n{request.story}",
                },
            ],
            response_format={"type": "json_object"},
        )
    except GroqError as error:
        raise HTTPException(status_code=502, detail="Groq summarization request failed") from error

    if not completion.choices:
        raise HTTPException(status_code=502, detail="Groq returned no summary choices")

    response_content = completion.choices[0].message.content
    if not response_content:
        raise HTTPException(status_code=502, detail="Groq returned an empty summary")

    try:
        result = json.loads(response_content)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail="Groq returned invalid JSON") from error

    if (
        not isinstance(result, dict)
        or set(result) != {"key", "content"}
        or not isinstance(result["content"], str)
        or not result["content"].strip()
    ):
        raise HTTPException(status_code=502, detail="Groq returned an invalid summary format")

    return SummaryResponse(key=story_key, content=result["content"])
