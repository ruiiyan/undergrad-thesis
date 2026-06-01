"""
utils/bloom_prompt.py
---------------------
Reusable Bloom's taxonomy annotation prompt utilities.
Used by both offline (phase3_bloom_annotation.py) and online phase.
"""

import os
import json
from pathlib import Path
from jinja2 import Template

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

def load_system_prompt() -> str:
    """Load system prompt from markdown file."""
    path = PROMPT_DIR / "system_bloom_annotation.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

USER_TEMPLATE = Template("""Please assign Bloom's Taxonomy levels to each section of this reflection:

Topic: {{ topic }}
Cluster keywords: {{ keywords }}

RESULT:
{{ result }}

PERSONAL FEELING/REFLECTION:
{{ pfr }}

LEARNING:
{{ learning }}""")

def build_user_message(topic: str,
                       keywords: str,
                       result: str, 
                       pfr: str,
                       learning: str) -> str:
    """Render user message from Jinja2 template."""
    return USER_TEMPLATE.render(
        topic=topic.strip(),
        keywords=keywords.strip(),
        result=result.strip(),
        pfr=pfr.strip(),
        learning=learning.strip()
    )

def annotate_bloom(client, topic, keywords, result, pfr, learning, model="claude-sonnet-4-6"):
    user_message = build_user_message(topic, keywords, result, pfr, learning)
    system_prompt = load_system_prompt()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result_json = json.loads(raw)
        return {
            "success":  True,
            "result":   result_json.get("result"),
            "pfr":      result_json.get("pfr"),
            "learning": result_json.get("learning")
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}