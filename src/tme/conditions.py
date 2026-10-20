"""What the model sees in each condition, built as an Ollama chat request.

- A, no memory: the instructions, the components to choose from, and the new ticket.
- B, similar past tickets: A plus the 5 earlier tickets the memory search returns (the same
  5 the vote-of-5 baseline uses), each with its key, summary, the start of its description,
  component and type. With `duplicate_answer = "pick"` (duplicates v1, v2) the model names one
  of them as the original or null; with "judge" (v3) it labels each one same, related or
  different, and the first labelled same is taken as the original.

The components offered are those used by any ticket created before the new one, in
alphabetical order. That covers every right answer in dev and test and uses nothing from
later tickets. (The list rarely changes between tickets, but with qwen3.5:4b Ollama re-read
the whole prompt on every call in the timing check, so this saves no time.)
"""

from __future__ import annotations

from tme.data import ISSUE_TYPES, mask

CONDITIONS = ("A", "B")
MEMORY_DESCRIPTION_CHARS = 300
JUDGEMENTS = ("same", "related", "different")
# A judged answer is about 100 tokens against about 37 for a single pick, so it gets more room.
# Only those requests change; A's and earlier B requests keep their options and recordings.
JUDGED_NUM_PREDICT = 256


def components_before(tickets: list[dict], memory_size: int) -> list[str]:
    return sorted({c for t in tickets[:memory_size] for c in t["components"]})


def _clip(text: str | None, limit: int, full_length: int) -> str:
    """Cut, then mask, so a key cut in half is still masked; mark text that was cut."""
    text = (text or "")[:limit]
    clipped = mask(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clipped:
        return "(none)"
    return clipped + (" …" if full_length > limit else "")


def render_new_ticket(ticket: dict) -> str:
    description = _clip(ticket["description"], len(ticket["description"] or ""), ticket["description_chars"])
    return f"New ticket\nSummary: {mask(ticket['summary'])}\nDescription:\n{description}"


def render_memory_ticket(ticket: dict) -> str:
    component = ", ".join(ticket["components"]) or "none"
    description = _clip(ticket["description"], MEMORY_DESCRIPTION_CHARS, ticket["description_chars"])
    return (f"{ticket['key']} (component: {component}; type: {ticket['issuetype']})\n"
            f"Summary: {mask(ticket['summary'])}\nDescription: {description}")


def build_request(condition: str, ticket: dict, memory: list[dict], components: list[str],
                  config: dict, prompts: dict[str, str]) -> tuple[dict, dict]:
    """Return the Ollama request and the JSON schema its answer must follow."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    properties = {
        "component": {"type": "string", "enum": components},
        "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
    }
    system = [prompts["route"].strip()]
    user = []
    options = config["options"]
    if condition == "B":
        system.append(prompts["duplicates"].strip())
        if config.get("duplicate_answer", "pick") == "judge":
            # One label per shown ticket, in the order shown; the first "same" is the answer.
            properties["earlier_tickets"] = {
                "type": "object",
                "properties": {t["key"]: {"type": "string", "enum": list(JUDGEMENTS)} for t in memory},
                "required": [t["key"] for t in memory],
            }
            options = {**options, "num_predict": JUDGED_NUM_PREDICT}
        else:
            properties["duplicate_of"] = {"enum": [t["key"] for t in memory] + [None]}
        user.append("Earlier tickets most similar to the new one, most similar first:\n\n"
                    + "\n\n".join(render_memory_ticket(t) for t in memory))
    system.append("Components:\n" + "\n".join(components))
    system.append(f"Answer with JSON only, with the keys {', '.join(properties)}.")
    user.append(render_new_ticket(ticket))

    schema = {"type": "object", "properties": properties, "required": list(properties)}
    request = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "\n\n".join(system)},
            {"role": "user", "content": "\n\n".join(user)},
        ],
        "format": schema,
        "options": options,
    }
    if config.get("think") is not None:  # only for models that can think; part of the cassette key
        request["think"] = config["think"]
    return request, schema
