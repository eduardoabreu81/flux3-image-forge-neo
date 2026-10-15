# FLUX 3 Image Klein support for Forge Neo
# The bounding-box layout FLUX 3 reads from the prompt: a JSON list of elements appended to the caption,
# [{"id": ..., "bbox": [top, left, bottom, right], "desc": ...}, ...] (Black Forest Labs' bounding box guide).
#
# sd-dynamic-prompts reads {x} as a one-option variant and strips the braces, which leaves
# ["id": ..., "bbox": [...], "id": ...]. Every element starts with its "id", so the list is rebuilt by wrapping
# each run that starts at an "id" key in braces; the result is kept only when it parses as JSON.

import json
import re

_LIST_START = re.compile(r'\[\s*"id"\s*:')
_ELEMENT_START = re.compile(r'\s*"id"\s*:')


def _closing_bracket(text: str, start: int) -> int:
    """Index of the ] that closes the [ at start, skipping brackets inside JSON strings; -1 if none."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _top_level_commas(body: str) -> list[int]:
    commas, depth, in_string, escaped = [], 0, False, False
    for i, c in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "," and depth == 0:
            commas.append(i)
    return commas


def restore_layout_json(text: str) -> str:
    """The prompt with a brace-stripped layout list rebuilt; unchanged when there is none or it does not parse."""
    match = _LIST_START.search(text)
    if match is None:
        return text
    start = match.start()
    end = _closing_bracket(text, start)
    if end < 0:
        return text
    body = text[start + 1:end]
    # split before every top-level "id" key: each one opens an element
    cuts = [0] + [i + 1 for i in _top_level_commas(body) if _ELEMENT_START.match(body, i + 1)] + [len(body) + 1]
    elements = [body[a:b - 1].strip() for a, b in zip(cuts, cuts[1:])]
    rebuilt = "[" + ", ".join("{" + e + "}" for e in elements) + "]"
    try:
        parsed = json.loads(rebuilt)
    except ValueError:
        return text
    if not all(isinstance(e, dict) for e in parsed):
        return text
    return text[:start] + rebuilt + text[end + 1:]
