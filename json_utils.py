# json_utils.py
"""
Utility functions for extracting and parsing JSON objects from a streaming buffer.
"""

MAX_TAIL = 2_000_000  # safety limit for rolling buffer

def extract_json_from_buffer(buffer: str):
    """
    Extract complete JSON objects from a streaming buffer.
    Returns a tuple: (objects, tail)
      - objects: list of JSON strings found in the buffer
      - tail: any remaining partial (unclosed) JSON fragment
    The parser is string-escape aware, so braces inside strings won't confuse it.
    """
    objs = []
    n = len(buffer)
    i = 0
    in_str = False
    esc = False
    depth = 0
    start = -1

    while i < n:
        ch = buffer[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        objs.append(buffer[start:i + 1])
                        start = -1
        i += 1

    # Keep any incomplete JSON fragment (tail)
    if depth > 0 and start != -1:
        tail = buffer[start:]
    else:
        tail = buffer[-1000:]  # small suffix to capture potential starts

    if len(tail) > MAX_TAIL:
        tail = tail[-MAX_TAIL:]

    return objs, tail
