import json
from typing import Dict, Any


def _build_sample_text(files: Dict[str, Any]) -> str:
    ''' Single shared text-building routine, used by both functions below, so
    the char-length fed into the fallback token estimate is identical
    regardless of which of the two call sites is running. '''
    return "\n\n".join(
        f"### {path} ###\n{content if isinstance(content, str) else json.dumps(content)}"
        for path, content in files.items()
    )