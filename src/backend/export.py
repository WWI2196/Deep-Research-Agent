"""Report export — markdown file, PDF via browser print."""

import os
from pathlib import Path
from typing import Optional


def export_markdown(content: str, output_path: Optional[str] = None) -> str:
    """Save report as markdown file. Returns the file path."""
    if output_path:
        path = Path(output_path)
    else:
        output_dir = Path(os.getcwd()) / "research-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        import time
        path = output_dir / f"report-{int(time.time())}.md"

    path.write_text(content, encoding="utf-8")
    return str(path.resolve())
