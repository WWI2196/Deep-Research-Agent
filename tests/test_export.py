"""Tests for report export."""

import tempfile
from pathlib import Path


def test_export_markdown_default_path():
    from src.backend.export import export_markdown
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            path = export_markdown("# Test Report\n\nContent")
            assert path.endswith(".md")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "# Test Report" in content
        finally:
            os.chdir(orig_cwd)


def test_export_markdown_custom_path():
    from src.backend.export import export_markdown
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir) / "custom-report.md"
        result = export_markdown("# Custom Report", str(custom_path))
        assert result == str(custom_path.resolve())
        assert custom_path.exists()
        assert "# Custom Report" in custom_path.read_text()


def test_export_markdown_creates_output_dir():
    from src.backend.export import export_markdown
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            path = export_markdown("Content")
            assert Path(path).parent.name == "research-output"
            assert Path(path).exists()
        finally:
            os.chdir(orig_cwd)
