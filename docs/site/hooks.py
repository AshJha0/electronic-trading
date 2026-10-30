"""MkDocs hook: publish the repository's Markdown files as site pages.

The guide, cookbook, contract, architecture notes and changelog stay where
they are (and keep rendering on GitHub); this hook adds them to the build
under short URLs, so there is one source for each document.
"""

from __future__ import annotations

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[2]

# site path -> source file, relative to the repository root
PAGES = {
    "learn.md": "LEARN.md",
    "cookbook.md": "COOKBOOK.md",
    "api-spec.md": "API_SPEC.md",
    "architecture.md": "docs/ARCHITECTURE.md",
    "changelog.md": "CHANGELOG.md",
    "publishing.md": "docs/GITHUB_PAGES.md",
}


def on_files(files: Files, config: MkDocsConfig) -> Files:
    for uri, src in PAGES.items():
        files.append(File.generated(config, uri, abs_src_path=str(ROOT / src)))
    return files
