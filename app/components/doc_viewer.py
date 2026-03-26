"""Markdown documentation viewer with sidebar table of contents."""

import re
from pathlib import Path

import streamlit as st


def _parse_headings(markdown_text: str) -> list[dict]:
    """Extract ## and ### headings from markdown for the TOC.

    Returns a list of dicts with keys: level (2 or 3), text, anchor.
    """
    headings = []
    for match in re.finditer(r"^(#{2,3})\s+(.+)$", markdown_text, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        # Build a URL-friendly anchor from the heading text
        anchor = re.sub(r"[^\w\s-]", "", text.lower())
        anchor = re.sub(r"[\s]+", "-", anchor)
        headings.append({"level": level, "text": text, "anchor": anchor})
    return headings


def _inject_anchors(markdown_text: str) -> str:
    """Inject HTML anchor tags before each ## and ### heading.

    Streamlit's st.markdown doesn't auto-generate IDs for headings,
    so we insert <a> tags manually to enable TOC linking.
    """

    def _replace_heading(match: re.Match) -> str:
        hashes = match.group(1)
        text = match.group(2).strip()
        anchor = re.sub(r"[^\w\s-]", "", text.lower())
        anchor = re.sub(r"[\s]+", "-", anchor)
        return f'<a id="{anchor}"></a>\n\n{hashes} {text}'

    return re.sub(r"^(#{2,3})\s+(.+)$", _replace_heading, markdown_text, flags=re.MULTILINE)


def render(doc_path: str, title: str = "Documentation") -> None:
    """Render a markdown doc inside an expander with a sidebar TOC.

    Args:
        doc_path: Path to the markdown file (relative to repo root or absolute).
        title: Label shown on the expander.
    """
    path = Path(doc_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    if not path.exists():
        st.warning(f"Documentation not found: {doc_path}")
        return

    markdown_text = path.read_text(encoding="utf-8")
    headings = _parse_headings(markdown_text)
    enriched_markdown = _inject_anchors(markdown_text)

    with st.expander(f"📚 {title}", expanded=False):
        toc_col, doc_col = st.columns([1, 3], gap="medium")

        # -- Left column: Table of Contents --
        with toc_col:
            st.markdown("#### Contents")
            for h in headings:
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;" if h["level"] == 3 else ""
                st.markdown(
                    f'{indent}<a href="#{h["anchor"]}" target="_self">{h["text"]}</a>',
                    unsafe_allow_html=True,
                )

        # -- Right column: Full document --
        with doc_col:
            st.markdown(enriched_markdown, unsafe_allow_html=True)
