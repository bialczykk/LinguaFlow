"""Integration tests for the grammar analysis chain.

These tests call the real Anthropic API and verify that the chain
returns properly structured GrammarFeedback. They require a valid
ANTHROPIC_API_KEY in the root .env file.
"""

import os
import pytest

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


def test_analyze_grammar_returns_grammar_feedback():
    """The analysis chain returns a GrammarFeedback with at least one issue
    for text that contains obvious grammar errors."""
    from chains import analyze_grammar
    from models import GrammarFeedback

    text = "He go to school every day. She have many book on the table."

    result = analyze_grammar(text)

    assert isinstance(result, GrammarFeedback)
    assert len(result.issues) > 0
    assert result.proficiency.cefr_level in ("A1", "A2", "B1", "B2", "C1", "C2")
    assert len(result.corrected_full_text) > 0


def test_analyze_grammar_handles_correct_text():
    """The analysis chain returns a valid GrammarFeedback even for
    well-written text (may have zero issues)."""
    from chains import analyze_grammar
    from models import GrammarFeedback

    text = "The quick brown fox jumps over the lazy dog."

    result = analyze_grammar(text)

    assert isinstance(result, GrammarFeedback)
    assert result.proficiency.cefr_level in ("A1", "A2", "B1", "B2", "C1", "C2")
