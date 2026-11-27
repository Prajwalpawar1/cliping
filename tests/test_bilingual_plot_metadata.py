"""Tests for bilingual English and Hindi plot-based metadata generation."""
import pytest

from app import selector
from app.selector import Sentence


def _dummy_sentences(n=10, dur=5.0):
    return [Sentence(i * dur, (i + 1) * dur - 0.1, f"Plot sentence number {i} with secret twist") for i in range(n)]


def test_ai_prompt_contains_english_hindi_plot_rules():
    sents = _dummy_sentences(5)
    prompt = selector._prompt(sents, "Mystery of the Island", 3, 30, 60)
    # Verify English and Hindi plot rules are present
    assert "English and Hindi" in prompt
    assert "plot" in prompt.lower()
    assert "title, hook, description and hashtags" in prompt


def test_ai_system_prompt_enforces_plot_and_bilingual():
    assert "English and Hindi" in selector._SYSTEM
    assert "plot" in selector._SYSTEM.lower()
    assert "Do not use any other language" in selector._SYSTEM


def test_select_heuristic_generates_bilingual_plot_metadata(words_factory):
    text = " ".join(f"Why is the secret truth number {i} so shocking? | Because it changes the entire plot." for i in range(70))
    sents = selector.to_sentences(words_factory(text))
    clips = selector.select_heuristic(sents, 2, 30, 60)
    assert len(clips) >= 1
    for c in clips:
        # Check bilingual title
        assert "भाग" in c.title or "Part" in c.title
        # Check bilingual description
        assert "Plot Highlight" in c.description
        assert "कहानी" in c.description
        # Check hashtags in English and Hindi context
        assert any(h.lower() in ["hindi", "kahani", "story", "shorts", "reels"] for h in c.hashtags)
