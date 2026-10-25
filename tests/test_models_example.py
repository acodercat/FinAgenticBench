"""The documented setup must produce a runnable judge panel.

`models.toml` is gitignored, so `models.toml.example` is the only statement of
what a working configuration contains.  It shipped one section, `deepseek-flash`,
which is not in `DEFAULT_JUDGE_PANEL`, so
anyone following the documented setup could run `scripts.run` and then find that
two of the three scoring channels stop before making a call:
`registry.select()` raises `ValueError: unknown model 'qwen-max'` rather than
quietly scoring with a smaller panel.

The section NAME is the contract between this file and `core/judging.py`. These
tests pin that contract from both ends, so renaming a panel member without
updating the example fails here instead of at someone else's first scoring run.
"""

from __future__ import annotations


from config import PROJECT_ROOT
from core.judging import DEFAULT_JUDGE_PANEL, MIN_PANEL
from core.models import ModelRegistry

EXAMPLE = PROJECT_ROOT / "models.toml.example"


def _registry() -> ModelRegistry:
    return ModelRegistry.load(EXAMPLE)


def test_example_declares_every_default_judge():
    resolved = [config.name for config in _registry().select(DEFAULT_JUDGE_PANEL)]
    assert resolved == list(DEFAULT_JUDGE_PANEL)


def test_the_example_panel_can_meet_quorum():
    """A panel smaller than MIN_PANEL scores nothing, whatever it resolves to."""
    assert len(DEFAULT_JUDGE_PANEL) >= MIN_PANEL
    assert len(_registry().select(DEFAULT_JUDGE_PANEL)) >= MIN_PANEL


def test_the_example_panel_supports_same_lab_exclusion():
    """`--exclude-same-lab` needs labelled judges from at least two labs.

    It raises on an unlabelled judge and again when exclusion leaves fewer than
    two, so an example whose judges share a lab documents a panel that cannot
    run the robustness check the scoring design offers.
    """
    judges = _registry().select(DEFAULT_JUDGE_PANEL)
    labs = [judge.lab for judge in judges]
    assert all(labs), f"judges without a lab: {[j.name for j in judges if not j.lab]}"
    assert len(set(labs)) >= 2, f"every default judge is from {labs[0]}"
