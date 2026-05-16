"""Tests for robit.agent.widgets.enforcement — the chip overlay."""

from __future__ import annotations

import pytest

from robit.agent.loop import VetoFired
from robit.agent.widgets.enforcement import (
    ConductChip,
    EnforcementChip,
    RedactionChip,
    VetoChip,
)


# ---------------------------------------------------------------------------
# Constructor + kind validation
# ---------------------------------------------------------------------------


def test_veto_chip_renders_with_prefix_and_red_background():
    chip = EnforcementChip(kind="veto", label="x", detail="y")
    rendered = chip.render()
    plain = rendered.plain
    # The prefix glyph must be present, in front of the label.
    assert plain.startswith("✘ x")  # ✘ x
    # The rich.text.Text must carry a red background somewhere in its spans.
    styled = "".join(str(span.style) for span in rendered.spans)
    assert "on red" in styled


def test_unknown_kind_raises_with_valid_list_in_message():
    with pytest.raises(ValueError) as exc:
        EnforcementChip(kind="bogus", label="x")
    msg = str(exc.value)
    assert "valid kinds are" in msg
    # All five kinds appear in the error message — keeps it actionable.
    for kind in ("veto", "redaction", "conduct", "audit", "cost"):
        assert kind in msg


# ---------------------------------------------------------------------------
# VetoChip.from_event
# ---------------------------------------------------------------------------


def test_veto_chip_from_event_without_pattern_id_omits_bracket():
    event = VetoFired(
        plugin="destructive-op-gate",
        reason="rm -rf",
        phase="trust-gate",
        pattern_id=None,
    )
    chip = VetoChip.from_event(event)
    assert chip.label == "destructive-op-gate blocked"
    # No pattern_id → no trailing "[<id>]" inside the detail.
    assert chip.detail == "rm -rf"
    assert "[" not in chip.detail


def test_veto_chip_from_event_with_pattern_id_appends_bracket():
    event = VetoFired(
        plugin="destructive-op-gate",
        reason="rm -rf",
        phase="trust-gate",
        pattern_id="w5-rm-rf",
    )
    chip = VetoChip.from_event(event)
    assert chip.detail == "rm -rf [w5-rm-rf]"


# ---------------------------------------------------------------------------
# RedactionChip.from_patterns
# ---------------------------------------------------------------------------


def test_redaction_chip_singular_count_label():
    chip = RedactionChip.from_patterns(["aws-access-key-id"])
    assert chip.label == "secret-mask redacted 1 pattern(s)"
    assert chip.detail == "aws-access-key-id"


def test_redaction_chip_empty_list_still_renders():
    chip = RedactionChip.from_patterns([])
    assert chip.label == "secret-mask redacted 0 pattern(s)"
    assert chip.detail == ""
    # Render must not raise on an empty detail.
    rendered = chip.render()
    assert "0 pattern(s)" in rendered.plain


# ---------------------------------------------------------------------------
# ConductChip.from_modules
# ---------------------------------------------------------------------------


def test_conduct_chip_module_count_label():
    chip = ConductChip.from_modules(["discipline", "verification"])
    assert chip.label == "conduct: 2 module(s) injected"
    assert chip.detail == "discipline, verification"


# ---------------------------------------------------------------------------
# 80-char width contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chip",
    [
        EnforcementChip(kind="veto", label="x" * 30, detail="y" * 200),
        VetoChip.from_event(
            VetoFired(
                plugin="some-plugin-with-a-very-long-name",
                reason="reason " * 30,
                phase="trust-gate",
                pattern_id="pattern-id-also-long",
            )
        ),
        RedactionChip.from_patterns([f"pattern-{i}" for i in range(50)]),
        ConductChip.from_modules([f"module-{i}" for i in range(50)]),
        EnforcementChip(kind="audit", label="fast-path bypass", detail="some-tool"),
        EnforcementChip(kind="cost", label="$0.0023 this call", detail=None),
    ],
)
def test_rendered_chip_fits_within_80_chars(chip):
    rendered = chip.render()
    assert len(rendered.plain) <= 80, (
        f"chip kind={chip.kind} rendered {len(rendered.plain)} chars: "
        f"{rendered.plain!r}"
    )
