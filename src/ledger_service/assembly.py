"""Stage 8 — Ledger Assembly.

Synthesises live.md and lecture_history.md from the LedgerRegistry.
Both files are written atomically (tmp → rename).

live.md             — active entities only, spatial order (top-to-bottom)
lecture_history.md  — all entities chronologically; erased ones in Archives;
                      corrections shown as inline diffs
"""

from __future__ import annotations

from pathlib import Path

from ledger_service.registry import LedgerEntry, LedgerRegistry


def synthesize(registry: LedgerRegistry, output_dir: Path) -> None:
    """Write live.md and lecture_history.md to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_dir / "live.md", _render_live(registry))
    _write_atomic(output_dir / "lecture_history.md", _render_history(registry))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_live(registry: LedgerRegistry) -> str:
    active = registry.get_active()
    if not active:
        return "# Whiteboard — Live\n\n*(board empty)*\n"

    lines = ["# Whiteboard — Live\n"]
    for entry in active:
        ts = registry.mono_to_wall_str(entry.first_seen)
        text = entry.versions[-1].text
        lines.append(f"## [{ts}] Region {entry.region_id}\n")
        lines.append(f"{text}\n")
    return "\n".join(lines)


def _render_history(registry: LedgerRegistry) -> str:
    all_entries = registry.get_all()
    if not all_entries:
        return "# Whiteboard — Lecture History\n\n*(no content yet)*\n"

    active = [e for e in all_entries if e.erased_at is None]
    erased = [e for e in all_entries if e.erased_at is not None]

    sections: list[str] = ["# Whiteboard — Lecture History\n"]

    for entry in active:
        sections.append(_render_entry(entry, registry))

    if erased:
        sections.append("---\n")
        sections.append("## Archives\n")
        for entry in erased:
            sections.append(_render_entry(entry, registry, archived=True))

    return "\n".join(sections)


def _render_entry(
    entry: LedgerEntry,
    registry: LedgerRegistry,
    archived: bool = False,
) -> str:
    ts_start = registry.mono_to_wall_str(entry.first_seen)
    heading_prefix = "###" if archived else "##"

    if archived and entry.erased_at is not None:
        ts_end = registry.mono_to_wall_str(entry.erased_at)
        header = f"{heading_prefix} [{ts_start}–{ts_end}] Region {entry.region_id} *(erased)*"
    else:
        header = f"{heading_prefix} [{ts_start}] Region {entry.region_id}"

    latest_text = entry.versions[-1].text
    parts = [header, "", latest_text]

    if len(entry.versions) > 1:
        parts.append("")
        parts.append("#### Corrections")
        # Show all versions except the latest as corrections
        for v in entry.versions[:-1]:
            ts = registry.mono_to_wall_str(v.timestamp)
            escaped = v.text.replace('"', '\\"')
            parts.append(f'→ [{ts}] Original: "{escaped}"')

    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)
