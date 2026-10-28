"""Framework picker component.

Renders a grid of AutoML framework cards with:
  - Installed / Not-installed / Unsupported badges
  - One-line search strategy
  - Checkbox for selection
  - Default budget annotation

Returns: list of selected framework IDs.
"""
from __future__ import annotations

import streamlit as st

_BADGE_CSS = {
    "installed": ("✅", "#e8f5e9", "#2e7d32"),
    "not_installed": ("⚠️", "#fff8e1", "#f57f17"),
    "unsupported": ("🚫", "#fce4ec", "#b71c1c"),
}


def _badge_key(fw: dict) -> str:
    if not fw.get("os_supported", True):
        return "unsupported"
    if fw.get("installed", False):
        return "installed"
    return "not_installed"


def framework_picker(fw_list: list[dict]) -> list[str]:
    """Render a grid of framework cards and return selected framework ids."""
    selected: list[str] = []

    if not fw_list:
        st.info("No framework metadata available.")
        return selected

    cols_per_row = 2
    rows = [fw_list[i : i + cols_per_row] for i in range(0, len(fw_list), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, fw in zip(cols, row, strict=False):
            bk = _badge_key(fw)
            icon, bg, fg = _BADGE_CSS[bk]
            label_txt = {
                "installed": f"Installed {fw.get('version') or ''}",
                "not_installed": "Not installed",
                "unsupported": "Unsupported on this OS",
            }[bk]

            with col:
                st.markdown(
                    f"""<div style="background:{bg};border-left:4px solid {fg};
                        padding:10px 14px;border-radius:8px;margin-bottom:8px">
                        <span style="font-size:1.1em;font-weight:700">{icon} {fw.get('name', fw['id'])}</span>
                        <span style="float:right;font-size:0.75em;color:{fg}">{label_txt}</span>
                        <div style="font-size:0.8em;color:#555;margin-top:4px">{fw.get('strategy','')}</div>
                        {f'<div style="font-size:0.75em;color:#888;margin-top:3px">⚠ {fw["reason"]}</div>' if fw.get('reason') else ''}
                        </div>""",
                    unsafe_allow_html=True,
                )
                disabled = not fw.get("installed", False)
                checked = st.checkbox(
                    f"Include {fw['id']}",
                    value=False,
                    disabled=disabled,
                    key=f"fw_pick_{fw['id']}",
                )
                if checked:
                    selected.append(fw["id"])

    return selected
