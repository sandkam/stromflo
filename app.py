from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Stromflo", layout="wide")

CBPF_URL = "https://api.energy-charts.info/cbpf"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_DIR = REPO_ROOT / "json_2019"

META_KEYS = {
    "unix_seconds",
    "timestamps",
    "status",
    "license_info",
    "unit",
    "deprecated",
    "substitute",
}


def _to_unix_seconds(date_str: str, end_of_day: bool = False) -> int:
    """Convert a YYYY-MM-DD date string or integer string to UTC Unix seconds."""
    cleaned = date_str.strip()
    if cleaned.isdigit():
        return int(cleaned)
    dt = datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp())


def _extract_country_from_filename(path: Path) -> str:
    """Extract country code from filenames like cbpf_de_2019.json."""
    parts = path.stem.split("_")
    if len(parts) >= 2 and len(parts[1]) in {2, 3}:
        return parts[1].upper()
    return "DE"


def _series_to_daily_gwh(
    series_vals: list[Any],
    dt_index: pd.DatetimeIndex,
    day_keys: pd.Index,
    country: str,
    partner: str,
) -> list[dict[str, Any]]:
    """Convert a power series (MW) into daily import and export energy (GWh)."""
    if not isinstance(series_vals, list) or len(series_vals) != len(dt_index) or len(dt_index) == 0:
        return []

    s = pd.to_numeric(pd.Series(series_vals, index=dt_index), errors="coerce").fillna(0.0)
    deltas_h = dt_index.to_series().diff().dt.total_seconds().div(3600.0).dropna()
    dt_h = float(deltas_h.median()) if not deltas_h.empty and deltas_h.median() > 0 else 1.0

    # Positive values = import into country; negative values = export from country
    imp_gwh = s.clip(lower=0.0).mul(dt_h).div(1000.0).groupby(day_keys).sum()
    exp_gwh = s.clip(upper=0.0).abs().mul(dt_h).div(1000.0).groupby(day_keys).sum()

    rows: list[dict[str, Any]] = []
    for d in imp_gwh.index:
        rows.append(
            {
                "date": pd.Timestamp(d),
                "country": country,
                "partner": partner,
                "import_gwh": float(imp_gwh.loc[d]),
                "export_gwh": float(exp_gwh.loc[d]),
            }
        )
    return rows


def _collect_from_container(
    container: Any,
    dt_index: pd.DatetimeIndex,
    day_keys: pd.Index,
    default_country: str,
) -> list[dict[str, Any]]:
    """Extract daily rows from dict or list-of-dicts containers."""
    rows: list[dict[str, Any]] = []

    if isinstance(container, list):
        for item in container:
            if not isinstance(item, dict):
                continue
            label = str(item.get("name") or item.get("country") or item.get("code") or "").strip()
            vals = item.get("data") if isinstance(item.get("data"), list) else item.get("values")
            if label and isinstance(vals, list):
                rows.extend(
                    _series_to_daily_gwh(
                        series_vals=vals,
                        dt_index=dt_index,
                        day_keys=day_keys,
                        country=default_country,
                        partner=label,
                    )
                )
    elif isinstance(container, dict):
        for key, val in container.items():
            if key in META_KEYS:
                continue
            if isinstance(val, dict):
                target = str(key).upper()
                sub_series = val.get("data") if isinstance(val.get("data"), list) else None
                if sub_series is not None:
                    label = str(val.get("name") or target)
                    rows.extend(
                        _series_to_daily_gwh(
                            series_vals=sub_series,
                            dt_index=dt_index,
                            day_keys=day_keys,
                            country=default_country,
                            partner=label,
                        )
                    )
                else:
                    for neighbor, series_vals in val.items():
                        if isinstance(series_vals, list):
                            rows.extend(
                                _series_to_daily_gwh(
                                    series_vals=series_vals,
                                    dt_index=dt_index,
                                    day_keys=day_keys,
                                    country=target,
                                    partner=str(neighbor),
                                )
                            )
            elif isinstance(val, list):
                rows.extend(
                    _series_to_daily_gwh(
                        series_vals=val,
                        dt_index=dt_index,
                        day_keys=day_keys,
                        country=default_country,
                        partner=str(key),
                    )
                )

    return rows


def _parse_cbpf_payload(payload: Any, fallback_country: str = "DE") -> tuple[pd.DataFrame, list[str]]:
    """Parse Energy-Charts /cbpf JSON payloads into daily import/export rows."""
    if not isinstance(payload, dict):
        return pd.DataFrame(), [f"Payload is {type(payload).__name__}, expected dict"]

    timestamps = payload.get("unix_seconds") or payload.get("timestamps") or []
    if not isinstance(timestamps, list) or not timestamps:
        return pd.DataFrame(), [f"Missing unix_seconds; keys={list(payload.keys())[:10]}"]

    dt_index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Europe/Berlin")
    day_keys = pd.Index(dt_index.tz_localize(None).date)
    country = fallback_country.upper()

    rows: list[dict[str, Any]] = []
    if "countries" in payload and payload["countries"] is not None:
        rows.extend(_collect_from_container(payload["countries"], dt_index, day_keys, country))
    if not rows and "data" in payload and payload["data"] is not None:
        rows.extend(_collect_from_container(payload["data"], dt_index, day_keys, country))
    if not rows:
        rows.extend(_collect_from_container(payload, dt_index, day_keys, country))

    if not rows:
        return pd.DataFrame(), [f"0 series matched; keys={list(payload.keys())[:12]}"]

    return pd.DataFrame(rows), []


@st.cache_data(show_spinner="Reading JSON files...")
def load_json_folder(folder_path_str: str, glob_pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """Load JSON files from folder_path_str matching glob_pattern."""
    folder = Path(folder_path_str)
    if not folder.exists() or not folder.is_dir():
        return pd.DataFrame(), [f"Folder does not exist: {folder}"]

    files = sorted(folder.glob(glob_pattern))
    if not files:
        files = sorted(folder.glob("*.json"))

    if not files:
        return pd.DataFrame(), [f"No JSON files found in {folder} matching '{glob_pattern}'."]

    frames: list[pd.DataFrame] = []
    notes: list[str] = []

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            parsed, diag = _parse_cbpf_payload(payload, fallback_country=_extract_country_from_filename(fp))
            if parsed.empty:
                detail = "; ".join(diag) if diag else "no valid flow series found"
                notes.append(f"Skipped {fp.name}: {detail}.")
            else:
                frames.append(parsed)
        except Exception as exc:
            notes.append(f"Skipped {fp.name}: {exc}")

    if not frames:
        return pd.DataFrame(), notes

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.groupby(["date", "country", "partner"], as_index=False)[["import_gwh", "export_gwh"]]
        .sum()
        .sort_values(["date", "country", "partner"])
    )
    return df, notes


@st.cache_data(show_spinner="Fetching Energy-Charts /cbpf data...")
def fetch_cbpf_fallback(country: str, start_str: str, end_str: str) -> tuple[pd.DataFrame, str | None]:
    """Optional API fallback when local JSON files are unavailable."""
    try:
        resp = requests.get(
            CBPF_URL,
            params={
                "country": country.strip().lower(),
                "start": _to_unix_seconds(start_str, end_of_day=False),
                "end": _to_unix_seconds(end_str, end_of_day=True),
            },
            timeout=20,
        )
        resp.raise_for_status()
        df, diag = _parse_cbpf_payload(resp.json(), fallback_country=country.strip().upper())
        if df.empty:
            return pd.DataFrame(), "; ".join(diag) if diag else "API response contained no parseable flow series."
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _chord_patch(
    a_start: float,
    a_end: float,
    radius: float,
    color: Any,
    lw: float,
    alpha: float,
) -> PathPatch:
    """Quadratic Bezier chord connecting two angles around the circle."""
    x0, y0 = radius * np.cos(a_start), radius * np.sin(a_start)
    x1, y1 = radius * np.cos(a_end), radius * np.sin(a_end)
    cx, cy = 0.2 * (x0 + x1), 0.2 * (y0 + y1)
    path = MplPath([(x0, y0), (cx, cy), (x1, y1)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    return PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round")


def build_chord_figure(day_df: pd.DataFrame, target_date: pd.Timestamp, scale_max_gwh: float) -> plt.Figure:
    """Build circular diagram: arc size = gross flow (import + export), link = max(import, export)."""
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_aspect("equal")
    ax.axis("off")

    if day_df.empty:
        ax.text(0, 0, f"No data for {target_date.strftime('%Y-%m-%d')}", color="white", ha="center", va="center")
        return fig

    work = day_df.copy()
    work["gross_gwh"] = work["import_gwh"] + work["export_gwh"]
    work["max_gwh"] = np.maximum(work["import_gwh"], work["export_gwh"])

    nodes = sorted(set(work["country"]).union(set(work["partner"])))
    gross_map: dict[str, float] = {}
    for n in nodes:
        g = work.loc[work["country"] == n, "gross_gwh"].sum() + work.loc[work["partner"] == n, "gross_gwh"].sum()
        gross_map[n] = float(g)

    total_gross = sum(gross_map.values())
    if total_gross <= 0:
        total_gross = float(len(nodes))
        gross_map = {n: 1.0 for n in nodes}

    radius = 1.0
    gap = 0.03
    avail_span = max(2.0 * np.pi - gap * len(nodes), np.pi)
    cur = np.pi / 2.0
    angles: dict[str, tuple[float, float]] = {}
    arc_cmap = plt.cm.tab20

    for i, node in enumerate(nodes):
        span = (gross_map[node] / total_gross) * avail_span
        a0, a1 = cur, cur - span
        angles[node] = (a0, a1)

        ax.add_patch(
            Arc(
                (0, 0),
                width=2.0 * radius,
                height=2.0 * radius,
                angle=0.0,
                theta1=np.degrees(a1),
                theta2=np.degrees(a0),
                color=arc_cmap(i % 20),
                lw=10.0,
                solid_capstyle="butt",
            )
        )

        mid = 0.5 * (a0 + a1)
        ax.text(
            1.14 * radius * np.cos(mid),
            1.14 * radius * np.sin(mid),
            f"{node}\n{gross_map[node]:.1f} GWh",
            color="#f0f2f6",
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            va="center",
        )
        cur = a1 - gap

    denom = max(scale_max_gwh, float(work["max_gwh"].max()), 1e-6)
    link_cmap = plt.cm.plasma

    for _, row in work.iterrows():
        u, v = row["country"], row["partner"]
        if u not in angles or v not in angles or u == v:
            continue
        strength = float(np.clip(row["max_gwh"] / denom, 0.0, 1.0))
        if strength <= 0.01:
            continue

        u_mid = 0.5 * (angles[u][0] + angles[u][1])
        v_mid = 0.5 * (angles[v][0] + angles[v][1])
        ax.add_patch(
            _chord_patch(
                a_start=u_mid,
                a_end=v_mid,
                radius=radius * 0.96,
                color=link_cmap(strength),
                lw=0.8 + 6.5 * strength,
                alpha=0.25 + 0.65 * strength,
            )
        )

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.set_title(
        f"{target_date.strftime('%Y-%m-%d')} — Arc: Gross Flow (GWh) | Link: Max(Import, Export) (GWh)",
        color="#f0f2f6",
        fontsize=10.5,
        pad=14,
    )
    fig.tight_layout()
    return fig


def render_controls(num_dates: int) -> tuple[int, float]:
    """Render Play/Pause, Step +1, Days/frame, and delay slider."""
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("idx", 0)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        label = "Pause" if st.session_state.playing else "Play"
        if st.button(label, use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            st.rerun()

    with b2:
        if st.button("Step +1", use_container_width=True):
            st.session_state.playing = False
            st.session_state.idx = (int(st.session_state.idx) + 1) % max(num_dates, 1)

    with b3:
        step_days = st.number_input("Days per frame", min_value=1, max_value=30, value=1, step=1)

    delay_sec = st.slider("Frame delay (seconds)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    return int(step_days), float(delay_sec)


@st.fragment
def run_animated_plot(
    df: pd.DataFrame,
    dates: list[Any],
    step_days: int,
    delay_sec: float,
    scale_max_gwh: float,
) -> None:
    """Fragment loop updating the date slider, summary, and circular diagram."""
    status_box = st.empty()
    chart_box = st.empty()

    while True:
        idx = int(st.session_state.idx) % len(dates)
        st.session_state.idx = idx
        current_date = pd.Timestamp(dates[idx])

        picked_idx = st.slider(
            "Date",
            min_value=0,
            max_value=len(dates) - 1,
            value=idx,
            key="chord_date_slider",
        )
        if picked_idx != idx:
            st.session_state.idx = int(picked_idx)
            st.session_state.playing = False
            idx = int(st.session_state.idx)
            current_date = pd.Timestamp(dates[idx])

        day_df = df[df["date"] == current_date]
        gross_gwh = float((day_df["import_gwh"] + day_df["export_gwh"]).sum()) if not day_df.empty else 0.0
        max_link_gwh = float(np.maximum(day_df["import_gwh"], day_df["export_gwh"]).max()) if not day_df.empty else 0.0

        status_box.markdown(
            f"**Date:** `{current_date.strftime('%Y-%m-%d')}` ({idx + 1}/{len(dates)}) "
            f"| **Gross Flow:** `{gross_gwh:,.1f} GWh` "
            f"| **Peak Link:** `{max_link_gwh:,.1f} GWh`"
        )

        fig = build_chord_figure(day_df, current_date, scale_max_gwh=scale_max_gwh)
        with chart_box.container():
            st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if not st.session_state.playing:
            break

        time.sleep(delay_sec)
        st.session_state.idx = (idx + step_days) % len(dates)


def main() -> None:
    st.title("Stromflo — European Cross-Border Electricity Flows")

    with st.sidebar:
        st.header("Data")
        folder_str = st.text_input("JSON directory", value=str(DEFAULT_JSON_DIR))
        glob_pattern = st.text_input("File pattern", value="cbpf_*_2019.json")

        st.divider()
        use_fallback = st.checkbox("Fallback to Energy-Charts API if empty", value=True)
        fb_country = st.text_input("Fallback country", value="DE")
        fb_start = st.text_input("Fallback start (YYYY-MM-DD)", value="2019-01-01")
        fb_end = st.text_input("Fallback end (YYYY-MM-DD)", value="2019-01-31")

    df, parse_notes = load_json_folder(folder_str, glob_pattern)
    api_err: str | None = None

    if df.empty and use_fallback:
        df, api_err = fetch_cbpf_fallback(fb_country, fb_start, fb_end)

    if df.empty:
        st.warning("No flow data loaded.")
        for note in parse_notes:
            st.caption(note)
        if api_err:
            st.error(f"API fallback error: {api_err}")
        st.stop()

    dates = sorted(df["date"].unique())
    scale_max_gwh = float(np.maximum(df["import_gwh"], df["export_gwh"]).quantile(0.98))

    step_days, delay_sec = render_controls(len(dates))
    run_animated_plot(df, dates, step_days, delay_sec, scale_max_gwh=scale_max_gwh)


if __name__ == "__main__":
    main()
