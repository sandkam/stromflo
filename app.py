import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Stromflo — European Cross-Border Flows", layout="wide")

CBPF_URL = "https://api.energy-charts.info/cbpf"
DEFAULT_COUNTRIES = [
    "DE",
    "FR",
    "NL",
    "BE",
    "DK",
    "AT",
    "CH",
    "CZ",
    "PL",
    "SE",
    "NO",
    "IT",
    "ES",
]


def _parse_flow_series(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse an Energy-Charts /cbpf JSON payload into daily energy (GWh) rows.

    API convention: positive values = imports into the target country;
    negative values = exports from the target country.
    """
    timestamps = payload.get("unix_seconds") or payload.get("timestamps") or []
    if not timestamps:
        return pd.DataFrame()

    dt_index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Europe/Berlin")
    records: list[dict[str, Any]] = []

    # Support either {"countries": {code: {neighbor: [vals...]}}} or flat {neighbor: [vals...]}
    source_map = payload.get("countries") if isinstance(payload.get("countries"), dict) else payload
    if not isinstance(source_map, dict):
        return pd.DataFrame()

    for key, val in source_map.items():
        if key in {"unix_seconds", "timestamps", "status", "license_info"}:
            continue
        if isinstance(val, dict):
            # Nested mapping: target -> neighbor -> list of power values (MW)
            target = str(key)
            for neighbor, series in val.items():
                if not isinstance(series, list) or len(series) != len(dt_index):
                    continue
                s = pd.to_numeric(pd.Series(series, index=dt_index), errors="coerce").fillna(0.0)
                # Approximate step duration in hours from median timestamp delta
                deltas_h = dt_index.to_series().diff().dt.total_seconds().div(3600.0).dropna()
                dt_h = float(deltas_h.median()) if not deltas_h.empty and deltas_h.median() > 0 else 1.0

                imp_gwh = s.clip(lower=0.0).mul(dt_h).div(1000.0).groupby(s.index.date).sum()
                exp_gwh = s.clip(upper=0.0).abs().mul(dt_h).div(1000.0).groupby(s.index.date).sum()

                for d in imp_gwh.index:
                    records.append(
                        {
                            "date": pd.Timestamp(d),
                            "country": target,
                            "partner": str(neighbor),
                            "import_gwh": float(imp_gwh.loc[d]),
                            "export_gwh": float(exp_gwh.loc[d]),
                        }
                    )
        elif isinstance(val, list) and len(val) == len(dt_index):
            # Flat series for a single country (e.g., DE) where positive = import, negative = export
            s = pd.to_numeric(pd.Series(val, index=dt_index), errors="coerce").fillna(0.0)
            deltas_h = dt_index.to_series().diff().dt.total_seconds().div(3600.0).dropna()
            dt_h = float(deltas_h.median()) if not deltas_h.empty and deltas_h.median() > 0 else 1.0

            imp_gwh = s.clip(lower=0.0).mul(dt_h).div(1000.0).groupby(s.index.date).sum()
            exp_gwh = s.clip(upper=0.0).abs().mul(dt_h).div(1000.0).groupby(s.index.date).sum()

            for d in imp_gwh.index:
                records.append(
                    {
                        "date": pd.Timestamp(d),
                        "country": "DE",
                        "partner": str(key),
                        "import_gwh": float(imp_gwh.loc[d]),
                        "export_gwh": float(exp_gwh.loc[d]),
                    }
                )

    return pd.DataFrame(records)


@st.cache_data(show_spinner="Loading flow data...")
def load_flow_data(folder_str: str, pattern: str) -> pd.DataFrame:
    """Load matching JSON files from disk and aggregate to daily import/export (GWh)."""
    folder = Path(folder_str)
    if not folder.exists() or not folder.is_dir():
        return pd.DataFrame()

    files = sorted(folder.glob(pattern))
    frames: list[pd.DataFrame] = []

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            parsed = _parse_flow_series(payload)
            if not parsed.empty:
                frames.append(parsed)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["date", "country", "partner"], as_index=False)[["import_gwh", "export_gwh"]]
        .sum()
        .sort_values(["date", "country", "partner"])
    )
    return grouped


@st.cache_data(show_spinner="Fetching sample data from Energy-Charts API...")
def fetch_api_sample(country: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """Fetch a date range from Energy-Charts /cbpf and convert to daily GWh."""
    try:
        resp = requests.get(
            CBPF_URL,
            params={"country": country.lower(), "start": start_iso, "end": end_iso},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        df = _parse_flow_series(payload)
        if not df.empty:
            df["country"] = country.upper()
        return df
    except Exception:
        return pd.DataFrame()


def _bezier_chord_patch(
    start_angle: float,
    end_angle: float,
    radius: float,
    color: Any,
    lw: float,
    alpha: float,
) -> PathPatch:
    """Create a curved chord path between two angles on a circle of given radius."""
    x0, y0 = radius * np.cos(start_angle), radius * np.sin(start_angle)
    x1, y1 = radius * np.cos(end_angle), radius * np.sin(end_angle)
    # Control point pulled inward toward center for chord curvature
    cx, cy = 0.25 * (x0 + x1), 0.25 * (y0 + y1)

    verts = [(x0, y0), (cx, cy), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    return PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round")


def make_circle_figure(day_df: pd.DataFrame, selected_date: pd.Timestamp, max_ref: float) -> plt.Figure:
    """Render circular chord diagram where arc size = gross flow and link = max(import, export)."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_aspect("equal")
    ax.axis("off")

    if day_df.empty:
        ax.text(0, 0, f"No flow data for {selected_date.strftime('%Y-%m-%d')}", color="white", ha="center", va="center")
        return fig

    work = day_df.copy()
    work["gross_gwh"] = work["import_gwh"] + work["export_gwh"]
    work["max_flow_gwh"] = np.maximum(work["import_gwh"], work["export_gwh"])

    nodes = sorted(set(work["country"]).union(set(work["partner"])))
    gross_by_node: dict[str, float] = {}
    for node in nodes:
        g_main = work.loc[work["country"] == node, "gross_gwh"].sum()
        g_part = work.loc[work["partner"] == node, "gross_gwh"].sum()
        gross_by_node[node] = float(g_main + g_part)

    total_gross = sum(gross_by_node.values())
    if total_gross <= 0:
        total_gross = float(len(nodes))
        gross_by_node = {n: 1.0 for n in nodes}

    radius = 1.0
    gap_rad = 0.03
    usable_span = max(2.0 * np.pi - gap_rad * len(nodes), np.pi)

    node_angles: dict[str, tuple[float, float]] = {}
    cur = np.pi / 2.0
    cmap = plt.cm.tab20

    for idx, node in enumerate(nodes):
        span = (gross_by_node[node] / total_gross) * usable_span
        a0 = cur
        a1 = cur - span
        node_angles[node] = (a0, a1)

        # Draw outer arc proportional to gross flow (import + export)
        deg0, deg1 = np.degrees(a1), np.degrees(a0)
        arc = Arc(
            (0, 0),
            width=2.0 * radius,
            height=2.0 * radius,
            angle=0.0,
            theta1=deg0,
            theta2=deg1,
            color=cmap(idx % 20),
            lw=10.0,
            solid_capstyle="butt",
        )
        ax.add_patch(arc)

        mid = 0.5 * (a0 + a1)
        lx, ly = 1.13 * radius * np.cos(mid), 1.13 * radius * np.sin(mid)
        ax.text(
            lx,
            ly,
            f"{node}\n({gross_by_node[node]:.1f})",
            color="#f0f2f6",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        cur = a1 - gap_rad

    # Draw links sized and colored by max(import, export)
    link_cmap = plt.cm.plasma
    denom = max(max_ref, float(work["max_flow_gwh"].max()), 1e-6)

    for _, row in work.iterrows():
        u, v = row["country"], row["partner"]
        if u not in node_angles or v not in node_angles or u == v:
            continue
        intensity = float(np.clip(row["max_flow_gwh"] / denom, 0.0, 1.0))
        if intensity <= 0.01:
            continue

        u_mid = 0.5 * (node_angles[u][0] + node_angles[u][1])
        v_mid = 0.5 * (node_angles[v][0] + node_angles[v][1])

        patch = _bezier_chord_patch(
            start_angle=u_mid,
            end_angle=v_mid,
            radius=radius * 0.96,
            color=link_cmap(intensity),
            lw=0.8 + 6.5 * intensity,
            alpha=0.25 + 0.65 * intensity,
        )
        ax.add_patch(patch)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(
        f"Cross-Border Electricity Flows — {selected_date.strftime('%Y-%m-%d')} (Arc = Gross GWh, Link = Max GWh)",
        color="#f0f2f6",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()
    return fig


def render_playback_controls(num_dates: int) -> tuple[int, float]:
    """Render Play/Pause, Step +1, step size, and delay controls."""
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("date_idx", 0)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Play" if not st.session_state.is_playing else "Pause", use_container_width=True):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()

    with c2:
        if st.button("Step +1", use_container_width=True):
            st.session_state.is_playing = False
            st.session_state.date_idx = (int(st.session_state.date_idx) + 1) % max(num_dates, 1)

    with c3:
        step_days = st.number_input("Days / frame", min_value=1, max_value=30, value=1, step=1)

    delay_sec = st.slider("Frame delay (sec)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    return int(step_days), float(delay_sec)


@st.fragment
def animated_circle_view(df: pd.DataFrame, dates: list[Any], step_days: int, delay_sec: float, max_ref: float) -> None:
    """Update date readout, slider, and circular plot smoothly during playback."""
    info_placeholder = st.empty()
    plot_placeholder = st.empty()

    while True:
        idx = int(st.session_state.date_idx) % len(dates)
        st.session_state.date_idx = idx
        selected_date = pd.Timestamp(dates[idx])

        slider_val = st.slider(
            "Scrub date",
            min_value=0,
            max_value=len(dates) - 1,
            value=idx,
            key="stromflo_date_slider",
        )
        if slider_val != idx:
            st.session_state.date_idx = int(slider_val)
            st.session_state.is_playing = False
            idx = int(st.session_state.date_idx)
            selected_date = pd.Timestamp(dates[idx])

        day_df = df[df["date"] == selected_date]
        gross_total = float((day_df["import_gwh"] + day_df["export_gwh"]).sum()) if not day_df.empty else 0.0

        info_placeholder.markdown(
            f"**Date:** `{selected_date.strftime('%Y-%m-%d')}` ({idx + 1} / {len(dates)}) "
            f"| **Daily Gross Flow:** `{gross_total:,.1f} GWh`"
        )

        fig = make_circle_figure(day_df, selected_date, max_ref=max_ref)
        with plot_placeholder.container():
            st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if not st.session_state.is_playing:
            break

        time.sleep(delay_sec)
        st.session_state.date_idx = (idx + step_days) % len(dates)


def main() -> None:
    st.title("Stromflo — European Cross-Border Electricity Flows")

    repo_root = Path(__file__).resolve().parent
    default_dir = (
        repo_root / "json_2019"
        if (repo_root / "json_2019").exists()
        else repo_root / "sample_json_2019"
    )

    with st.sidebar:
        st.header("Data Source")
        folder_input = st.text_input("JSON folder", value=str(default_dir))
        pattern_input = st.text_input("Filename glob", value="cbpf_*_2019.json")

        st.divider()
        st.caption("Fallback if no local JSON files match:")
        api_country = st.selectbox("API country", options=DEFAULT_COUNTRIES, index=0)
        api_start = st.text_input("Start date (YYYY-MM-DD)", value="2019-01-01")
        api_end = st.text_input("End date (YYYY-MM-DD)", value="2019-01-31")
        use_api_fallback = st.checkbox("Use Energy-Charts API fallback", value=True)

    df = load_flow_data(folder_input, pattern_input)

    if df.empty and use_api_fallback:
        df = fetch_api_sample(api_country, api_start, api_end)

    if df.empty:
        st.warning(
            f"No valid JSON files found matching `{pattern_input}` in `{folder_input}`, "
            "and API fallback returned no rows."
        )
        st.stop()

    dates = sorted(df["date"].unique())
    max_ref = float(np.maximum(df["import_gwh"], df["export_gwh"]).quantile(0.98))

    step_days, delay_sec = render_playback_controls(len(dates))
    animated_circle_view(df, dates, step_days, delay_sec, max_ref=max_ref)


if __name__ == "__main__":
    main()
