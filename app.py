import json
import re
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="2019 Electricity Flow Circle", layout="wide")


@st.cache_data
def load_2019_jsons(json_dir_str: str, pattern: str = "cbpf_*_2019.json") -> pd.DataFrame:
    """Parse cbpf_*_2019.json files directly into a daily country-partner DataFrame."""
    json_dir = Path(json_dir_str)
    files = sorted(json_dir.glob(pattern))
    if not files:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for path in files:
        match = re.search(r"cbpf_([a-zA-Z]{2})_\d{4}\.json$", path.name)
        country = match.group(1).upper() if match else path.stem.upper()

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        unix_sec = payload.get("unix_seconds")
        series_list = payload.get("countries") or payload.get("data") or []
        if not unix_sec or not series_list:
            continue

        ts = pd.to_datetime(unix_sec, unit="s", utc=True)
        dt_hours = ts.to_series().diff().dt.total_seconds().div(3600.0)
        med_dt = dt_hours.dropna().median() if len(dt_hours) > 1 else 1.0
        dt_hours = dt_hours.fillna(med_dt if pd.notna(med_dt) and med_dt > 0 else 1.0)

        for entry in series_list:
            partner = str(entry.get("name") or entry.get("code") or "unknown")
            power_gw = pd.to_numeric(pd.Series(entry.get("data", []), index=ts), errors="coerce")
            energy_gwh = power_gw * dt_hours.to_numpy()

            s_df = pd.DataFrame({"energy_gwh": energy_gwh}, index=ts)
            s_df["date"] = s_df.index.date

            daily = s_df.groupby("date", as_index=False)["energy_gwh"].sum().rename(
                columns={"energy_gwh": "net_energy_gwh"}
            )
            daily["import_gwh"] = daily["net_energy_gwh"].clip(lower=0.0)
            daily["export_gwh"] = (-daily["net_energy_gwh"]).clip(lower=0.0)
            daily["gross_flow_gwh"] = daily["import_gwh"] + daily["export_gwh"]
            daily["max_flow_gwh"] = np.maximum(daily["import_gwh"], daily["export_gwh"])
            daily["country"] = country
            daily["partner"] = partner
            rows.append(daily)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def make_circle_figure(day_df: pd.DataFrame, selected_date: pd.Timestamp) -> plt.Figure:
    """Build circular chart: arc fraction = share of import+export, link width/color = max."""
    c_totals = (
        day_df.groupby("country", as_index=False)["gross_flow_gwh"]
        .sum()
        .sort_values("gross_flow_gwh", ascending=False)
    )
    total_gross = c_totals["gross_flow_gwh"].sum()

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140)
    ax.set_aspect("equal")
    ax.axis("off")

    if total_gross <= 0 or c_totals.empty:
        ax.text(0, 0, f"No data for {selected_date.date()}", ha="center", va="center")
        return fig

    radius = 1.0
    gap_deg = 2.0
    usable_deg = max(360.0 - gap_deg * len(c_totals), 180.0)
    arc_angles = {}
    cursor_deg = 90.0

    for idx, row in c_totals.iterrows():
        c = row["country"]
        frac = row["gross_flow_gwh"] / total_gross
        span = frac * usable_deg

        ax.add_patch(
            Arc(
                (0.0, 0.0),
                2.0 * radius,
                2.0 * radius,
                angle=0.0,
                theta1=cursor_deg - span,
                theta2=cursor_deg,
                linewidth=10,
                color=plt.cm.tab20(idx % 20),
            )
        )

        mid_rad = np.deg2rad(cursor_deg - 0.5 * span)
        arc_angles[c] = mid_rad
        ax.text(
            1.15 * radius * np.cos(mid_rad),
            1.15 * radius * np.sin(mid_rad),
            f"{c}\n{frac * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        cursor_deg -= span + gap_deg

    links = day_df.copy()
    links["p_code"] = links["partner"].str.upper().str.extract(r"([A-Z]{2})")[0]
    links = links[links["p_code"].isin(arc_angles.keys()) & (links["country"] != links["p_code"])]
    max_ref = max(links["max_flow_gwh"].max() if not links.empty else 1.0, 1e-6)

    for _, r in links.iterrows():
        a1, a2 = arc_angles[r["country"]], arc_angles[r["p_code"]]
        p1 = (0.94 * radius * np.cos(a1), 0.94 * radius * np.sin(a1))
        p2 = (0.94 * radius * np.cos(a2), 0.94 * radius * np.sin(a2))
        ctrl = (0.15 * radius * np.cos(0.5 * (a1 + a2)), 0.15 * radius * np.sin(0.5 * (a1 + a2)))
        strength = float(r["max_flow_gwh"]) / max_ref

        path = MplPath([p1, ctrl, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=plt.cm.viridis(np.clip(strength, 0.0, 1.0)),
                linewidth=0.7 + 5.5 * (strength ** 0.7),
                alpha=0.75,
            )
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(f"Electricity Flow Circle — {selected_date.strftime('%Y-%m-%d')}", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Fraunhofer-Style Electricity Flow Circle (2019 JSON)")

    with st.sidebar:
        st.header("Data & Playback")
        json_folder = st.text_input("JSON Folder Path", value="json_2019")
        file_glob = st.text_input("File Glob Pattern", value="cbpf_*_2019.json")
        auto_play = st.checkbox("Auto-step dates", value=False)
        step_delay = st.slider("Step delay (sec)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)

    df = load_2019_jsons(json_folder, file_glob)
    if df.empty:
        st.warning(f"No valid JSON files found matching `{file_glob}` in `{json_folder}`.")
        st.stop()

    dates = sorted(df["date"].unique())
    if "date_idx" not in st.session_state:
        st.session_state.date_idx = 0

    selected_idx = st.slider(
        "Day Index (2019)",
        min_value=0,
        max_value=len(dates) - 1,
        value=st.session_state.date_idx,
        key="slider_2019",
    )
    st.session_state.date_idx = selected_idx

    selected_date = pd.Timestamp(dates[selected_idx])
    st.write(f"**Selected Date:** {selected_date.strftime('%Y-%m-%d')}")

    day_df = df[df["date"] == selected_date]
    col_plot, col_data = st.columns([2.0, 1.0])

    with col_plot:
        fig = make_circle_figure(day_df, selected_date)
        st.pyplot(fig, clear_figure=True)

    with col_data:
        st.subheader("Daily Country Shares")
        summary = (
            day_df.groupby("country", as_index=False)[["import_gwh", "export_gwh", "gross_flow_gwh", "max_flow_gwh"]]
            .sum()
            .sort_values("gross_flow_gwh", ascending=False)
        )
        total = summary["gross_flow_gwh"].sum()
        summary["arc_%"] = (summary["gross_flow_gwh"] / total * 100.0) if total > 0 else 0.0
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

    if auto_play:
        st.session_state.date_idx = (selected_idx + 1) % len(dates)
        time.sleep(step_delay)
        st.rerun()


if __name__ == "__main__":
    main()
