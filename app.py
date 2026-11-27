"""Fraunhofer-style electricity flow circle for Germany and its 11 neighbours.

Design (kept from the original app)
  * One arc per country; arc length = share of the day's gross flow (import + export).
  * One curved link per border; link width and colour = the larger of import / export.
  * A day slider and auto-play step through the year.

Data: Energy-Charts "cbpf" JSON files, one per country, e.g. json_2019/cbpf_de_2019.json.
Each file holds `unix_seconds` and a `countries` list of {name, data} where `data` is
power in GW (positive = import into the file's country).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path as MplPath

st.set_page_config(page_title="Germany electricity flow circle", layout="wide")

# Clockwise order around the circle, starting at the top. Germany closes the ring.
NODES = {
    "NO": "Norway", "SE": "Sweden", "DK": "Denmark", "PL": "Poland",
    "CZ": "Czechia", "AT": "Austria", "CH": "Switzerland", "FR": "France",
    "LU": "Luxembourg", "BE": "Belgium", "NL": "Netherlands", "DE": "Germany",
}
NAME_TO_CODE = {name.lower(): code for code, name in NODES.items()}
NAME_TO_CODE["czech republic"] = "CZ"
GAP_DEG = 2.0
TIMEZONE = "Europe/Berlin"


# --------------------------------------------------------------------------- data
def to_code(name: str) -> str | None:
    """Map a partner name from the JSON (e.g. 'Austria') to a two-letter code."""
    key = name.strip().lower()
    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]
    return key.upper() if key.upper() in NODES else None


def parse_file(path: Path, country: str) -> list[pd.DataFrame]:
    """Daily import/export energy (GWh) between `country` and each partner."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    seconds = payload.get("unix_seconds")
    series = payload.get("countries") or payload.get("data") or []
    if not seconds or not series:
        return []

    ts = pd.to_datetime(seconds, unit="s", utc=True)
    step_h = pd.Series(ts).diff().median().total_seconds() / 3600.0
    days = ts.tz_convert(TIMEZONE).tz_localize(None).normalize()

    frames = []
    for entry in series:
        partner = to_code(str(entry.get("name") or entry.get("code") or ""))
        if partner is None or partner == country:
            continue
        power = np.asarray(entry.get("data", []), dtype=float)  # GW, None -> NaN
        if len(power) != len(ts):
            continue
        energy = pd.Series(power * step_h, index=days)
        # Import / export are summed from the positive / negative intervals of each day.
        daily = pd.DataFrame({
            "import_gwh": energy.clip(lower=0).groupby(level=0).sum(),
            "export_gwh": (-energy).clip(lower=0).groupby(level=0).sum(),
        })
        frames.append(daily.rename_axis("date").reset_index().assign(country=country, partner=partner))
    return frames


@st.cache_data(show_spinner="Reading JSON files...")
def load_flows(folder: str, year: int) -> tuple[pd.DataFrame, list[str]]:
    """Load all node files. Germany is read first so its rows win when a border appears twice."""
    frames, missing = [], []
    for code in ["DE"] + [c for c in NODES if c != "DE"]:
        path = Path(folder) / f"cbpf_{code.lower()}_{year}.json"
        if path.exists():
            frames.extend(parse_file(path, code))
        else:
            missing.append(code)
    if not frames:
        return pd.DataFrame(), missing

    df = pd.concat(frames, ignore_index=True)
    df["pair"] = ["-".join(sorted(p)) for p in zip(df["country"], df["partner"])]
    df = df.drop_duplicates(["date", "pair"], keep="first")  # A->B and B->A are the same border
    df["gross_gwh"] = df["import_gwh"] + df["export_gwh"]
    df["max_gwh"] = df[["import_gwh", "export_gwh"]].max(axis=1)
    return df, missing


# ------------------------------------------------------------------------- figure
def make_circle_figure(links: pd.DataFrame, date: pd.Timestamp, max_ref: float) -> plt.Figure:
    gross = {c: 0.0 for c in NODES}
    for r in links.itertuples():
        gross[r.country] += r.gross_gwh
        gross[r.partner] += r.gross_gwh
    total = sum(gross.values())

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(f"Electricity flow circle - {date:%Y-%m-%d}", fontsize=12)
    if total <= 0:
        ax.text(0, 0, "No flow data for this day", ha="center", va="center")
        return fig

    # Arc layout (degrees clockwise from the top). A small floor keeps idle nodes visible.
    weight = {c: g + 0.004 * total for c, g in gross.items()}
    usable = 360.0 - GAP_DEG * len(NODES)
    arcs, cursor = {}, 0.0
    for c in NODES:
        span = weight[c] / sum(weight.values()) * usable
        arcs[c] = (cursor, cursor + span)
        cursor += span + GAP_DEG
    mid = {c: (a + b) / 2 for c, (a, b) in arcs.items()}

    # Attachment point of every link inside its arcs. Ordering partners by clockwise
    # distance (farthest first) keeps the links of one node from crossing each other.
    attach = {}
    for c, (a, b) in arcs.items():
        mine = [(r.partner if r.country == c else r.country, r.gross_gwh)
                for r in links.itertuples() if c in (r.country, r.partner)]
        mine.sort(key=lambda x: -((mid[x[0]] - mid[c]) % 360.0))
        node_total = sum(w for _, w in mine) or 1.0
        pos = a
        for partner, w in mine:
            span = (b - a) * w / node_total
            attach[(c, partner)] = pos + span / 2
            pos += span

    def point(cw_deg: float, radius: float) -> np.ndarray:
        theta = np.deg2rad(90.0 - cw_deg)
        return radius * np.array([np.cos(theta), np.sin(theta)])

    for i, (c, (a, b)) in enumerate(arcs.items()):
        ax.add_patch(Arc((0, 0), 2, 2, theta1=90 - b, theta2=90 - a, linewidth=10,
                         color="#2b2d42" if c == "DE" else plt.cm.tab20(i % 20)))
        x, y = point(mid[c], 1.15)
        ax.text(x, y, f"{c}\n{gross[c] / total * 100:.1f}%", ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    cmap = plt.cm.viridis
    for r in links.sort_values("max_gwh").itertuples():  # strongest links drawn last
        p1 = point(attach[(r.country, r.partner)], 0.94)
        p2 = point(attach[(r.partner, r.country)], 0.94)
        ctrl = 0.12 * (p1 + p2)
        strength = float(np.clip(r.max_gwh / max_ref, 0.0, 1.0))
        path = MplPath([p1, ctrl, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=cmap(strength),
                               linewidth=0.7 + 5.5 * strength ** 0.7, alpha=0.75))

    fig.colorbar(ScalarMappable(Normalize(0, max_ref), cmap=cmap), ax=ax, fraction=0.035,
                 pad=0.02, label="Larger of import / export per day (GWh)")
    return fig


# --------------------------------------------------------------------------- app
def main() -> None:
    st.title("Germany's cross-border electricity flows")

    with st.sidebar:
        st.header("Data & playback")
        year = int(st.number_input("Year", min_value=2000, max_value=2100, value=2019, step=1))
        folder = st.text_input("JSON folder", value=f"json_{year}", key=f"folder_{year}")
        neighbours = st.checkbox("Include links between neighbours", value=False)
        st.checkbox("Auto-play", key="playing")
        delay = st.slider("Step delay (s)", 0.05, 1.0, 0.2, 0.05)

    df, missing = load_flows(folder, year)
    if df.empty:
        st.warning(f"No cbpf_*_{year}.json files found in `{folder}`.")
        st.stop()
    if missing:
        st.sidebar.info("Files not found for: " + ", ".join(missing))

    links_all = df if neighbours else df[(df["country"] == "DE") | (df["partner"] == "DE")]
    max_ref = max(float(links_all["max_gwh"].max()), 1e-6)  # fixed scale for the whole year
    dates = np.sort(links_all["date"].unique())
    n = len(dates)

    st.session_state.setdefault("day_idx", 0)
    st.session_state.day_idx = min(st.session_state.day_idx, n - 1)
    if st.session_state.pop("advance", False):  # set at the end of the previous run
        st.session_state.day_idx = (st.session_state.day_idx + 1) % n
    idx = st.slider("Day of year", 0, n - 1, key="day_idx")

    date = pd.Timestamp(dates[idx])
    st.subheader(f"{date:%A, %d %B %Y}")
    day = links_all[links_all["date"] == date]
    de = df[(df["date"] == date) & (df["country"] == "DE")]

    col_plot, col_data = st.columns([2, 1])
    with col_plot:
        fig = make_circle_figure(day, date, max_ref)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with col_data:
        st.subheader("Germany, this day (GWh)")
        if de.empty:
            st.info("No Germany data (cbpf_de file missing).")
        else:
            imp, exp = de["import_gwh"].sum(), de["export_gwh"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Import", f"{imp:,.0f}")
            c2.metric("Export", f"{exp:,.0f}")
            c3.metric("Net", f"{imp - exp:+,.0f}")
            table = pd.DataFrame({
                "Neighbour": de["partner"].map(NODES),
                "Import": de["import_gwh"],
                "Export": de["export_gwh"],
                "Net": de["import_gwh"] - de["export_gwh"],
            }).sort_values("Import", ascending=False)
            st.dataframe(table.round(1), hide_index=True)

    if st.session_state.get("playing"):
        time.sleep(delay)
        st.session_state["advance"] = True
        st.rerun()


if __name__ == "__main__":
    main()
