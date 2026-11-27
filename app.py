"""Fraunhofer-style electricity flow circle for Germany and its 11 neighbours.

Design (kept from the original app)
  * One arc per country; arc length = share of the day's gross flow (import + export).
  * One ribbon per border; width = its gross flow, colour = the net exporting country.
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
from matplotlib.patches import PathPatch, Wedge
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
    return df, missing


# ------------------------------------------------------------------------- figure
RING_INNER, RING_OUTER = 0.945, 1.0
CENTER = np.zeros(2)


def point(cw_deg: float, radius: float) -> np.ndarray:
    """Point on the circle; angles run clockwise from the top."""
    theta = np.deg2rad(90.0 - cw_deg)
    return radius * np.array([np.cos(theta), np.sin(theta)])


def ribbon_path(slot_a: tuple[float, float], slot_b: tuple[float, float]) -> MplPath:
    """Closed chord ribbon: arc along A, quadratic curve to B, arc along B, curve back to A."""
    def arc(s0: float, s1: float) -> list[np.ndarray]:
        steps = max(2, int(abs(s1 - s0) / 1.5) + 2)
        return [point(t, RING_INNER) for t in np.linspace(s0, s1, steps)]

    pa, pb = arc(*slot_a), arc(*slot_b)
    # Pull quadratic Bezier control point slightly toward the center for natural chord curve
    ctrl_ab = 0.22 * (pa[-1] + pb[0])
    ctrl_ba = 0.22 * (pb[-1] + pa[0])

    verts = (
        pa
        + [ctrl_ab, pb[0]]
        + pb[1:]
        + [ctrl_ba, pa[0], pa[0]]
    )
    codes = (
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(pa) - 1)
        + [MplPath.CURVE3, MplPath.LINETO]
        + [MplPath.LINETO] * (len(pb) - 1)
        + [MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    )
    return MplPath(verts, codes)


def make_circle_figure(links: pd.DataFrame, date: pd.Timestamp) -> plt.Figure:
    """Chord diagram: arc length and ribbon width = gross flow (import + export)."""
    gross = {c: 0.0 for c in NODES}
    for r in links.itertuples():
        gross[r.country] += r.gross_gwh
        gross[r.partner] += r.gross_gwh
    total = sum(gross.values())

    bg = "#0e1117"
    fg = "#f0f2f6"
    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=120)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    ax.set_title(
        f"Electricity Flow Circle — {date:%Y-%m-%d}",
        fontsize=12.5,
        fontweight="bold",
        color=fg,
        pad=14,
    )

    if total <= 0:
        ax.text(0, 0, "No flow data for this day", ha="center", va="center", color="#9aa4b2", fontsize=11)
        fig.tight_layout()
        return fig

    # Subtle inner track guide
    ax.add_patch(
        plt.Circle(
            (0, 0),
            RING_INNER,
            fill=False,
            edgecolor=(1.0, 1.0, 1.0, 0.08),
            linewidth=0.8,
            linestyle="--",
        )
    )

    # Arc layout (degrees clockwise from the top). Small floor keeps idle nodes visible.
    weight = {c: g + 0.004 * total for c, g in gross.items()}
    usable = 360.0 - GAP_DEG * len(NODES)
    arcs, cursor = {}, 0.0
    for c in NODES:
        span = weight[c] / sum(weight.values()) * usable
        arcs[c] = (cursor, cursor + span)
        cursor += span + GAP_DEG
    mid = {c: (a + b) / 2 for c, (a, b) in arcs.items()}
    palette = [plt.cm.tab20(i % 20) for i in range(len(NODES))]
    color = {c: palette[i] for i, c in enumerate(NODES)}

    # Each link gets a slot inside both of its arcs. Ordering partners by clockwise
    # distance (farthest first) keeps the ribbons of one node from crossing each other.
    slots = {}
    for c, (a, b) in arcs.items():
        mine = [
            (r.partner if r.country == c else r.country, r.gross_gwh)
            for r in links.itertuples()
            if c in (r.country, r.partner)
        ]
        mine.sort(key=lambda x: -((mid[x[0]] - mid[c]) % 360.0))
        node_total = sum(w for _, w in mine) or 1.0
        pos = a
        for partner, w in mine:
            span = (b - a) * w / node_total
            slots[(c, partner)] = (pos, pos + span)
            pos += span

    # Draw outer country arcs and labels
    for c, (a, b) in arcs.items():
        ax.add_patch(
            Wedge(
                (0, 0),
                RING_OUTER,
                90 - b,
                90 - a,
                width=RING_OUTER - RING_INNER,
                facecolor=color[c][:3] + (0.88,),
                edgecolor=bg,
                linewidth=1.0,
            )
        )
        lx, ly = point(mid[c], 1.13)
        ax.text(
            lx,
            ly,
            f"{c}\n{gross[c]:,.0f}",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=fg,
            linespacing=1.15,
        )

    # Draw ribbons largest-first so smaller border flows stay visible on top
    for r in links.sort_values("gross_gwh", ascending=False).itertuples():
        if r.gross_gwh <= 0:
            continue
        exporter = r.partner if r.import_gwh >= r.export_gwh else r.country
        ax.add_patch(
            PathPatch(
                ribbon_path(slots[(r.country, r.partner)], slots[(r.partner, r.country)]),
                facecolor=color[exporter][:3] + (0.72,),
                edgecolor=(0.05, 0.07, 0.10, 0.35),
                linewidth=0.5,
            )
        )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- app
def main() -> None:
    st.title("Germany's cross-border electricity flows")

    with st.sidebar:
        st.header("Data & playback")
        year = int(st.number_input("Year", min_value=2000, max_value=2100, value=2019, step=1))
        folder = st.text_input("JSON folder", value=f"json_{year}", key=f"folder_{year}")
        neighbours = st.checkbox("Include links between neighbours", value=True)
        st.checkbox("Auto-play", key="playing")
        delay = st.slider("Step delay (s)", 0.05, 1.0, 0.2, 0.05)

    df, missing = load_flows(folder, year)
    if df.empty:
        st.warning(f"No cbpf_*_{year}.json files found in `{folder}`.")
        st.stop()
    if missing:
        st.sidebar.info("Files not found for: " + ", ".join(missing))

    links_all = df if neighbours else df[(df["country"] == "DE") | (df["partner"] == "DE")]
    dates = np.sort(links_all["date"].unique())
    n = len(dates)

    st.session_state.setdefault("day_idx", 0)
    st.session_state.day_idx = min(st.session_state.day_idx, n - 1)
    if st.session_state.pop("advance", False):  # set at the end of the previous run
        st.session_state.day_idx = (st.session_state.day_idx + 1) % n
    idx = st.slider("Day of year", 0, n - 1, key="day_idx")

    date = pd.Timestamp(dates[idx])
    st.subheader(f"{date:%A, %d %B %Y}")
    st.caption("Arc length and ribbon width show gross flow (import + export). "
               "Ribbon colour is the net exporting country on that border.")
    day = links_all[links_all["date"] == date]
    de = df[(df["date"] == date) & (df["country"] == "DE")]

    col_plot, col_data = st.columns([2, 1])
    with col_plot:
        fig = make_circle_figure(day, date)
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
