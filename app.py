import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
API_URL = "https://api.energy-charts.info/v2/cbpf"
YEAR_DEFAULT = 2025

COUNTRIES = {
    "DE": "Germany",
    "NO": "Norway",
    "SE": "Sweden",
    "DK": "Denmark",
    "PL": "Poland",
    "CZ": "Czechia",
    "AT": "Austria",
    "CH": "Switzerland",
    "FR": "France",
    "LU": "Luxembourg",
    "BE": "Belgium",
    "NL": "Netherlands",
}

# Circular order chosen to reflect Germany's geographic neighbourhood.
RING_ORDER = ["NO", "SE", "DK", "PL", "CZ", "AT", "CH", "FR", "LU", "BE", "NL"]

# Colours: Germany-connected links are highlighted; non-German links are neutral.
IMPORT_COLOR = "#20A6A6"   # flow into Germany
EXPORT_COLOR = "#E58B36"   # flow out of Germany
BACKGROUND_COLOR = "rgba(125, 132, 140, 0.34)"
NODE_GREY = "#AEB6BF"
GERMANY_COLOR = "#E9B949"


# ------------------------------------------------------------
# Data acquisition and normalization
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_cbpf_all(start: str, end: str) -> dict:
    """Fetch the historical CBPF series for all European countries."""
    response = requests.get(
        API_URL,
        params={"country": "all", "start": start, "end": end},
        timeout=90,
        headers={"User-Agent": "Germany-CBPF-Streamlit-Visualization/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Unexpected API response: the v2 'data' field is missing.")
    return payload


def country_code_from_text(text: str):
    """Find a country code in a series ID/name, if the API exposes pair IDs."""
    text = str(text).upper()
    # Prefer two-letter codes delimited by punctuation/word boundaries.
    for code in COUNTRIES:
        if re.search(rf"(?<![A-Z]){code}(?![A-Z])", text):
            return code
    aliases = {
        "GERMANY": "DE", "NORWAY": "NO", "SWEDEN": "SE",
        "DENMARK": "DK", "POLAND": "PL", "CZECHIA": "CZ",
        "CZECH REPUBLIC": "CZ", "AUSTRIA": "AT",
        "SWITZERLAND": "CH", "FRANCE": "FR",
        "LUXEMBOURG": "LU", "BELGIUM": "BE",
        "NETHERLANDS": "NL",
    }
    for name, code in aliases.items():
        if name in text:
            return code
    return None


def normalize_all_country_payload(payload: dict) -> pd.DataFrame:
    """
    Convert the v2 row-oriented response into:
      timestamp, source, neighbour, flow_gw

    The all-country endpoint is expected to expose pair/border series. Since
    API series IDs may evolve, this parser deliberately validates the mapping
    rather than silently assigning a wrong country pair.
    """
    descriptors = payload.get("series", [])
    rows = payload.get("data", [])
    if not rows or not descriptors:
        raise ValueError("The API returned no CBPF series for this date range.")

    # Build a lookup from series ID to a country pair.
    pair_for_id = {}
    for item in descriptors:
        sid = str(item.get("id", ""))
        label = str(item.get("name", ""))
        combined = f"{sid} {label}"
        # Preserve the order in the series ID: for a pair ID such as
        # DE_AT, the first code is treated as the country whose import/export
        # sign is reported. This matters when translating the sign to Germany.
        ordered = re.findall(r"(?<![A-Z])(?:DE|NO|SE|DK|PL|CZ|AT|CH|FR|LU|BE|NL)(?![A-Z])", sid.upper())
        if len(ordered) >= 2:
            pair_for_id[sid] = (ordered[0], ordered[1])
        else:
            # Fall back to ordered country-name/code tokens in the descriptor.
            ordered = re.findall(r"(?<![A-Z])(?:DE|NO|SE|DK|PL|CZ|AT|CH|FR|LU|BE|NL)(?![A-Z])", label.upper())
            if len(ordered) >= 2:
                pair_for_id[sid] = (ordered[0], ordered[1])

    # Some API deployments may encode country names as IDs.
    if not pair_for_id:
        raise ValueError(
            "The API's country=all response did not expose recognizable "
            "two-country series IDs. See the README troubleshooting note; "
            "the app stops rather than guessing which border a series represents."
        )

    normalized = []
    for row in rows:
        timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if pd.isna(timestamp):
            continue
        values = row.get("values", {})
        for sid, pair in pair_for_id.items():
            value = values.get(sid)
            if value is None or pd.isna(value):
                continue
            a, b = pair
            if a not in COUNTRIES or b not in COUNTRIES or a == b:
                continue
            normalized.append({
                "timestamp": timestamp,
                "a": a,
                "b": b,
                "flow_gw": float(value),
                "series_id": sid,
            })

    if not normalized:
        raise ValueError("No usable country-pair observations were found in the API response.")

    df = pd.DataFrame(normalized)
    # Remove duplicate representations of the same border if the API returns
    # both directions. Keep one consistently oriented series per border.
    df["border"] = df.apply(lambda r: tuple(sorted((r["a"], r["b"]))), axis=1)
    df = df.sort_values("timestamp").drop_duplicates(["timestamp", "border"], keep="first")
    return df


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_historical_flows(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    payload = fetch_cbpf_all(start, end)
    df = normalize_all_country_payload(payload)
    metadata = {
        "unit": payload.get("unit", "GW"),
        "timezone": payload.get("timezone", "API local time"),
        "resolution": payload.get("resolution", "unspecified"),
        "license": payload.get("license", "Check API response"),
        "generated_at": payload.get("generated_at", "not provided"),
        "available_from": payload.get("available_from", start),
        "available_until": payload.get("available_until", end),
    }
    return df, metadata


def daily_mean_flows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["timestamp"]).dt.date
    return (
        work.groupby(["date", "a", "b"], as_index=False)["flow_gw"]
        .mean()
    )


# ------------------------------------------------------------
# Plotly radial network
# ------------------------------------------------------------
def bezier_points(p0, p1, p2, p3, n=30):
    t = np.linspace(0, 1, n)
    x = ((1-t)**3)*p0[0] + 3*((1-t)**2)*t*p1[0] + 3*(1-t)*(t**2)*p2[0] + (t**3)*p3[0]
    y = ((1-t)**3)*p0[1] + 3*((1-t)**2)*t*p1[1] + 3*(1-t)*(t**2)*p2[1] + (t**3)*p3[1]
    return np.column_stack([x, y])


def ribbon_polygon(p0, p3, width, bend=0.20, n=32):
    """A tapered, filled cubic-Bezier ribbon polygon."""
    v = np.array(p3) - np.array(p0)
    length = np.linalg.norm(v)
    if length == 0:
        return None
    normal = np.array([-v[1], v[0]]) / length
    # Control points pull the curve toward the centre, creating a radial chord.
    c1 = np.array(p0) * (1-bend) + np.array([0.0, 0.0]) * bend
    c2 = np.array(p3) * (1-bend) + np.array([0.0, 0.0]) * bend
    center = bezier_points(p0, c1, c2, p3, n=n)
    # Taper slightly at the far end for a clean ribbon silhouette.
    taper = np.linspace(0.62, 1.0, n)[:, None]
    half = normal[None, :] * (width * taper / 2)
    upper = center + half
    lower = center - half
    polygon = np.vstack([upper, lower[::-1], upper[:1]])
    return polygon


def make_network_figure(day_df: pd.DataFrame, selected_day, max_flow: float,
                        show_labels=True, background_opacity=0.34):
    # Put Germany at the centre and neighbouring countries around the ring.
    radius = 1.0
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, len(RING_ORDER), endpoint=False)
    positions = {"DE": (0.0, 0.0)}
    for code, angle in zip(RING_ORDER, angles):
        positions[code] = (radius*np.cos(angle), radius*np.sin(angle))

    fig = go.Figure()
    if day_df.empty:
        return fig

    # Background country-to-country connections (grey).
    # Flow values are shown only when the API returned a country-pair series.
    bg = day_df[(day_df["a"] != "DE") & (day_df["b"] != "DE")].copy()
    for _, row in bg.iterrows():
        a, b = row["a"], row["b"]
        if a not in positions or b not in positions:
            continue
        magnitude = abs(float(row["flow_gw"]))
        width = 0.006 + 0.035 * np.sqrt(magnitude / max(max_flow, 1e-9))
        poly = ribbon_polygon(positions[a], positions[b], width, bend=0.12)
        if poly is None:
            continue
        fig.add_trace(go.Scatter(
            x=poly[:, 0], y=poly[:, 1], mode="lines",
            fill="toself", fillcolor=BACKGROUND_COLOR,
            line=dict(color=BACKGROUND_COLOR, width=0.2),
            hovertemplate=(
                f"{COUNTRIES[a]} ↔ {COUNTRIES[b]}<br>"
                f"Signed flow relative to {COUNTRIES[a]}: {row['flow_gw']:.2f} GW"
                "<extra>Other European borders</extra>"
            ),
            showlegend=False,
        ))

    # Germany-connected ribbons, highlighted by net direction.
    de = day_df[(day_df["a"] == "DE") | (day_df["b"] == "DE")].copy()
    for _, row in de.iterrows():
        partner = row["b"] if row["a"] == "DE" else row["a"]
        if partner not in positions:
            continue
        # The pair series is oriented from row["a"] to row["b"]. CBPF is
        # positive for import into the queried/source country. Convert it to
        # Germany's perspective so positive always means import into Germany.
        raw_value = float(row["flow_gw"])
        value = raw_value if row["a"] == "DE" else -raw_value
        colour = IMPORT_COLOR if value >= 0 else EXPORT_COLOR
        magnitude = abs(value)
        width = 0.018 + 0.105 * np.sqrt(magnitude / max(max_flow, 1e-9))
        poly = ribbon_polygon(positions["DE"], positions[partner], width, bend=0.28)
        if poly is None:
            continue
        direction = "Import to Germany" if value >= 0 else "Export from Germany"
        fig.add_trace(go.Scatter(
            x=poly[:, 0], y=poly[:, 1], mode="lines",
            fill="toself", fillcolor=colour,
            line=dict(color=colour, width=0.5),
            hovertemplate=(
                f"Germany ↔ {COUNTRIES[partner]}<br>"
                f"{direction}<br>"
                f"Signed flow: {value:.2f} GW"
                "<extra>Germany border flow</extra>"
            ),
            showlegend=False,
        ))

    # Nodes on top of the ribbons.
    xs, ys, labels, sizes, colours, hovers = [], [], [], [], [], []
    for code in ["DE"] + RING_ORDER:
        x, y = positions[code]
        xs.append(x); ys.append(y)
        labels.append(code if show_labels else "")
        sizes.append(29 if code == "DE" else 17)
        colours.append(GERMANY_COLOR if code == "DE" else NODE_GREY)
        hovers.append(COUNTRIES[code])

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text" if show_labels else "markers",
        text=labels, textposition="middle center",
        textfont=dict(size=11, color="#18212B", family="Arial"),
        marker=dict(size=sizes, color=colours, line=dict(color="white", width=1.2)),
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    # Central Germany label.
    fig.add_annotation(
        x=0, y=0, text="<b>GERMANY</b>", showarrow=False,
        font=dict(size=11, color="#17202A"), yshift=-27
    )

    # Legend keys.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=IMPORT_COLOR, width=8),
        name="Import to Germany (positive CBPF)",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=EXPORT_COLOR, width=8),
        name="Export from Germany (negative CBPF)",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(125,132,140,0.65)", width=5),
        name="Other country pairs (grey)",
    ))

    fig.update_layout(
        height=760,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Cross-border physical electricity flows · {selected_day}", x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.03, xanchor="center", x=0.5),
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(visible=False, range=[-1.28, 1.28], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.28, 1.28]),
    )
    return fig


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------
st.set_page_config(page_title="Germany Electricity Flow Network", layout="wide")
st.title("Germany · Cross-border physical electricity flows")
st.caption(
    "Fraunhofer-inspired radial network. Germany's borders are highlighted; "
    "connections among neighbouring countries are shown in greyscale."
)

with st.sidebar:
    st.header("Historical period")
    year = st.selectbox("Year", list(range(2020, date.today().year + 1)), index=list(range(2020, date.today().year + 1)).index(YEAR_DEFAULT) if YEAR_DEFAULT in range(2020, date.today().year + 1) else len(list(range(2020, date.today().year + 1)))-1)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    st.divider()
    st.subheader("Display")
    show_labels = st.checkbox("Show country codes", value=True)
    daily_aggregation = st.selectbox("Time aggregation", ["Daily mean", "Hourly mean"], index=0)
    st.caption("The default daily mean keeps a full-year animation responsive.")

try:
    with st.spinner(f"Loading CBPF history for {year}…"):
        raw_df, metadata = load_historical_flows(start, end)
except Exception as exc:
    st.error(f"Could not load the all-country CBPF dataset: {exc}")
    st.markdown(
        "The app uses the Energy-Charts v2 CBPF endpoint with `country=all`. "
        "If the endpoint's series IDs do not identify country pairs in your response, "
        "the parser intentionally stops rather than assigning incorrect border flows."
    )
    st.stop()

if daily_aggregation == "Daily mean":
    frame_df = daily_mean_flows(raw_df)
else:
    tmp = raw_df.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.floor("h")
    frame_df = tmp.groupby(["date", "a", "b"], as_index=False)["flow_gw"].mean()

available_dates = sorted(frame_df["date"].unique())
if not available_dates:
    st.warning("No observations were available for the selected year.")
    st.stop()

# Playback controls: Streamlit reruns on each interaction.
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
st.session_state.frame_index = min(st.session_state.frame_index, len(available_dates)-1)

c1, c2, c3 = st.columns([1.2, 3.5, 1.2])
with c1:
    if st.button("◀ Previous day", use_container_width=True):
        st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
with c3:
    if st.button("Next day ▶", use_container_width=True):
        st.session_state.frame_index = min(len(available_dates)-1, st.session_state.frame_index + 1)

with c2:
    selected_index = st.slider(
        "Historical playback",
        min_value=0,
        max_value=len(available_dates)-1,
        value=st.session_state.frame_index,
        format="%d",
    )
    st.session_state.frame_index = selected_index

selected_day = available_dates[st.session_state.frame_index]
selected_df = frame_df[frame_df["date"] == selected_day].copy()
max_flow = max(float(frame_df["flow_gw"].abs().quantile(0.98)), 0.1)

fig = make_network_figure(
    selected_df, selected_day, max_flow,
    show_labels=show_labels,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Date navigation and summary.
de_df = selected_df[(selected_df["a"] == "DE") | (selected_df["b"] == "DE")].copy()
if not de_df.empty:
    de_df["partner"] = de_df.apply(lambda r: r["b"] if r["a"] == "DE" else r["a"], axis=1)
    de_df["direction"] = np.where(de_df["flow_gw"] >= 0, "Import (+)", "Export (−)")
    de_df["magnitude_gw"] = de_df["flow_gw"].abs()
    de_df = de_df.sort_values("magnitude_gw", ascending=False)
    st.subheader(f"Germany's border flows · {selected_day}")
    st.dataframe(
        de_df[["partner", "direction", "flow_gw"]]
        .rename(columns={"partner": "Neighbour", "direction": "Direction", "flow_gw": "Signed flow (GW)"})
        .assign(Neighbour=lambda x: x["Neighbour"].map(COUNTRIES)),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Data and interpretation"):
    st.write(
        f"**Source:** Energy-Charts.info, Fraunhofer ISE · **Endpoint:** `/v2/cbpf`  \n"
        f"**Unit:** {metadata['unit']} · **Timezone:** {metadata['timezone']} · "
        f"**Resolution returned:** {metadata['resolution']}  \n"
        f"**Available:** {metadata['available_from']} to {metadata['available_until']}  \n"
        f"**Generated:** {metadata['generated_at']}"
    )
    st.write(
        "CBPF is a signed net physical flow. Positive values denote imports and "
        "negative values exports for the queried country. A single signed border "
        "series cannot show simultaneous gross import and gross export on the same "
        "interconnector. Ribbon width represents the absolute net flow."
    )
    st.write(
        "The API is licensed under CC BY 4.0 unless the response states otherwise. "
        "Please retain attribution to Energy-Charts.info / Fraunhofer ISE."
    )

st.caption("Data: Energy-Charts.info · Fraunhofer ISE · Cross-border physical flows (CBPF), CC BY 4.0")
