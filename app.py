import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# ── Paths & assets ───────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.extend([str(APP_DIR), str(PROJECT_ROOT)])  # allow imports from ./ and ../

CANDIDATE_ASSETS = [APP_DIR / "assets", PROJECT_ROOT / "assets"]
ASSETS_DIR = next((p for p in CANDIDATE_ASSETS if p.exists()), APP_DIR / "assets")

def find_gps_root() -> Path | None:
    env = os.environ.get("GPS_ROOT")
    if env and Path(env).exists():
        return Path(env).resolve()
    for p in [
        PROJECT_ROOT / "GPS_Data", PROJECT_ROOT / "GPS_data",
        APP_DIR / "GPS_Data", APP_DIR / "GPS_data",
    ]:
        if p.exists():
            return p.resolve()
    return None

GPS_ROOT = find_gps_root()

# ── Try to use repo Navbar/Footer (with .render()) ───────────────────────────
try:
    from layout.navbar import Navbar  # expects Navbar(children).render()
except Exception:
    Navbar = None

try:
    from layout.footer import Footer  # expects Footer().render()
except Exception:
    Footer = None

# ── Helpers (ported) ─────────────────────────────────────────────────────────
# Compact breakdown: "Split\nSR {stroke}\n{speed} m/s" per 250m column
DISTANCES = ["250m","500m","750m","1000m","1250m","1500m","1750m","2000m"]

def make_compact_breakdown(table_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty:
        return table_df
    out = table_df[["Country, Lane", "Rank"]].copy()
    for d in DISTANCES:
        split_col  = f"{d} Split"
        stroke_col = f"{d} Stroke"
        speed_col  = f"{d} Speed"
        out[d] = table_df.apply(
            lambda r: f"{r[split_col]}\nSR {r[stroke_col]}\n{r[speed_col]} m/s",
            axis=1
        )
    return out


def convert_seconds_to_time(seconds: float) -> str:
    m = int(seconds // 60); s = int(seconds % 60); ms = int((seconds - int(seconds)) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"

def time_to_seconds(time_str: str) -> float:
    minutes, seconds = time_str.split(":"); return float(minutes) * 60 + float(seconds)

def rename_duplicate_columns(columns):
    counts, out = {}, []
    for c in columns:
        if c in counts:
            counts[c] += 1; out.append(f"{c}{counts[c]}")
        else:
            counts[c] = 0; out.append(c)
    return out

# ── Data access ──────────────────────────────────────────────────────────────
def list_events() -> list[str]:
    if not GPS_ROOT or not GPS_ROOT.exists(): return []
    return sorted([p.name for p in GPS_ROOT.iterdir() if p.is_dir()])

def build_race_info(event: str) -> list[dict]:
    if not GPS_ROOT: return []
    race_list = glob.glob(str(GPS_ROOT / event / "**" / "*.csv"))
    info = []
    for file in race_list:
        file_name = Path(file).stem
        parts = file_name.split("_")
        phase = parts[-1] if parts else "Unknown"
        if "2025" in event:
            display = file_name; b_class = parts[3] if len(parts) > 3 else "Unknown"
        else:
            display = parts[-1]; b_class = parts[0] if len(parts) > 0 else "Unknown"
        info.append({"display": display, "file_path": file, "b_class": b_class, "Phase": phase})
    uniq, seen = [], set()
    for r in sorted(info, key=lambda r: r["display"]):
        if r["display"] not in seen: uniq.append(r); seen.add(r["display"])
    return uniq

def load_selected_races(selected_displays: list[str], race_info: list[dict]) -> pd.DataFrame:
    frames = []
    for i, disp in enumerate(selected_displays):
        r = next((x for x in race_info if x["display"] == disp), None)
        if not r: continue
        df = pd.read_csv(r["file_path"], delimiter=";")
        df.columns = [f"{c}_{i+1}" for c in df.columns]
        frames.append(df)
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    distance_cols = [c for c in df.columns if c.startswith("Distance")]
    if distance_cols:
        keep = distance_cols[0]
        df.drop(columns=[c for c in distance_cols if c != keep], inplace=True, errors="ignore")
        df.rename(columns={keep: "Distance"}, inplace=True)
    return df

# ── Metrics & figures ────────────────────────────────────────────────────────
def build_outputs(df: pd.DataFrame):
    if df.empty or "Distance" not in df.columns: return None

    palette = px.colors.qualitative.Plotly
    speed_cols  = [c for c in df.columns if c.startswith("Speed")]
    stroke_cols = [c for c in df.columns if c.startswith("Stroke")]
    name_cols   = [c for c in df.columns if c.startswith("ShortName")]
    split_cols  = [c for c in df.columns if c.startswith("Time")]

    country_list, lane_list, _ = [], [], []
    for col in name_cols:
        lane = col.split("_")[0][-1]
        lane_list.append(lane)
        country = df[col].iloc[0]
        country_list.append(f"{country}, {lane}")

    times = df[split_cols].iloc[-1, :] if len(split_cols) else pd.Series(dtype=float)
    times_in_sec = [(label, time_to_seconds(t)) for label, t in times.items()]
    sorted_times = sorted(times_in_sec, key=lambda x: x[1])
    rank_map = {label: (rk + 1) for rk, (label, _) in enumerate(sorted_times)}
    ranks = [rank_map.get(lbl, None) for lbl in times.keys()]

    # Velocity
    vel_fig = go.Figure()
    for i, col in enumerate(speed_cols):
        if df[col].mean() <= 0.5: continue
        line = dict(color="red", width=4, dash="dash") if "CAN" in country_list[i] else dict(color=palette[i % len(palette)])
        vel_fig.add_trace(go.Scatter(x=df["Distance"], y=savgol_filter(df[col], 30, 2),
                                     mode="lines", line=line, name=country_list[i]))
    vel_fig.update_layout(title="Boat Velocity Vs. Distance", xaxis_title="Distance (m)",
                          yaxis_title="Velocity (m/s)", margin=dict(l=40,r=20,t=60,b=40),)
                          #legend=dict(orientation="v", yanchor="bottom", y=1.02, x=0))

    # Stroke rate
    stroke_fig = go.Figure()
    for i, col in enumerate(stroke_cols):
        try:
            series = df[col]; series = series[series > 20]
            line = dict(color="red", width=4, dash="dash") if "CAN" in country_list[i] else dict(color=palette[i % len(palette)])
            stroke_fig.add_trace(go.Scatter(x=df.loc[series.index, "Distance"], y=savgol_filter(series, 30, 2),
                                            mode="lines", line=line, name=country_list[i]))
        except Exception:
            pass
    stroke_fig.update_layout(title="Boat Stroke Rate Vs. Distance", xaxis_title="Distance (m)",
                             yaxis_title="Stroke Rate (SPM)", margin=dict(l=40,r=20,t=60,b=40),)
                             #legend=dict(orientation="v", yanchor="bottom", y=1.02, x=0))

    # Section indices
    dist_vals = df["Distance"].values
    def idx_of(d): return int(np.where(dist_vals == d)[0][0])
    try:
        i250,i500,i750,i1000,i1250,i1500,i1750,i2000 = map(idx_of, (250,500,750,1000,1250,1500,1750,2000))
    except Exception:
        return {"vel_fig": vel_fig, "stroke_fig": stroke_fig,
                "splits_plot": go.Figure(),
                "table_df": pd.DataFrame(),
                "timing_df": pd.DataFrame({"Note": ["Distances (250..2000) not all present"]})}

    v_sections = [
        df[speed_cols][:i250].mean(),
        df[speed_cols][i250:i500].mean(),
        df[speed_cols][i500:i750].mean(),
        df[speed_cols][i750:i1000].mean(),
        df[speed_cols][i1000:i1250].mean(),
        df[speed_cols][i1250:i1500].mean(),
        df[speed_cols][i1500:i1750].mean(),
        df[speed_cols][i1750:i2000].mean(),
    ]
    sr_sections = [
        df[stroke_cols][:i250].mean(),
        df[stroke_cols][i250:i500].mean(),
        df[stroke_cols][i500:i750].mean(),
        df[stroke_cols][i750:i1000].mean(),
        df[stroke_cols][i1000:i1250].mean(),
        df[stroke_cols][i1250:i1500].mean(),
        df[stroke_cols][i1500:i1750].mean(),
        df[stroke_cols][i1750:i2000].mean(),
    ]

    final_times = [convert_seconds_to_time(float(sec)) for _, sec in times_in_sec]
    timing_df = pd.DataFrame({
        "Country": [c.split(",")[0] for c in country_list],
        "Lane":    [c.split(",")[-1].strip() for c in country_list],
        "Race Time": final_times
    })

    # Breakdown table
    data = {"Country, Lane": [], "Rank": []}
    names = ["250m","500m","750m","1000m","1250m","1500m","1750m","2000m"]
    for nm in names:
        data[f"{nm} Split"] = []; data[f"{nm} Stroke"] = []; data[f"{nm} Speed"] = []
    for i in range(len(v_sections[0])):
        try:
            data["Country, Lane"].append(country_list[i]); data["Rank"].append(ranks[i])
            for nm, vs in zip(names, v_sections):
                data[f"{nm} Split"].append(convert_seconds_to_time(500.0 / float(vs[i])))
                data[f"{nm} Speed"].append(round(float(vs[i]), 2))
            for nm, srs in zip(names, sr_sections):
                data[f"{nm} Stroke"].append(round(float(srs[i]), 2))
        except Exception:
            pass

    table_df_unsorted = pd.DataFrame(data)
    table_df = table_df_unsorted.sort_values(by="Rank", na_position="last").reset_index(drop=True)

    
    # Splits plot
    transposed = table_df_unsorted.T
    transposed.columns = transposed.iloc[0, :]
    transposed = transposed.iloc[2:, :]
    transposed.columns = rename_duplicate_columns(transposed.columns)
    transposed_splits = transposed[transposed.index.str.contains('Split')]



    splits_plot = go.Figure()
    x_vals = [250,500,750,1000,1250,1500,1750,2000]
    for i, col in enumerate(transposed_splits.columns):
        line = dict(color="red", width=4, dash="dash") if "CAN" in col else dict(color=palette[i % len(palette)])
        print(transposed_splits[col])
        try:
           
            splits_plot.add_trace(go.Scatter(x=x_vals, y=list(pd.to_datetime(transposed_splits[col])), 
                                                mode="lines+markers", 
                                                line=line, 
                                                name=col,
                                                hovertemplate="%{y:.2f}s at %{x}m<extra></extra>"))
        except Exception:
                pass
    splits_plot.update_layout(title="Race Split Vs. Distance", xaxis_title="Distance (m)",
                              yaxis=dict(title="Time for 500m (s)", autorange="reversed"),
                              margin=dict(l=40,r=20,t=60,b=40),)
                              #legend=dict(orientation="v", yanchor="bottom", y=1.02, x=0))

    return {"vel_fig": vel_fig, "stroke_fig": stroke_fig,
            "splits_plot": splits_plot, "table_df": table_df, "timing_df": timing_df}

# ── App init ─────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True
)
server = app.server

# Controls
controls = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Event"),
                dcc.Dropdown(id="event",
                             options=[{"label": e, "value": e} for e in list_events()],
                             placeholder="Choose an event", clearable=False),
            ], md=4),
            dbc.Col([
                dbc.Label("Select Race(s)"),
                dcc.Dropdown(id="races", options=[], multi=True, placeholder="Choose races"),
            ], md=8),
        ], className="g-3"),
        html.Hr(),
        dbc.Checklist(
            options=[{"label": "Show overall timing summary", "value": "lane_det"}],
            value=[],
            id="lane_det",
            switch=True
        ),
        html.Div(id="gps-warning", className="mt-2"),
    ]),
    className="mb-3 shadow-sm"
)

graphs_row = dbc.Row([
    dbc.Col(dcc.Graph(id="vel_fig"), md=7),
    dbc.Col(dcc.Graph(id="stroke_fig"), md=5),
], className="g-3")

splits_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id="splits_plot"),
        html.Hr(),
        html.H5("Race Split Breakdown", className="card-title"),
        dash_table.DataTable(
            id="breakdown_table",
            # data + columns come from the callback
            style_table={"overflowX": "auto", "minWidth": "90%"},
            style_header={
                "backgroundColor": "#6c757d",  # Bootstrap gray-700
                "color": "white",
                "fontWeight": "600",
                "border": "1px solid #6c757d",
            },
            style_cell={
                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                "fontSize": 14,
                "whiteSpace": "pre-line",     # <-- render \n as new lines
                "height": "auto",
                "lineHeight": "1.25rem",
                "padding": "8px 10px",
                "border": "1px solid #e9ecef",
            },
            style_cell_conditional=[
                {"if": {"column_id": "Country, Lane"}, "width": "120px", "minWidth": "220px"},
                {"if": {"column_id": "Rank"},          "width": "72px",  "textAlign": "center"},
                # distance columns: keep readable width
                *[{"if": {"column_id": d}, "width": "132px"} for d in DISTANCES],
            ],
            style_data_conditional=[
                # zebra striping
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                # highlight CAN row like your Plotly table
                {"if": {"filter_query": '{Country, Lane} contains "CAN"'}, "backgroundColor": "#f3d1db"},
            ],
            fixed_rows={"headers": True},
            sort_action="native",
            page_size=20,
            # allow simple markdown (not required for \n line breaks)
            markdown_options={"html": False},
        ),
    ]),
    className="mb-4 shadow-sm"
)
timing_card = dbc.Card(
    dbc.CardBody([
        html.H5("Timing Summary", className="card-title"),
        dash_table.DataTable(
            id="timing_table",
            style_table={"overflowX": "auto"},
            page_size=50
        )
    ]),
    className="mb-4 shadow-sm"
)

# Layout (no tabs)
app.layout = html.Div([
    dcc.Location(id="redirect-to", refresh=True),
    dcc.Interval(id="init-interval", interval=500, n_intervals=0, max_intervals=1),
    dcc.Interval(id="user-refresh", interval=60_000, n_intervals=0),

    (Navbar([html.Span(id="navbar-user", className="text-white-50 small", children="")]).render()
        if Navbar else
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    html.Img(src="/assets/img/csi-pacific-logo-reverse.png", height="40px"),
                    dbc.NavbarBrand("World Rowing GPS Analysis", className="ms-2"),
                    html.Span(id="navbar-user", className="text-white-50 small ms-3", children="")
                ], align="center", className="g-2"),
            ]),
            color="dark", dark=True, className="mb-3 rounded-3 shadow-sm"
        )
    ),

    dbc.Container([
        controls,
        graphs_row,
        splits_card,
        timing_card,
    ], fluid=True),

    (Footer().render() if Footer else html.Div())
])

# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("races", "options"),
    Output("races", "value"),
    Input("event", "value"),
)
def on_event_change(event):
    if not event:
        return [], []
    info = build_race_info(event)
    return [{"label": r["display"], "value": r["display"]} for r in info], []

@app.callback(
    Output("vel_fig", "figure"),
    Output("stroke_fig", "figure"),
    Output("splits_plot", "figure"),
    Output("breakdown_table", "data"),
    Output("breakdown_table", "columns"),
    Output("timing_table", "data"),
    Output("timing_table", "columns"),
    Output("navbar-user", "children"),
    Output("gps-warning", "children"),
    Input("event", "value"),
    Input("races", "value"),
    Input("lane_det", "value"),
    prevent_initial_call=False
)
def on_selection(event, races, lane_flags):
    gps_msg = None
    if not GPS_ROOT:
        gps_msg = dbc.Alert(
            "GPS data folder not found. Set GPS_ROOT or place it at ../GPS_Data (or ../GPS_data).",
            color="warning", className="mt-2"
        )
    if not event or not races:
        empty = go.Figure()
        return empty, empty, empty, [], [], [], [], "", gps_msg

    info = build_race_info(event)
    df = load_selected_races(races, info)
    out = build_outputs(df)
    if not out:
        empty = go.Figure()
        return empty, empty, empty, [], [], [], [], "", gps_msg

    #tbl_cols    = [{"name": c, "id": c} for c in out["table_df"].columns]
    compact_df = make_compact_breakdown(out["table_df"])
    tbl_cols = (
        [{"name": "Country, Lane", "id": "Country, Lane"},
         {"name": "Rank", "id": "Rank"}] +
        [{"name": d, "id": d, "presentation": "markdown"} for d in DISTANCES]
    )
    table_df = compact_df.to_dict("records")

    timing_cols = [{"name": c, "id": c} for c in out["timing_df"].columns]
    show_timing = ("lane_det" in (lane_flags or []))
    timing_data = out["timing_df"].to_dict("records") if show_timing else []
    timing_cols = timing_cols if show_timing else [{"name": "Timing summary hidden", "id": "Note"}]

    nav_text = f"{event} — {len(races)} race(s) selected"

    return (
        out["vel_fig"], out["stroke_fig"], out["splits_plot"],
        table_df, tbl_cols,
        timing_data, timing_cols,
        nav_text, gps_msg
    )

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ASSETS_DIR ->", ASSETS_DIR)
    print("GPS_ROOT   ->", GPS_ROOT)
    app.run_server(debug=True)
