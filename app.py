
import io
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import pydeck as pdk


st.set_page_config(page_title="Rail Path Solver (OEM Lines → FRAARCID Chain)", layout="wide")
st.title("Rail Path Solver (OEM Lines → FRAARCID Chain)")


# ---------- Data loading helpers ----------

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("__", "_")
    )
    return df


# ---------- FRA graph + nodes ----------

@st.cache_data(show_spinner=False)
def build_fra_graph_and_nodes(fra_df: pd.DataFrame):
    \"\"\"
    Build:
    - Undirected graph with edge weights = MILES
    - nodes_df: unique FRANODE with lat/long
    - edge_lookup: (min(u,v), max(u,v)) -> list of FRAARCID(s)
    \"\"\"
    # Required core columns (after cleaning underscores)
    required = {
        "FRAARCID",
        "FRFRANODE",
        "TOFRANODE",
        "MILES",
        "Start_Node__Lat",
        "Start_Node__Long",
        "End_Node_Lat",
        "End_Node_Long",
    }
    missing = required - set(fra_df.columns)
    if missing:
        raise ValueError(f"Missing required FRA columns after cleaning: {missing}")

    fra_df = fra_df.copy()
    fra_df["MILES"] = pd.to_numeric(fra_df["MILES"], errors="coerce")
    fra_df["Start_Node__Lat"] = pd.to_numeric(fra_df["Start_Node__Lat"], errors="coerce")
    fra_df["Start_Node__Long"] = pd.to_numeric(fra_df["Start_Node__Long"], errors="coerce")
    fra_df["End_Node_Lat"] = pd.to_numeric(fra_df["End_Node_Lat"], errors="coerce")
    fra_df["End_Node_Long"] = pd.to_numeric(fra_df["End_Node_Long"], errors="coerce")

    # Build node table
    start_nodes = fra_df[["FRFRANODE", "Start_Node__Lat", "Start_Node__Long"]].rename(
        columns={
            "FRFRANODE": "FRANODE",
            "Start_Node__Lat": "lat",
            "Start_Node__Long": "lon",
        }
    )
    end_nodes = fra_df[["TOFRANODE", "End_Node_Lat", "End_Node_Long"]].rename(
        columns={
            "TOFRANODE": "FRANODE",
            "End_Node_Lat": "lat",
            "End_Node_Long": "lon",
        }
    )

    nodes_df = pd.concat([start_nodes, end_nodes], ignore_index=True)
    nodes_df = nodes_df.dropna(subset=["lat", "lon"])
    nodes_df = nodes_df.drop_duplicates(subset=["FRANODE"]).reset_index(drop=True)

    G = nx.Graph()
    edge_lookup = defaultdict(list)

    for row in fra_df.itertuples(index=False):
        u = getattr(row, "FRFRANODE")
        v = getattr(row, "TOFRANODE")
        miles = getattr(row, "MILES")
        if pd.isna(miles):
            continue
        miles = float(miles)
        fraarcid = getattr(row, "FRAARCID")

        G.add_edge(u, v, weight=miles)
        key = tuple(sorted((u, v)))
        edge_lookup[key].append(fraarcid)

    return G, nodes_df, edge_lookup


# ---------- Nearest node snapping ----------

@st.cache_data(show_spinner=False)
def build_node_coord_arrays(nodes_df: pd.DataFrame):
    return (
        nodes_df["FRANODE"].to_numpy(),
        nodes_df["lat"].to_numpy(),
        nodes_df["lon"].to_numpy(),
    )


def find_nearest_node(nodes_ids, nodes_lat, nodes_lon, q_lat, q_lon):
    d2 = (nodes_lat - q_lat) ** 2 + (nodes_lon - q_lon) ** 2
    idx = int(np.argmin(d2))
    return nodes_ids[idx]


def snap_oem_lines_to_nodes(oem_df: pd.DataFrame, nodes_df: pd.DataFrame):
    required = {
        "OEM_LINE_START_LAT",
        "OEM_LINE_START_LONG",
        "OEM_LINE_END_LAT",
        "OEM_LINE_END_LONG",
    }
    missing = required - set(oem_df.columns)
    if missing:
        raise ValueError(f"OEM file is missing required columns: {missing}")

    oem = oem_df.copy()
    oem["OEM_LINE_START_LAT"] = pd.to_numeric(oem["OEM_LINE_START_LAT"], errors="coerce")
    oem["OEM_LINE_START_LONG"] = pd.to_numeric(oem["OEM_LINE_START_LONG"], errors="coerce")
    oem["OEM_LINE_END_LAT"] = pd.to_numeric(oem["OEM_LINE_END_LAT"], errors="coerce")
    oem["OEM_LINE_END_LONG"] = pd.to_numeric(oem["OEM_LINE_END_LONG"], errors="coerce")

    nodes_ids, nodes_lat, nodes_lon = build_node_coord_arrays(nodes_df)

    start_nodes = []
    end_nodes = []

    for row in oem.itertuples(index=False):
        s_lat = getattr(row, "OEM_LINE_START_LAT")
        s_lon = getattr(row, "OEM_LINE_START_LONG")
        e_lat = getattr(row, "OEM_LINE_END_LAT")
        e_lon = getattr(row, "OEM_LINE_END_LONG")

        if any(pd.isna(x) for x in [s_lat, s_lon, e_lat, e_lon]):
            start_nodes.append(None)
            end_nodes.append(None)
            continue

        s_node = find_nearest_node(nodes_ids, nodes_lat, nodes_lon, s_lat, s_lon)
        e_node = find_nearest_node(nodes_ids, nodes_lat, nodes_lon, e_lat, e_lon)

        start_nodes.append(s_node)
        end_nodes.append(e_node)

    oem["start_node"] = start_nodes
    oem["end_node"] = end_nodes
    return oem


# ---------- Shortest path solving ----------

def solve_one_pair_nodes(G, edge_lookup, start_node, end_node):
    if start_node is None or end_node is None:
        return {
            "status": "error",
            "message": "Missing snapped start or end node",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    if (start_node not in G) or (end_node not in G):
        return {
            "status": "error",
            "message": "Start or end node not present in graph",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    try:
        path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
        dist = nx.path_weight(G, path_nodes, weight="weight")
    except nx.NetworkXNoPath:
        return {
            "status": "error",
            "message": "No path between snapped start and end nodes",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    fraarcids = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        key = tuple(sorted((u, v)))
        arcids = edge_lookup.get(key)
        if not arcids:
            fraarcids.append("MISSING")
        else:
            fraarcids.append(arcids[0])

    return {
        "status": "ok",
        "message": "",
        "rail_miles": round(dist, 3),
        "num_segments": len(fraarcids),
        "fraarcid_chain": " > ".join(fraarcids),
        "frnode_chain": " > ".join(path_nodes),
    }


def solve_oem_lines(G, edge_lookup, oem_df_with_nodes: pd.DataFrame):
    results = []
    for row in oem_df_with_nodes.itertuples(index=False):
        start_node = getattr(row, "start_node")
        end_node = getattr(row, "end_node")
        res = solve_one_pair_nodes(G, edge_lookup, start_node, end_node)
        base_row = row._asdict()
        base_row.update(res)
        results.append(base_row)
    return pd.DataFrame(results)


# ---------- UI ----------

st.sidebar.header("Step 1: Upload data")

oem_file = st.sidebar.file_uploader(
    "Upload OEM line file (start/end coords)",
    type=["csv"],
    help="Must contain OEM_LINE_START_LAT, OEM_LINE_START_LONG, OEM_LINE_END_LAT, OEM_LINE_END_LONG",
)
fra_file = st.sidebar.file_uploader(
    "Upload FRA master segments file",
    type=["csv"],
    help="Must contain FRAARCID, FRFRANODE, TOFRANODE, MILES, Start Node _Lat, Start Node _Long, End Node_Lat, End Node_Long",
)

if oem_file is not None and fra_file is not None:
    try:
        oem_df_raw = load_csv(oem_file)
        fra_df_raw = load_csv(fra_file)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    st.success(f"Loaded OEM file with {len(oem_df_raw):,} lines.")
    st.success(f"Loaded FRA master file with {len(fra_df_raw):,} segments.")

    # Build FRA graph + nodes
    try:
        G, nodes_df, edge_lookup = build_fra_graph_and_nodes(fra_df_raw)
    except Exception as e:
        st.error(f"Error building FRA graph/nodes: {e}")
        st.stop()

    st.write(f"FRA graph has **{G.number_of_nodes():,} nodes** and **{G.number_of_edges():,} edges**.")

    # Snap OEM lines to nearest FRA nodes
    try:
        oem_with_nodes = snap_oem_lines_to_nodes(oem_df_raw, nodes_df)
    except Exception as e:
        st.error(f"Error snapping OEM lines to FRA nodes: {e}")
        st.stop()

    st.subheader("Snapped OEM lines (preview)")
    st.dataframe(oem_with_nodes.head(20), use_container_width=True)

    if st.button("Run shortest path solver for all OEM lines"):
        with st.spinner("Computing shortest rail paths for OEM lines..."):
            results_df = solve_oem_lines(G, edge_lookup, oem_with_nodes)

        st.subheader("Results")
        st.dataframe(results_df, use_container_width=True)

        # Cache for mapping & export
        st.session_state["results_df"] = results_df
        st.session_state["fra_df"] = fra_df_raw

        # Full results download
        csv_buf_full = io.StringIO()
        results_df.to_csv(csv_buf_full, index=False)
        st.download_button(
            label="Download full results (OEM + snapped nodes + path metrics) as CSV",
            data=csv_buf_full.getvalue(),
            file_name="oem_rail_paths_full.csv",
            mime="text/csv",
        )

        # Minimal OD/path export
        export_cols = []
        for col in [
            "OEM_LINE_START_LAT",
            "OEM_LINE_START_LONG",
            "OEM_LINE_END_LAT",
            "OEM_LINE_END_LONG",
            "start_node",
            "end_node",
            "rail_miles",
            "num_segments",
            "fraarcid_chain",
            "frnode_chain",
            "status",
            "message",
        ]:
            if col in results_df.columns and col not in export_cols:
                export_cols.append(col)

        od_export_df = results_df[export_cols].copy()
        csv_buf_export = io.StringIO()
        od_export_df.to_csv(csv_buf_export, index=False)
        st.download_button(
            label="Download OEM line distance + FRA chain export CSV",
            data=csv_buf_export.getvalue(),
            file_name="oem_rail_paths_export.csv",
            mime="text/csv",
        )

    # Mapping section
    st.subheader("Map a solved OEM line")
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        fra_df_cached = st.session_state["fra_df"]

        if len(results_df) > 0:
            def make_label(idx):
                row = results_df.loc[idx]
                s_lat = row.get("OEM_LINE_START_LAT", "")
                s_lon = row.get("OEM_LINE_START_LONG", "")
                e_lat = row.get("OEM_LINE_END_LAT", "")
                e_lon = row.get("OEM_LINE_END_LONG", "")
                miles = row.get("rail_miles", "")
                miles_str = f"{miles} mi" if pd.notna(miles) else ""
                status = row.get("status", "")
                return f"{idx}: ({s_lat}, {s_lon}) → ({e_lat}, {e_lon}) [{status}] {miles_str}"

            selected_idx = st.selectbox(
                "Select an OEM line to map",
                options=results_df.index.tolist(),
                format_func=make_label,
            )

            row = results_df.loc[selected_idx]
            if row.get("status") != "ok" or not isinstance(row.get("fraarcid_chain"), str):
                st.warning("Selected OEM line does not have a valid solved path.")
            else:
                chain_str = row["fraarcid_chain"]
                fra_ids = [c.strip() for c in chain_str.split(">") if c.strip() and c.strip() != "MISSING"]

                fra_df = fra_df_cached.copy()
                fra_df.columns = (
                    fra_df.columns.str.strip()
                    .str.replace(" ", "_")
                    .str.replace("__", "_")
                )

                fra_segments = fra_df[fra_df["FRAARCID"].isin(fra_ids)].copy()

                if fra_segments.empty:
                    st.warning("Could not find FRA segments for this chain in the FRA file.")
                else:
                    for col in ["Start_Node__Lat", "Start_Node__Long", "End_Node_Lat", "End_Node_Long"]:
                        if col in fra_segments.columns:
                            fra_segments[col] = pd.to_numeric(fra_segments[col], errors="coerce")

                    line_data = []
                    for seg in fra_segments.itertuples(index=False):
                        line_data.append({
                            "fraarcid": getattr(seg, "FRAARCID"),
                            "from_lng": getattr(seg, "Start_Node__Long"),
                            "from_lat": getattr(seg, "Start_Node__Lat"),
                            "to_lng": getattr(seg, "End_Node_Long"),
                            "to_lat": getattr(seg, "End_Node_Lat"),
                        })

                    line_df = pd.DataFrame(line_data).dropna()

                    if line_df.empty:
                        st.warning("No valid lat/long data to map for this path.")
                    else:
                        center_lat = (line_df["from_lat"].mean() + line_df["to_lat"].mean()) / 2
                        center_lng = (line_df["from_lng"].mean() + line_df["to_lng"].mean()) / 2

                        layer = pdk.Layer(
                            "LineLayer",
                            data=line_df,
                            get_source_position=["from_lng", "from_lat"],
                            get_target_position=["to_lng", "to_lat"],
                            get_color=[255, 0, 0],
                            get_width=6,
                            pickable=True,
                        )

                        view_state = pdk.ViewState(
                            latitude=center_lat,
                            longitude=center_lng,
                            zoom=6,
                            bearing=0,
                            pitch=0,
                        )

                        st.pydeck_chart(
                            pdk.Deck(
                                layers=[layer],
                                initial_view_state=view_state,
                                map_style="light",
                                tooltip={"text": "{fraarcid}"},
                            )
                        )
        else:
            st.info("Run the solver to see OEM lines you can map.")
    else:
        st.info("Run the solver to enable mapping.")
else:
    st.info("Upload both the OEM line file and the FRA master segments file to begin.")
