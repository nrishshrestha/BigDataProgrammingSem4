"""
SCOX Bus Performance Dashboard
Simple Streamlit UI for Big Data Programming – Clustering Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCOX Bus Performance Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths (change these if your folders are elsewhere) ───────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
FIGS = BASE / "Figures"

# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_csv(DATA / "clustering_features.csv")
    line_summary = pd.read_parquet(DATA / "line_level_summary.parquet")
    model_comp = pd.read_csv(DATA / "clustering_model_comparison.csv")
    cluster_profile = pd.read_csv(DATA / "cluster_profile_etl_context.csv")
    disruptions = pd.read_csv(DATA / "clean_disruptiondata.csv")
    return features, line_summary, model_comp, cluster_profile, disruptions


features, line_summary, model_comp, cluster_profile, disruptions = load_data()

# ── Simple on-the-fly clustering so we always have labels ────────────────────
@st.cache_data
def assign_clusters(df: pd.DataFrame):
    feat_cols = [
        "disruption_count", "avg_severity", "avg_duration_hours",
        "n_scheduled_trips", "avg_run_time_minutes",
        "n_avl_observations", "avg_delay_minutes", "avg_fare",
    ]
    X = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Rank clusters by mean disruption_count → Low / Moderate / High
    temp = df.copy()
    temp["cluster"] = labels
    order = temp.groupby("cluster")["disruption_count"].mean().sort_values().index.tolist()
    name_map = {
        order[0]: "Low Disruptions",
        order[1]: "Moderate Disruptions",
        order[2]: "High Disruptions",
    }
    temp["cluster_name"] = temp["cluster"].map(name_map)

    # PCA for 2-D view
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    temp["pca1"] = coords[:, 0]
    temp["pca2"] = coords[:, 1]
    return temp, name_map


clustered, name_map = assign_clusters(features)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🚌 SCOX Dashboard")
st.sidebar.markdown("**Stagecoach Oxfordshire**  \nBig Data Programming – Sem 4")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Line Performance", "Clustering", "Disruptions & Delays", "Model Comparison"],
)

selected_lines = st.sidebar.multiselect(
    "Filter by Line",
    options=sorted(features["line_code"].unique()),
    default=[],
)

week_range = st.sidebar.slider(
    "Week range",
    int(features["week_number"].min()),
    int(features["week_number"].max()),
    (int(features["week_number"].min()), int(features["week_number"].max())),
)

# Apply filters
mask = (
    (clustered["week_number"] >= week_range[0])
    & (clustered["week_number"] <= week_range[1])
)
if selected_lines:
    mask &= clustered["line_code"].isin(selected_lines)
view = clustered[mask]

# ── Helper ───────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("SCOX Bus Network – Performance Overview")
    st.caption("Aggregated line-level insights from Timetable · AVL · Fares · Disruption data")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Lines", f"{line_summary['line_code'].nunique()}")
    with c2:
        kpi_card("Scheduled Trips", f"{int(line_summary['n_scheduled_trips'].sum()):,}")
    with c3:
        kpi_card("Total Disruptions", f"{int(line_summary['n_disruptions'].sum())}")
    with c4:
        kpi_card("Avg Delay (min)", f"{line_summary['avg_delay_minutes'].mean():.1f}")
    with c5:
        kpi_card("Avg Fare (£)", f"{line_summary['avg_fare'].mean():.2f}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Trips vs Disruptions (by Line)")
        fig = px.scatter(
            line_summary,
            x="n_scheduled_trips",
            y="n_disruptions",
            size="avg_delay_minutes",
            color="avg_fare",
            hover_name="line_code",
            color_continuous_scale="Viridis",
            labels={
                "n_scheduled_trips": "Scheduled Trips",
                "n_disruptions": "Disruption Count",
                "avg_delay_minutes": "Avg Delay (min)",
                "avg_fare": "Avg Fare (£)",
            },
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Top 10 Lines by Disruption Count")
        top = line_summary.nlargest(10, "n_disruptions")
        fig2 = px.bar(
            top,
            x="line_code",
            y="n_disruptions",
            color="avg_delay_minutes",
            color_continuous_scale="Reds",
            labels={"n_disruptions": "Disruptions", "line_code": "Line"},
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Line-level Summary Table")
    st.dataframe(
        line_summary.sort_values("n_disruptions", ascending=False).style.format({
            "avg_run_time_minutes": "{:.2f}",
            "avg_delay_minutes": "{:.2f}",
            "avg_fare": "£{:.2f}",
        }),
        use_container_width=True,
        height=360,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Line Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Line Performance":
    st.title("Line Performance Explorer")

    line = st.selectbox("Select a line", sorted(features["line_code"].unique()))
    line_data = features[features["line_code"] == line].sort_values("week_number")
    line_info = line_summary[line_summary["line_code"] == line].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scheduled Trips", int(line_info["n_scheduled_trips"]))
    m2.metric("Avg Run Time (min)", f"{line_info['avg_run_time_minutes']:.2f}")
    m3.metric("Avg Delay (min)", f"{line_info['avg_delay_minutes']:.1f}")
    m4.metric("Total Disruptions", int(line_info["n_disruptions"]))

    st.subheader(f"Weekly Disruption & Delay Trend – Line {line}")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=line_data["week_number"],
        y=line_data["disruption_count"],
        name="Disruptions",
        marker_color="#ef4444",
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=line_data["week_number"],
        y=line_data["avg_delay_minutes"],
        name="Avg Delay (min)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3),
    ))
    fig.update_layout(
        yaxis=dict(title="Disruption Count"),
        yaxis2=dict(title="Avg Delay (min)", overlaying="y", side="right"),
        height=400,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Lines – Delay Distribution")
    fig_box = px.box(
        line_summary,
        y="avg_delay_minutes",
        points="all",
        hover_data=["line_code"],
        labels={"avg_delay_minutes": "Average Delay (minutes)"},
    )
    fig_box.update_layout(height=350)
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Clustering
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title("Disruption Clustering Results")
    st.markdown(
        "Lines × Weeks are clustered into **Low / Moderate / High Disruption** groups "
        "using K-Means (k=3) on standardised features."
    )

    c1, c2, c3 = st.columns(3)
    counts = view["cluster_name"].value_counts()
    c1.metric("Low Disruptions", int(counts.get("Low Disruptions", 0)))
    c2.metric("Moderate Disruptions", int(counts.get("Moderate Disruptions", 0)))
    c3.metric("High Disruptions", int(counts.get("High Disruptions", 0)))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA Projection of Clusters")
        fig_pca = px.scatter(
            view,
            x="pca1",
            y="pca2",
            color="cluster_name",
            hover_data=["line_code", "week_number", "disruption_count", "avg_delay_minutes"],
            color_discrete_map={
                "Low Disruptions": "#22c55e",
                "Moderate Disruptions": "#f59e0b",
                "High Disruptions": "#ef4444",
            },
            labels={"pca1": "PCA Component 1", "pca2": "PCA Component 2"},
        )
        fig_pca.update_layout(height=450)
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.subheader("Cluster Profiles (mean features)")
        profile = (
            view.groupby("cluster_name")[
                ["n_scheduled_trips", "avg_run_time_minutes",
                 "n_avl_observations", "avg_delay_minutes",
                 "avg_fare", "disruption_count"]
            ]
            .mean()
            .round(2)
            .reset_index()
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)

        st.markdown("**Original project cluster profile (GMM)**")
        st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    st.subheader("Cluster Assignment Table")
    show_cols = [
        "line_code", "week_number", "cluster_name",
        "disruption_count", "avg_delay_minutes",
        "n_scheduled_trips", "avg_fare",
    ]
    st.dataframe(
        view[show_cols].sort_values(["cluster_name", "disruption_count"], ascending=[True, False]),
        use_container_width=True,
        height=400,
    )

    st.subheader("Project-generated Figures")
    fc1, fc2 = st.columns(2)
    with fc1:
        if (FIGS / "clustering_pca_comparison.png").exists():
            st.image(str(FIGS / "clustering_pca_comparison.png"), caption="PCA Comparison")
    with fc2:
        if (FIGS / "clustering_silhouette_comparison.png").exists():
            st.image(str(FIGS / "clustering_silhouette_comparison.png"), caption="Silhouette Comparison")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Disruptions & Delays
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Disruptions & Delays":
    st.title("Disruptions & Delay Analysis")

    if "Severity" in disruptions.columns:
        st.subheader("Disruption Severity Distribution")
        sev = disruptions["Severity"].value_counts().reset_index()
        sev.columns = ["Severity", "Count"]
        fig_sev = px.pie(sev, names="Severity", values="Count", hole=0.4)
        st.plotly_chart(fig_sev, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weekly Disruption Volume (all lines)")
        weekly = (
            view.groupby("week_number")["disruption_count"]
            .sum()
            .reset_index()
        )
        fig_w = px.area(
            weekly,
            x="week_number",
            y="disruption_count",
            labels={"week_number": "ISO Week", "disruption_count": "Total Disruptions"},
        )
        fig_w.update_layout(height=350)
        st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        st.subheader("Avg Delay vs Disruption Count")
        fig_sc = px.scatter(
            view,
            x="disruption_count",
            y="avg_delay_minutes",
            color="cluster_name",
            size="n_scheduled_trips",
            hover_data=["line_code", "week_number"],
            color_discrete_map={
                "Low Disruptions": "#22c55e",
                "Moderate Disruptions": "#f59e0b",
                "High Disruptions": "#ef4444",
            },
        )
        fig_sc.update_layout(height=350)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Exploratory Figures from the Project")
    eds = list(FIGS.glob("eda_*.png"))
    if eds:
        cols = st.columns(min(3, len(eds)))
        for i, p in enumerate(eds):
            with cols[i % 3]:
                st.image(str(p), caption=p.stem.replace("eda_", "").replace("_", " ").title())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Clustering Model Comparison")
    st.markdown(
        "Comparison of K-Means, DBSCAN and Gaussian Mixture Model "
        "as evaluated in the project notebook."
    )

    st.dataframe(model_comp.style.format({"Silhouette": "{:.3f}"}), use_container_width=True)

    fig = px.bar(
        model_comp,
        x="Model",
        y="Silhouette",
        color="Type",
        text="Silhouette",
        labels={"Silhouette": "Silhouette Score"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, yaxis_range=[0, 0.8])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Interpretation:** Higher silhouette indicates better-separated clusters. "
        "In the project, GMM achieved the highest score (~0.56)."
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: BODS / Stagecoach Oxfordshire (SCOX)\n"
    "Pipeline: PySpark ETL → Feature Engineering → Clustering\n"
    "Dashboard: Streamlit"
)