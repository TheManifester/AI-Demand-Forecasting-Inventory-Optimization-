import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Inventory Optimization",
    page_icon="\U0001f4e6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR PALETTE (validated colorblind-safe from palette.md)
# ============================================================
CATEGORICAL = {
    "forecast": "#2a78d6",   # slot 1 blue
    "opening":   "#eb6834",  # slot 2 orange
    "inventory": "#1baf7a",  # slot 3 aqua
    "blue":      "#2a78d6",
    "violet":    "#4a3aa7",
    "red":       "#e34948",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

DATA_PATH = "data/processed/"

# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH + name)
    if df.index.name and df.index.name.lower() not in ("", "unnamed: 0"):
        df = df.reset_index()
    return df


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lower_name, real_name in cols_lower.items():
            if cand in lower_name:
                return real_name
    return None


def numeric_value_column(df: pd.DataFrame, exclude: list[str]) -> str | None:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return numeric_cols[-1] if numeric_cols else None


# ============================================================
# LOAD PROCESSED DATA (cached)
# ============================================================
inventory_by_store = load_csv("inventory_by_store.csv")
inventory_distribution = load_csv("inventory_distribution.csv")
inventory_simulation = load_csv("inventory_simulation.csv")
inventory_trend = load_csv("inventory_trend.csv")
kpis = load_csv("kpis.csv")
orders_by_store = load_csv("orders_by_store.csv")
orders_by_item = load_csv("orders_by_item.csv")
recommended_orders = load_csv("recommended_orders.csv")
top_store_item = load_csv("top_store_item.csv")

# Resolve column names once
store_col = find_column(inventory_simulation, ["store"])
item_col = find_column(inventory_simulation, ["item"])
inv_col = find_column(inventory_simulation, ["inventory_after_demand", "inventory"])
order_col = find_column(
    inventory_simulation,
    ["order_quantity", "order_decision_qty", "order_qty", "recommended_order", "order"],
)
date_col = find_column(inventory_simulation, ["date"])
pred_col = find_column(inventory_simulation, ["predicted_demand"])
opening_col = find_column(inventory_simulation, ["opening_inventory"])
inv_pos_col = find_column(inventory_simulation, ["inventory_position"])
rop_col = find_column(inventory_simulation, ["reorder_point"])

# Convert date column to datetime for filtering
if date_col:
    inventory_simulation[date_col] = pd.to_datetime(inventory_simulation[date_col])
    min_date = inventory_simulation[date_col].min().date()
    max_date = inventory_simulation[date_col].max().date()
else:
    min_date = None
    max_date = None

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("\U0001f50d Filters")

# Date range filter
if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = None

# Store filter (multiselect with "All" option)
if store_col:
    all_stores = sorted(inventory_simulation[store_col].dropna().unique())
    selected_stores = st.sidebar.multiselect(
        "Store(s)", options=["All"] + list(all_stores), default=["All"]
    )
    if "All" in selected_stores:
        selected_stores = list(all_stores)
else:
    selected_stores = []

# Item filter (multiselect with "All" option)
if item_col:
    all_items = sorted(inventory_simulation[item_col].dropna().unique())
    selected_items = st.sidebar.multiselect(
        "Item(s)", options=["All"] + list(all_items), default=["All"]
    )
    if "All" in selected_items:
        selected_items = list(all_items)
else:
    selected_items = []

# Reset button
if st.sidebar.button("Reset filters", use_container_width=True):
    st.rerun()

# ============================================================
# FILTER INVENTORY SIMULATION
# ============================================================
filtered = inventory_simulation.copy()

if selected_stores and store_col:
    filtered = filtered[filtered[store_col].isin(selected_stores)]

if selected_items and item_col:
    filtered = filtered[filtered[item_col].isin(selected_items)]

if date_col and start_date and end_date:
    filtered = filtered[
        (filtered[date_col].dt.date >= start_date) & (filtered[date_col].dt.date <= end_date)
    ]

# ============================================================
# KPI SECTION
# ============================================================
st.title("\U0001f4e6 AI Demand Forecasting & Inventory Optimization")
st.caption(f"Data: {min_date} → {max_date} | Store–Item granularity | Filters apply to all charts below")
st.divider()

st.subheader("\U0001f4ca Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if inv_col and len(filtered) > 0:
        total_units = filtered[inv_col].sum()
        st.metric(
            "Total Inventory (Units)",
            f"{total_units:,.0f}",
            help="Sum of inventory after demand across all selected stores/items/dates.",
        )
    else:
        st.metric("Total Inventory (Units)", "--")

with col2:
    if inv_col and len(filtered) > 0:
        avg_inv = filtered[inv_col].mean()
        st.metric(
            "Avg Inventory (Units)",
            f"{avg_inv:,.1f}",
            help="Average daily inventory level per store–item combination.",
        )
    else:
        st.metric("Avg Inventory (Units)", "--")

with col3:
    if inv_col and len(filtered) > 0:
        stockouts = (filtered[inv_col] <= 0).sum()
        stockout_pct = (stockouts / len(filtered)) * 100
        color = "inverse" if stockout_pct > 10 else "normal"
        st.metric(
            "Stock–out Risk",
            f"{stockout_pct:.1f}%",
            delta=f"{stockouts} days",
            delta_color=color,
            help="Percentage of store–item–days where inventory dropped to zero or below.",
        )
    else:
        st.metric("Stock–out Risk", "--")

with col4:
    if inv_pos_col and rop_col and len(filtered) > 0:
        service_days = (filtered[inv_pos_col] >= filtered[rop_col]).sum()
        service_pct = (service_days / len(filtered)) * 100
        color = "normal" if service_pct >= 90 else "inverse"
        st.metric(
            "Service Level",
            f"{service_pct:.1f}%",
            delta=f"{service_days}/{len(filtered)} days",
            delta_color=color,
            help="Percentage of days where inventory position met or exceeded the reorder point.",
        )
    else:
        st.metric("Service Level", "--")

# Expandable raw KPIs
with st.expander("View raw KPIs from kpis.csv"):
    st.dataframe(kpis, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# MAIN TABS
# ============================================================
tab_forecast, tab_stores, tab_recommend = st.tabs([
    "\U0001f4c8 Forecast & Trend",
    "\U0001f3ea Store & Orders",
    "\U0001f6d2 Recommendations",
])

# -----------------------------------------------------------
# TAB 1: FORECAST & TREND
# -----------------------------------------------------------
with tab_forecast:
    st.subheader("Demand Forecast vs. Inventory")

    if not filtered.empty and date_col and pred_col and inv_col:
        # Aggregate daily totals across selected stores/items
        daily = (
            filtered.groupby(date_col)
            .agg({pred_col: "sum", inv_col: "sum", opening_col: "sum" if opening_col else "first"})
            .reset_index()
            .sort_values(date_col)
        )

        fig = go.Figure()

        # Forecast line
        fig.add_trace(go.Scatter(
            x=daily[date_col],
            y=daily[pred_col],
            mode="lines",
            name="Forecasted Demand",
            line=dict(color=CATEGORICAL["forecast"], width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:,.0f} units<extra></extra>",
        ))

        # Inventory after demand line
        fig.add_trace(go.Scatter(
            x=daily[date_col],
            y=daily[inv_col],
            mode="lines",
            name="Inventory After Demand",
            line=dict(color=CATEGORICAL["inventory"], width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Inventory: %{y:,.0f} units<extra></extra>",
        ))

        fig.update_layout(
            title="Daily Forecasted Demand vs. Inventory After Demand",
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK_PRIMARY),
            xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            margin=dict(l=50, r=20, t=60, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Shows how the AI–predicted daily demand compares with the inventory remaining "
            "after demand for the selected store(s) and item(s)."
        )
    else:
        st.info("No data available for the selected filters.")

    st.divider()

    st.subheader("Total Inventory Trend")

    if not filtered.empty and date_col and inv_col:
        trend_data = (
            filtered.groupby(date_col)[inv_col]
            .sum()
            .reset_index()
            .sort_values(date_col)
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=trend_data[date_col],
            y=trend_data[inv_col],
            mode="lines",
            name="Total Inventory",
            line=dict(color=CATEGORICAL["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.1)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Total Inventory: %{y:,.0f} units<extra></extra>",
        ))

        fig2.update_layout(
            title="Aggregate Inventory Over Time",
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode="x unified",
            showlegend=False,
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK_PRIMARY),
            xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            margin=dict(l=50, r=20, t=60, b=50),
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Displays the aggregate inventory level across all selected stores/items, "
            "helping you see overall stock trends through the simulation period."
        )
    else:
        st.info("No inventory trend data available for the selected filters.")

# -----------------------------------------------------------
# TAB 2: STORE & ORDERS
# -----------------------------------------------------------
with tab_stores:
    col1, col2 = st.columns(2)

    # LEFT: Average Inventory by Store
    with col1:
        st.subheader("Average Inventory by Store")

        if store_col and inv_col:
            # Use inventory_simulation for accurate store labels (inventory_by_store.csv is missing store column)
            avg_by_store = (
                filtered.groupby(store_col)[inv_col]
                .mean()
                .reset_index()
                .sort_values(store_col)
            )

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=avg_by_store[store_col].astype(str),
                y=avg_by_store[inv_col],
                name="Avg Inventory",
                marker_color=CATEGORICAL["blue"],
                hovertemplate="Store: %{x}<br>Avg Inventory: %{y:,.1f} units<extra></extra>",
            ))

            fig3.update_layout(
                title="Mean Inventory Level per Store",
                xaxis_title="Store",
                yaxis_title="Avg Units",
                showlegend=False,
                plot_bgcolor=SURFACE,
                paper_bgcolor=SURFACE,
                font=dict(color=INK_PRIMARY),
                xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
                yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
                margin=dict(l=50, r=20, t=60, b=50),
            )

            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Shows the average daily inventory each store holds across the selected items and date range.")
        else:
            st.info("Store or inventory column not found.")

    # RIGHT: Orders by Store
    with col2:
        st.subheader("Orders by Store")

        store_col_os = find_column(orders_by_store, ["store"])
        units_col_os = find_column(orders_by_store, ["total_units", "units"])

        if store_col_os and units_col_os:
            df_os = orders_by_store.sort_values(store_col_os)

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=df_os[store_col_os].astype(str),
                y=df_os[units_col_os],
                name="Total Units Ordered",
                marker_color=CATEGORICAL["violet"],
                hovertemplate="Store: %{x}<br>Units Ordered: %{y:,.0f}<extra></extra>",
            ))

            fig4.update_layout(
                title="Total Units Ordered by Store",
                xaxis_title="Store",
                yaxis_title="Units Ordered",
                showlegend=False,
                plot_bgcolor=SURFACE,
                paper_bgcolor=SURFACE,
                font=dict(color=INK_PRIMARY),
                xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
                yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
                margin=dict(l=50, r=20, t=60, b=50),
            )

            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Total order volume (units) each store placed during the simulation period.")
        else:
            st.info("Orders by store data not available.")

    st.divider()

    # Orders by Item
    st.subheader("Orders by Item")

    item_col_oi = find_column(orders_by_item, ["item"])
    units_col_oi = find_column(orders_by_item, ["total_units", "units"])

    if item_col_oi and units_col_oi:
        df_oi = orders_by_item.sort_values(units_col_oi, ascending=False)

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=df_oi[item_col_oi].astype(str),
            y=df_oi[units_col_oi],
            name="Total Units Ordered",
            marker_color=CATEGORICAL["red"],
            hovertemplate="Item: %{x}<br>Units Ordered: %{y:,.0f}<extra></extra>",
        ))

        fig5.update_layout(
            title="Total Units Ordered by Item",
            xaxis_title="Item",
            yaxis_title="Units Ordered",
            showlegend=False,
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK_PRIMARY),
            xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            margin=dict(l=50, r=20, t=60, b=50),
        )

        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Reveals which items drive the most order volume across all stores.")
    else:
        st.info("Orders by item data not available.")

# -----------------------------------------------------------
# TAB 3: RECOMMENDATIONS
# -----------------------------------------------------------
with tab_recommend:
    st.subheader("Top Store–Item Combinations")

    if not top_store_item.empty:
        # Allow user to choose Top N
        top_n = st.slider("Show top N", min_value=5, max_value=min(20, len(top_store_item)), value=10)

        top_n_df = top_store_item.head(top_n).copy()
        top_n_df["label"] = top_n_df["store"].astype(str) + "-" + top_n_df["item"].astype(str)

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            y=top_n_df["label"],
            x=top_n_df["total_units_ordered"],
            orientation="h",
            name="Units Ordered",
            marker_color=CATEGORICAL["blue"],
            hovertemplate="Store-Item: %{y}<br>Units Ordered: %{x:,.0f}<extra></extra>",
        ))

        fig6.update_layout(
            title=f"Top {top_n} Store–Item Pairs by Total Units Ordered",
            xaxis_title="Total Units Ordered",
            yaxis_title="Store–Item",
            showlegend=False,
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK_PRIMARY),
            xaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED)),
            yaxis=dict(gridcolor=GRIDLINE, tickfont=dict(color=INK_MUTED), autorange="reversed"),
            margin=dict(l=100, r=20, t=60, b=50),
        )

        st.plotly_chart(fig6, use_container_width=True)
        st.caption("Helps you focus on the store–item combinations that move the most product.")
    else:
        st.info("No top store–item data available.")

    st.divider()

    st.subheader("Recommended Orders")

    rec_store_col = find_column(recommended_orders, ["store"])
    rec_item_col = find_column(recommended_orders, ["item"])
    rec_date_col = find_column(recommended_orders, ["date"])

    rec_df = recommended_orders.copy()

    if selected_stores and rec_store_col:
        rec_df = rec_df[rec_df[rec_store_col].isin(selected_stores)]

    if selected_items and rec_item_col:
        rec_df = rec_df[rec_df[rec_item_col].isin(selected_items)]

    if rec_date_col:
        rec_df[rec_date_col] = pd.to_datetime(rec_df[rec_date_col])
        rec_df = rec_df.sort_values(rec_date_col, ascending=False)

    # Show first 100 rows by default
    display_df = rec_df.head(100)

    # Column formatting for dataframe
    column_config = {}
    for c in display_df.columns:
        if display_df[c].dtype in ["float64", "float32", "int64", "int32"]:
            column_config[c] = st.column_config.NumberColumn(c, format="%.2f")

    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

    if len(rec_df) > 100:
        st.caption(f"Showing 100 of {len(rec_df)} rows. Download the full list below.")

    # Download button
    csv = rec_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"\U0001f4e5 Download all {len(rec_df)} recommended orders (CSV)",
        data=csv,
        file_name="recommended_orders_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Lists the order quantities the model recommends for each day, store, and item. "
        "Use the download button to get the full filtered list for planning."
    )