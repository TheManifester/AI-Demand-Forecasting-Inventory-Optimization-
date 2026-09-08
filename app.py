import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI Demand Forecasting & Inventory Optimization")
st.markdown("AI-powered demand forecasting and inventory management dashboard")
st.divider()

DATA_PATH = "data/processed/"


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV once and cache it across reruns (filter changes no
    longer re-read every file from disk)."""
    df = pd.read_csv(DATA_PATH + name)
    # If the store/item ended up as the index instead of a column
    # (a common cause of "column not found" after groupby+to_csv),
    # pull it back out as a real column.
    if df.index.name and df.index.name.lower() not in ("", "unnamed: 0"):
        df = df.reset_index()
    return df


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive search for the first column whose name
    contains any of the candidate keywords. Returns None if nothing matches."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lower_name, real_name in cols_lower.items():
            if cand in lower_name:
                return real_name
    return None


def numeric_value_column(df: pd.DataFrame, exclude: list[str]) -> str | None:
    """Pick the last numeric column that isn't one of the id columns."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return numeric_cols[-1] if numeric_cols else None


def grouped_bar_chart(df: pd.DataFrame, group_candidates: list[str], label: str, agg="mean"):
    """Reusable bar chart: finds the id column + a value column, groups, and renders.
    Shows a helpful message (including the real column names found) if anything's missing."""
    id_col = find_column(df, group_candidates)

    if id_col is None:
        st.info(f"Couldn't find a '{'/'.join(group_candidates)}' column. "
                f"Available columns: {list(df.columns)}")
        return

    value_col = numeric_value_column(df, exclude=[id_col])

    if value_col is None:
        st.info(f"No numeric value column found alongside '{id_col}'.")
        return

    chart_data = df.groupby(id_col)[value_col].agg(agg)
    st.bar_chart(chart_data)


# ============================================================
# LOAD PROCESSED DATA (cached)
# ============================================================

inventory_by_store = load_csv("inventory_by_store.csv")
inventory_distribution = load_csv("inventory_distribution.csv")
inventory_simulation = load_csv("inventory_simulation.csv")
inventory_trend = load_csv("inventory_trend.csv")
kpis = load_csv("kpis.csv")
orders_by_item = load_csv("orders_by_item.csv")
orders_by_store = load_csv("orders_by_store.csv")
recommended_orders = load_csv("recommended_orders.csv")
top_store_item = load_csv("top_store_item.csv")

# Resolve the real store/item/order/inventory column names ONCE,
# so every section below uses the same detection instead of
# repeating literal "store" / "order_quantity" checks that silently fail.
store_col = find_column(inventory_simulation, ["store"])
item_col = find_column(inventory_simulation, ["item"])
inv_col = find_column(inventory_simulation, ["inventory_after_demand", "inventory"])
order_col = find_column(
    inventory_simulation,
    ["order_quantity", "order_decision_qty", "order_qty", "recommended_order", "order"]
)
date_col = find_column(inventory_simulation, ["date"])


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔍 Filters")

selected_store = "All"
if store_col:
    stores = sorted(inventory_simulation[store_col].dropna().unique())
    selected_store = st.sidebar.selectbox("Select Store", ["All"] + list(stores))

selected_item = "All"
if item_col:
    items = sorted(inventory_simulation[item_col].dropna().unique())
    selected_item = st.sidebar.selectbox("Select Item", ["All"] + list(items))


# ============================================================
# FILTER INVENTORY SIMULATION
# ============================================================

filtered_inventory = inventory_simulation.copy()

if selected_store != "All" and store_col:
    filtered_inventory = filtered_inventory[filtered_inventory[store_col] == selected_store]

if selected_item != "All" and item_col:
    filtered_inventory = filtered_inventory[filtered_inventory[item_col] == selected_item]


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if inv_col:
        st.metric("📦 Total Inventory", f"{filtered_inventory[inv_col].sum():,.0f}")
    else:
        st.metric("📦 Total Inventory", "--")

with col2:
    if inv_col:
        st.metric("📈 Average Inventory", f"{filtered_inventory[inv_col].mean():,.2f}")
    else:
        st.metric("📈 Average Inventory", "--")

with col3:
    if order_col:
        total_orders = (filtered_inventory[order_col] > 0).sum()
        st.metric("🛒 Total Orders", f"{total_orders:,}")
    else:
        st.metric("🛒 Total Orders", "--")
        st.caption(f"⚠️ No order column found. Columns available: {list(inventory_simulation.columns)}")

with col4:
    if inv_col and len(filtered_inventory) > 0:
        stockouts = (filtered_inventory[inv_col] <= 0).sum()
        stockout_risk = (stockouts / len(filtered_inventory)) * 100
        st.metric("⚠️ Stockout Risk", f"{stockout_risk:.2f}%")
    else:
        st.metric("⚠️ Stockout Risk", "--")

st.divider()


# ============================================================
# INVENTORY TREND
# ============================================================

st.subheader("📈 Inventory Trend")

if not filtered_inventory.empty and date_col and inv_col:
    trend_data = filtered_inventory.groupby(date_col)[inv_col].sum().reset_index()
    trend_data[date_col] = pd.to_datetime(trend_data[date_col])
    trend_data = trend_data.sort_values(date_col)
    st.line_chart(trend_data.set_index(date_col)[inv_col])
else:
    st.info("No inventory trend data available for the selected filters.")

st.divider()


# ============================================================
# STORE ANALYSIS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏪 Average Inventory by Store")

    if find_column(inventory_by_store, ["store"]):
        grouped_bar_chart(inventory_by_store, ["store"], "store", agg="mean")
    elif store_col and inv_col:
        # inventory_by_store.csv is missing the store labels (likely
        # saved with index=False upstream, dropping them). Fall back
        # to computing the same chart from inventory_simulation, which
        # still has a real store column.
        chart_data = inventory_simulation.groupby(store_col)[inv_col].mean()
        st.bar_chart(chart_data)
    else:
        st.info(f"Couldn't find a 'store' column anywhere. "
                f"inventory_by_store.csv columns: {list(inventory_by_store.columns)}")

with col2:
    st.subheader("📊 Inventory Distribution")

    dist_col = find_column(inventory_distribution, ["inventory_after_demand", "inventory"])

    if dist_col:
        # Bin continuous values into ranges instead of value_counts()
        # on raw floats (which produced a count-of-1-per-bar mess).
        binned = pd.cut(inventory_distribution[dist_col], bins=20)
        counts = binned.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        st.bar_chart(counts)
    else:
        st.info(f"No inventory column found. Available columns: {list(inventory_distribution.columns)}")

st.divider()


# ============================================================
# ORDER ANALYSIS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏪 Orders by Store")
    grouped_bar_chart(orders_by_store, ["store"], "store", agg="sum")

with col2:
    st.subheader("📦 Orders by Item")
    grouped_bar_chart(orders_by_item, ["item"], "item", agg="sum")

st.divider()


# ============================================================
# TOP STORE-ITEM COMBINATIONS
# ============================================================

st.subheader("🔥 Top Store-Item Combinations")

if not top_store_item.empty:
    st.dataframe(top_store_item, use_container_width=True, hide_index=True)
else:
    st.info("No top store-item data available.")

st.divider()


# ============================================================
# RECOMMENDED ORDERS
# ============================================================

st.subheader("🛒 Recommended Orders")

recommended_data = recommended_orders.copy()
rec_store_col = find_column(recommended_data, ["store"])
rec_item_col = find_column(recommended_data, ["item"])

if selected_store != "All" and rec_store_col:
    recommended_data = recommended_data[recommended_data[rec_store_col] == selected_store]

if selected_item != "All" and rec_item_col:
    recommended_data = recommended_data[recommended_data[rec_item_col] == selected_item]

if not recommended_data.empty:
    st.dataframe(recommended_data, use_container_width=True, hide_index=True)
else:
    st.info("No recommended orders for the selected filters.")