import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import (get_group_color_map, load_and_clean, plot_predictions_level,
                   plot_product_trends, plot_subgroup_distribution,
                   plot_temporal_graph)

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Temporal Graph Analytics", layout="wide")
st.markdown("# 🧠 Supply Chain Temporal Graph Analytics Dashboard")
st.markdown(
    """
This interactive dashboard demonstrates the **end-to-end analytics pipeline**
for temporal and graph-based modeling of a real supply chain system.

It uses a **temporal graph dataset** that integrates production, delivery,
factory issues, and sales over time for each product node.
Each analytical step corresponds to a different stage in data exploration and modeling:
"""
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
node_idx = pd.read_csv("data/raw/homogenoeus/Nodes/NodesIndex.csv")
nodes = pd.read_csv(
    "data/raw/homogenoeus/Nodes/Node Types (Product Group and Subgroup).csv"
)
nodes = pd.merge(left=nodes, right=node_idx, on="Node")
node_to_idx = {node: idx for node, idx in zip(nodes["Node"], nodes["NodeIndex"])}

# Temporal data sources
delivery = load_and_clean(
    "data/raw/homogenoeus/Temporal Data/Weight/Delivery_to_Distributor.csv", node_to_idx
)
issue = load_and_clean(
    "data/raw/homogenoeus/Temporal Data/Weight/Factory_Issue.csv", node_to_idx
)
production = load_and_clean(
    "data/raw/homogenoeus/Temporal Data/Weight/Production.csv", node_to_idx
)
sales = load_and_clean(
    "data/raw/homogenoeus/Temporal Data/Weight/Sales_Order.csv", node_to_idx
)

data_map = {
    "Delivery": delivery,
    "Issues": issue,
    "Production": production,
    "Sales": sales,
}


st.markdown("## 📊 Dataset Summary")

# Compute dataset stats
num_nodes = len(nodes)
num_groups = nodes["Group"].nunique()
num_subgroups = nodes["Sub-Group"].nunique()

# Load edge list for stats
edges = pd.read_csv(
    "data/raw/homogenoeus/Edges/EdgesIndex/Edges (Plant).csv", index_col=0
)
num_edges = edges.shape[0]

# Time coverage
time_steps = len(sales.index)
time_start = sales.index[0]
time_end = sales.index[-1]

# Display in metric cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🧩 Total Nodes", f"{num_nodes:,}")
col2.metric("🏷️ Product Groups", f"{num_groups}")
col3.metric("🔖 Sub-Groups", f"{num_subgroups}")
col4.metric("🔗 Edges", f"{num_edges:,}")
col5.metric("⏱️ Time Steps", f"{time_steps}")

st.markdown(
    f"""
**Temporal coverage:** {time_start} → {time_end}
This dataset represents the operational dynamics of a large-scale supply chain,
where each node corresponds to a **product entity** and edges represent **shared plants or distribution links**.
The temporal dimension (daily measurements) enables the modeling of production, sales, delivery, and issue patterns over time.
"""
)

with st.expander("Read the paper abstract & dataset availability"):
    st.write(
        "Graph Neural Networks (GNNs) are well-suited for supply chain problems because supply chains "
        "are inherently graph-like. The SCG dataset was collected from a large FMCG company's central "
        "database, curated for temporal graph use, and is publicly available (DOI & GitHub)."
    )
    st.markdown(
        "**DOI / repository:** [SCG dataset (zenodo / GitHub)](https://doi.org/10.5281/zenodo.13652826)"
    )

# ------------------------------------------------------------
# STEP 1: PRODUCT HIERARCHY OVERVIEW
# ------------------------------------------------------------
st.markdown("## 🔹 Step 1 – Product Hierarchy Analysis")
st.markdown(
    """
This section explores the **composition of the product hierarchy** in the dataset.
You can toggle between viewing frequencies by **product group** or **sub-group** to understand
how products are distributed across the supply chain categories.
"""
)

# Select visualization level
level_choice = st.radio(
    "Select hierarchy level:",
    options=["Product Group", "Product Sub-Group"],
    index=0,
    horizontal=True,
)

# Compute and plot
group_order, color_map = get_group_color_map(nodes, "Group")
level = "Group" if level_choice == "Product Group" else "Sub-Group"
fig_hierarchy = plot_subgroup_distribution(
    nodes, level=level, color_map=color_map, group_order=group_order
)
st.plotly_chart(fig_hierarchy, use_container_width=True)

st.info(
    "✅ *Interpretation:* Larger bars indicate more nodes (products) belonging to that group or sub-group. "
    "This helps identify dominant product families or areas requiring deeper analysis.*"
)

# ------------------------------------------------------------
# STEP 2: TEMPORAL TRENDS EXPLORATION
# ------------------------------------------------------------
st.markdown("## 🔹 Step 2 – Temporal Trends Analysis")
st.markdown(
    """
In this section, we analyze the **temporal dynamics** of production, sales, delivery,
and issue volumes.
This helps identify **seasonal trends**, **demand–supply alignment**, or **operational anomalies**.
"""
)

# User inputs
col1, col2 = st.columns([2, 2])
with col1:
    trend_level = st.selectbox(
        "Select level of analysis:", ["Product", "Sub-Group", "Group"]
    )
with col2:
    selected_vars = st.multiselect(
        "Select variables to visualize:",
        options=["Production", "Sales", "Delivery", "Issues"],
        default=["Sales"],
    )

if trend_level == "Product":
    selected_products = st.multiselect(
        "Select one or more products:",
        options=sales.columns.tolist(),
        default=[sales.columns[0]],
    )
    fig_trends = plot_product_trends(
        selected_products, selected_vars, data_map, "Product", None
    )

elif trend_level == "Group":
    selected_group = st.selectbox("Select a group:", sorted(nodes["Group"].unique()))
    group_nodes = nodes[nodes["Group"] == selected_group]["NodeIndex"].tolist()
    fig_trends = plot_product_trends(
        group_nodes, selected_vars, data_map, "Group", selected_group
    )

else:  # Sub-Group
    selected_subgroup = st.selectbox(
        "Select a sub-group:", sorted(nodes["Sub-Group"].unique())
    )
    subgroup_nodes = nodes[nodes["Sub-Group"] == selected_subgroup][
        "NodeIndex"
    ].tolist()
    fig_trends = plot_product_trends(
        subgroup_nodes, selected_vars, data_map, "Sub-Group", selected_subgroup
    )

st.plotly_chart(fig_trends, use_container_width=True)
st.info(
    "✅ *Interpretation:* Each line represents the temporal evolution of a variable "
    "(e.g., production or sales). Comparing curves highlights mismatches or delays between "
    "production and demand.*"
)

# ------------------------------------------------------------
# STEP 3: TEMPORAL GRAPH MODELING
# ------------------------------------------------------------
st.markdown("## 🔹 Step 3 – Temporal Graph Representation and Dynamics")
st.markdown(
    """
Here we reconstruct the **graph structure** of the supply chain, where each node represents a product,
and edges represent operational relationships (e.g., shared plant, production flow).

The animation illustrates how **node-level activity** (e.g., sales or production) changes over time.
You can optionally color nodes by their **quantitative intensity** instead of group,
to emphasize high-activity regions in the network.
"""
)

selected_var = st.selectbox(
    "Select the variable to animate:",
    ["Delivery", "Issues", "Production", "Sales"],
)
filter_mode = st.radio(
    "Filter nodes by:",
    ["All Products", "By Product Group", "By Sub-Group"],
    horizontal=True,
)

selected_group, selected_subgroup = None, None
if filter_mode == "By Product Group":
    selected_group = st.selectbox(
        "Choose a product group:", sorted(nodes["Group"].unique())
    )
elif filter_mode == "By Sub-Group":
    selected_subgroup = st.selectbox(
        "Choose a sub-group:", sorted(nodes["Sub-Group"].unique())
    )

color_by_value = st.checkbox(
    "🎨 Color nodes by value intensity (instead of product group)",
    value=False,
    help="If enabled, nodes are colored based on the magnitude of the selected variable over time.",
)

fig_graph = plot_temporal_graph(
    nodes=nodes,
    node_to_idx=node_to_idx,
    edges_path="data/raw/homogenoeus/Edges/EdgesIndex/Edges (Plant).csv",
    temporal_data=data_map,
    selected_var=selected_var,
    filter_mode=filter_mode,
    selected_group=selected_group,
    selected_subgroup=selected_subgroup,
    color_by_value=color_by_value,
)

if fig_graph:
    st.plotly_chart(fig_graph, use_container_width=True)
    st.caption(
        "💡 *Node size reflects normalized activity intensity. Use the animation controls to observe temporal evolution across the supply chain network.*"
    )
else:
    st.warning("No nodes available for this selection.")


# =========================================================
# 🔹 Step 4 – Example of Prediction Task
# =========================================================
st.markdown("## 🔹 Step 4 – Example of Prediction Task")

st.markdown(
    """
    The **prediction task** explored herein evaluates whether temporal graph neural networks
    can forecast **future demand (sales)** one time step ahead, given information on
    **production**, **delivery**, and **factory issues** up to the current time.

    Use the selector below to explore predictions across individual products, sub-groups,
    or product groups. The model’s predicted demand (in weight units) is compared against
    the true observed demand for both the training and testing periods.
    """
)

# ---------------------------------------------------------
# Load and prepare predictions
# ---------------------------------------------------------
idx_to_node = {v: k for k, v in node_to_idx.items()}
idx_to_node_str = {str(v): k for k, v in node_to_idx.items()}

preds = pd.read_csv("predictions_next_step.csv", index_col=0)
preds.index.name = "Date"
preds = preds.rename(columns=idx_to_node_str)
sales = sales.rename(columns=idx_to_node)

# ---------------------------------------------------------
# Level selection
# ---------------------------------------------------------
st.markdown("### 🔍 Select Analysis Level")

level = st.radio(
    "View predictions by:",
    options=["Node", "Sub-Group", "Group"],
    horizontal=True,
    index=0,
)

if level == "Node":
    selection = st.selectbox("Select Product/Node", options=sales.columns.tolist())
elif level == "Sub-Group":
    selection = st.selectbox(
        "Select Sub-Group", options=sorted(nodes["Sub-Group"].unique().tolist())
    )
else:
    selection = st.selectbox(
        "Select Group", options=sorted(nodes["Group"].unique().tolist())
    )

# ---------------------------------------------------------
# Generate plot
# ---------------------------------------------------------
fig = plot_predictions_level(sales, preds, nodes, level=level, selection=selection)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Display performance metrics (RMSE & MAE)
# ---------------------------------------------------------


if level == "Node":
    y_true = sales[selection]
    y_pred = preds[selection]
elif level == "Sub-Group":
    subset = nodes.loc[nodes["Sub-Group"] == selection, "Node"].tolist()
    y_true = sales[subset].sum(axis=1)
    y_pred = preds[subset].sum(axis=1)
else:
    subset = nodes.loc[nodes["Group"] == selection, "Node"].tolist()
    y_true = sales[subset].sum(axis=1)
    y_pred = preds[subset].sum(axis=1)


# Compute metrics
rmse = np.sqrt(mean_squared_error(y_true[1:], y_pred))
mae = mean_absolute_error(y_true[1:], y_pred)

# Display in a stylish layout
st.markdown("### 📏 Model Performance Metrics")
col1, col2 = st.columns(2)
col1.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.2f}")
col2.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")

# ---------------------------------------------------------
# Footer note
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:15px; color:gray;">
        Built by <b>Eduardo Aguilar</b> · Contact: <i>ed.aguilar.bejarano@gmail.com</i><br>
        © 2025 – Temporal Graph Analytics Proof of Concept
    </div>
    """,
    unsafe_allow_html=True,
)
