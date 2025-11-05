import streamlit as st
import pandas as pd
from utils import *

node_idx = pd.read_csv('/Users/ed_aguilar/Documents/yokoli/supply_chain/data/raw/homogenoeus/Nodes/NodesIndex.csv')
nodes = pd.read_csv('/Users/ed_aguilar/Documents/yokoli/supply_chain/data/raw/homogenoeus/Nodes/Node Types (Product Group and Subgroup).csv')
nodes = pd.merge(left=nodes, right=node_idx, on='Node')
node_to_idx = {node: idx for node, idx in zip(nodes['Node'].tolist(), nodes['NodeIndex'].tolist())}

st.markdown("# Temporal Graph Dataset Modelling")
st.write("This is an example of an app for analysis of temporal graph models using Graph Neural Networks.")

# --- Dataset description & compact dashboard (replace your placeholder) ---
st.markdown("## About the dataset used in this demo")

st.markdown(
    """
This demo uses the **SCG** benchmark — a multi-perspective temporal supply-chain dataset
built for Graph Neural Network research (products as nodes; relations such as same group,
plant or storage as edges). It contains **temporal node features** for each product:
**Production**, **Sales Order**, **Delivery to Distributors**, and **Factory Issue** (available
both in *units* and *weight*). The temporal coverage is **Jan 1, 2023 — Aug 9, 2023**.
"""
)

# compact metrics (paper values)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Edge types", "62")
col_b.metric("Unique edges", "684")
col_c.metric("Product groups", "5")
col_d.metric("Product subgroups", "19")

col_e, col_f, col_g = st.columns(3)
col_e.metric("Plants (edge class)", "25")
col_f.metric("Storage (edge class)", "13")
col_g.metric("Temporal features", "4")

st.markdown(
    """
**Why SCG is ideal for this POC**
- Real-world FMCG supply-chain data reorganised for temporal graph modelling (nodes = SKUs).
- Contains both homogeneous and heterogeneous graph views (products, plants, storage).
- Designed for forecasting, relation detection/classification, and anomaly detection with GNNs.
"""
)

with st.expander("Read the paper abstract & dataset availability"):
    st.write(
        "Graph Neural Networks (GNNs) are well-suited for supply chain problems because supply chains "
        "are inherently graph-like. The SCG dataset was collected from a large FMCG company's central "
        "database, curated for temporal graph use, and is publicly available (DOI & GitHub)."
    )
    st.markdown("**DOI / repository:** [SCG dataset (zenodo / GitHub)](https://doi.org/10.5281/zenodo.13652826)")

# citation (from the uploaded paper)
st.caption("Dataset summary and key numbers taken from: Wasi et al., *Graph Neural Networks in Supply Chain Analytics and Optimization* (SCG dataset).")
# (source for above: file uploaded by user).  [oai_citation:0‡supply.pdf](sediment://file_000000009a70720a8fd671bdfa9a6740)


st.markdown("# Data visualisation")


# --- Load base node data ---
node_idx = pd.read_csv('data/raw/homogenoeus/Nodes/NodesIndex.csv')
nodes = pd.read_csv('data/raw/homogenoeus/Nodes/Node Types (Product Group and Subgroup).csv')
nodes = pd.merge(left=nodes, right=node_idx, on='Node')
node_to_idx = {node: idx for node, idx in zip(nodes['Node'], nodes['NodeIndex'])}

# --- Load temporal data ---
delivery = load_and_clean('data/raw/homogenoeus/Temporal Data/Weight/Delivery_to_Distributor.csv', node_to_idx)
issue = load_and_clean('data/raw/homogenoeus/Temporal Data/Weight/Factory_Issue.csv', node_to_idx)
production = load_and_clean('data/raw/homogenoeus/Temporal Data/Weight/Production.csv', node_to_idx)
sales = load_and_clean('data/raw/homogenoeus/Temporal Data/Weight/Sales_Order.csv', node_to_idx)
temporal_data = {"Delivery": delivery, "Issues": issue, "Production": production, "Sales": sales}

# Load adjacency
edges = pd.read_csv('/Users/ed_aguilar/Documents/yokoli/supply_chain/data/raw/homogenoeus/Edges/EdgesIndex/Edges (Plant).csv', index_col=0)
edges_np = edges.to_numpy().T
edge_list = list(zip(edges_np[0], edges_np[1]))

# --- UI for graph animation ---
st.markdown("## 🎬 Animated Supply Chain Graph")

selected_var = st.selectbox("Select temporal variable:", list(temporal_data.keys()))
filter_mode = st.radio("Select filtering mode:", ["All Products", "By Product Group", "By Sub-Group"], horizontal=True)

selected_group, selected_subgroup = None, None
if filter_mode == "By Product Group":
    selected_group = st.selectbox("Choose a product group:", sorted(nodes["Group"].unique()))
elif filter_mode == "By Sub-Group":
    selected_subgroup = st.selectbox("Choose a sub-group:", sorted(nodes["Sub-Group"].unique()))

# 🔥 New toggle
color_by_value = st.checkbox("Color nodes by value intensity (instead of group)", value=False)

fig = plot_temporal_graph(
    nodes=nodes,
    node_to_idx=node_to_idx,
    edges_path='data/raw/homogenoeus/Edges/EdgesIndex/Edges (Plant).csv',
    temporal_data=temporal_data,
    selected_var=selected_var,
    filter_mode=filter_mode,
    selected_group=selected_group,
    selected_subgroup=selected_subgroup,
    color_by_value=color_by_value,  # <--- use new flag
)

if fig:
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Temporal graph animation generated using Plotly and NetworkX.")
else:
    st.info("No nodes available for this selection.")


st.markdown("## 🧩 Product Hierarchy Overview")
# Let the user choose what to visualize
level_choice = st.radio(
    "Select the level to visualize:",
    options=["Product Group", "Product Sub-Group"],
    index=1,
    horizontal=True,
    help="Choose whether to display distribution by main product groups or by detailed sub-groups.",
)
level = "Group" if level_choice == "Product Group" else "Sub-Group"
fig = plot_subgroup_distribution(df=nodes, level=level)
st.plotly_chart(fig, use_container_width=True)

# delivery
delivery = pd.read_csv('data/raw/homogenoeus/Temporal Data/Weight/Delivery_to_Distributor.csv', index_col=0)
delivery = delivery.drop(columns=['POP001L12P.1'])
#delivery = delivery.rename(columns=node_to_idx)

# factory issue
issue = pd.read_csv('data/raw/homogenoeus/Temporal Data/Weight/Factory_Issue.csv', index_col=0)
issue = issue.drop(columns=['POP001L12P.1'])
#issue = issue.rename(columns=node_to_idx)

#production
production = pd.read_csv('data/raw/homogenoeus/Temporal Data/Weight/Production.csv', index_col=0)
production = production.drop(columns=['POP001L12P.1'])
#production = production.rename(columns=node_to_idx)

# sales order
sales = pd.read_csv('data/raw/homogenoeus/Temporal Data/Weight/Sales_Order.csv', index_col=0)
sales = sales.drop(columns=['POP001L12P.1'])
#sales = sales.rename(columns=node_to_idx)

data_map = {
        "Delivery": delivery,
        "Issues": issue,
        "Production": production,
        "Sales": sales,
    }

st.markdown("## 📊 Visualise the trends in the data")


analyse_data = st.selectbox(
    "Select a Hierarchy to analyse",
    options=['Products', 'Group', 'Sub-Group'],
    index=0
)

col1, col2 = st.columns(2)

with col1:
    if analyse_data == 'Products':
        selected_products = st.multiselect(
            "Select a product",
            options=sales.columns,
        )
        string = 'Product'
        prods = None
    elif analyse_data == 'Group':
        selected_group = st.selectbox(
            'Select a Group to analyse',
            options=nodes['Group'].unique(),
            index=0,
        )
        selected_products = nodes.loc[nodes['Group'] == selected_group, 'Node'].tolist()
        string = 'Group'
        prods = selected_group

    elif analyse_data == 'Sub-Group':
        selected_subgroup = st.selectbox(
            'Select a Group to analyse',
            options=nodes['Sub-Group'].unique(),
            index=0,
        )
        selected_products = nodes.loc[nodes['Sub-Group'] == selected_subgroup, 'Node'].tolist()
        string = 'Sub-Group'
        prods = selected_subgroup

with col2:
    selected_vars = st.multiselect(
        "Select variables to plot",
        options=["Delivery", "Issues", "Production", "Sales"],
        default=["Delivery"],
    )

if selected_products and selected_vars:
    fig = plot_product_trends(selected_products, selected_vars, data_map, string, prods)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Please select a product and at least one variable.")






st.markdown(" ## Example of prediction task.")

st.write("The prediction task explored herein is: given the production, delivery and issue data, can you predict the demand (sales) of the following timestamp?")

idx_to_node = {value: key for key, value in node_to_idx.items()}
idx_to_node_2 = {str(value): key for key, value in node_to_idx.items()}

preds = pd.read_csv('predictions_next_step.csv', index_col=0)
preds.index.name = 'Date'

preds = preds.rename(columns=idx_to_node_2)
sales = sales.rename(columns=idx_to_node)

# Assume `sales` and `preds` are already loaded pandas DataFrames
node_to_plot = st.selectbox("Select Node/Product", options=sales.columns.tolist())

fig = plot_node_predictions(node_to_plot, sales, preds)
st.plotly_chart(fig, use_container_width=True)

