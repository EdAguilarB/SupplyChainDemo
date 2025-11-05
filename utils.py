import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_group_color_map(df, group_col="Group"):
    """Create consistent color and order mapping for groups."""
    unique_groups = sorted(df[group_col].unique().tolist())
    palette = px.colors.qualitative.Set2
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
    return unique_groups, color_map


def plot_subgroup_distribution(
    df,
    group_col="Group",
    subgroup_col="Sub-Group",
    level="Sub-Group",
    color_map=None,
    group_order=None,
):
    """Plot frequency distribution at group or subgroup level with consistent colors and order."""
    if group_col not in df.columns or subgroup_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{group_col}' and '{subgroup_col}' columns."
        )

    if color_map is None or group_order is None:
        group_order, color_map = get_group_color_map(df, group_col)

    if level == "Group":
        counts = df[group_col].value_counts().reindex(group_order).reset_index()
        counts.columns = [group_col, "Count"]
        fig = px.bar(
            counts,
            x=group_col,
            y="Count",
            color=group_col,
            text="Count",
            color_discrete_map=color_map,
            category_orders={group_col: group_order},
            title=f"Frequency of {group_col}",
        )
    else:
        counts = df.groupby([subgroup_col, group_col]).size().reset_index(name="Count")

        # Define subgroup order respecting group order
        subgroup_order = []
        for g in group_order:
            subgroups = (
                counts.loc[counts[group_col] == g, subgroup_col].unique().tolist()
            )
            subgroup_order.extend(subgroups)

        fig = px.bar(
            counts,
            x=subgroup_col,
            y="Count",
            color=group_col,
            barmode="group",
            text="Count",
            color_discrete_map=color_map,
            category_orders={
                group_col: group_order,
                subgroup_col: subgroup_order,
            },
            title=f"Frequency of {subgroup_col} by {group_col}",
        )

    # Style
    fig.update_traces(
        textposition="outside",
        textfont_size=14,
        marker_line_width=0.8,
        width=0.35,
    )
    fig.update_layout(
        xaxis_title=level,
        yaxis_title="Frequency",
        legend_title=group_col,
        template="plotly_white",
        title_x=0.5,
        bargap=0.10,
        bargroupgap=0.05,
        margin=dict(t=80, b=120, l=80, r=80),
        height=700,
        font=dict(size=16),
        title_font=dict(size=22),
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        legend_font=dict(size=14),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    return fig


def plot_product_trends(products: list, variables: list[str], data_map, string, prods):
    """
    Plot time-series trends for the selected product and variables.
    Assumes each dataframe column corresponds to a product,
    and index corresponds to chronological days.
    """
    # Mapping from variable name to dataframe

    # Initialize Plotly figure
    fig = go.Figure()

    # Add one line per selected variable
    for var in variables:
        df = data_map[var]
        y = df[products].sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y,
                mode="lines+markers",
                name=var,
            )
        )

    if string == "Product":
        prd = products.pop(0)
        for product in products:
            prd += f", {product}"
        title = f"Trends for the products: {prd}"
    elif string == "Group":
        title = f"Trend for group: {prods}"
    elif string == "Sub-Group":
        title = f"Trend for sub-group: {prods}"

    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Variable"),
    )

    return fig


def load_and_clean(path, node_map):
    """Load temporal CSV and rename columns to node indices."""
    df = pd.read_csv(path, index_col=0).drop(columns=["POP001L12P.1"], errors="ignore")
    # df = df.rename(columns=node_map)
    return df


def plot_temporal_graph(
    nodes: pd.DataFrame,
    node_to_idx: dict,
    edges_path: str,
    temporal_data: dict,
    selected_var: str,
    filter_mode: str = "All Products",
    selected_group: str | None = None,
    selected_subgroup: str | None = None,
    color_by_value: bool = False,  # 🔥 New parameter
):
    """Build and animate the temporal supply-chain graph with optional value-based coloring."""

    # --- Load adjacency ---
    edges = pd.read_csv(edges_path, index_col=0)
    edges_np = edges.to_numpy().T
    edge_list = list(zip(edges_np[0], edges_np[1]))

    # --- Filter nodes ---
    if filter_mode == "By Product Group" and selected_group:
        filtered_nodes = nodes[nodes["Group"] == selected_group]
    elif filter_mode == "By Sub-Group" and selected_subgroup:
        filtered_nodes = nodes[nodes["Sub-Group"] == selected_subgroup]
    else:
        filtered_nodes = nodes.copy()

    selected_node_indices = set(filtered_nodes["NodeIndex"].tolist())
    selected_nodes = set(filtered_nodes["Node"].tolist())

    # --- Build NetworkX graph ---
    G = nx.Graph()
    for _, row in filtered_nodes.iterrows():
        G.add_node(
            row["NodeIndex"],
            label=row["Node"],
            group=row["Group"],
            subgroup=row["Sub-Group"],
        )
    for src, tgt in edge_list:
        if src in selected_node_indices and tgt in selected_node_indices:
            G.add_edge(src, tgt)

    if len(G.nodes) == 0:
        return None

    # --- Layout and color mapping ---
    pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)
    unique_groups = sorted(nodes["Group"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    # --- Temporal data ---
    df_time = temporal_data[selected_var].loc[
        :, temporal_data[selected_var].columns.isin(selected_nodes)
    ]
    df_time_norm = (df_time - df_time.min()) / (df_time.max() - df_time.min() + 1e-6)
    df_time_norm = df_time_norm.fillna(0)

    node_order = list(G.nodes())
    x_nodes = np.array([pos[n][0] for n in node_order])
    y_nodes = np.array([pos[n][1] for n in node_order])

    # --- Animation frames ---
    frames = []
    for timestamp in df_time_norm.index:
        sizes = [max(8, 20 * df_time_norm.loc[timestamp].get(n, 0)) for n in node_order]

        # Color logic 🔥
        if color_by_value:
            colors = [df_time_norm.loc[timestamp].get(n, 0) for n in node_order]
        else:
            colors = [color_map[G.nodes[n]["group"]] for n in node_order]

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=[pos[e[0]][0] for e in G.edges()]
                    + [pos[e[1]][0] for e in G.edges()],
                    y=[pos[e[0]][1] for e in G.edges()]
                    + [pos[e[1]][1] for e in G.edges()],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(180,180,180,0.3)"),
                    hoverinfo="none",
                ),
                go.Scatter(
                    x=x_nodes,
                    y=y_nodes,
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale="Turbo" if color_by_value else None,
                        cmin=0 if color_by_value else None,
                        cmax=1 if color_by_value else None,
                        showscale=color_by_value,
                        line=dict(width=1.5, color="DarkSlateGrey"),
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"<b>{G.nodes[n]['label']}</b><br>"
                        f"Group: {G.nodes[n]['group']}<br>"
                        f"Sub-group: {G.nodes[n]['subgroup']}<br>"
                        f"{selected_var}: {df_time.loc[timestamp].get(n, 0):.2f}"
                        for n in node_order
                    ],
                ),
            ],
            name=str(timestamp),
        )
        frames.append(frame)

    # --- Static traces for initial frame ---
    edge_x, edge_y = [], []
    for e in G.edges():
        edge_x += [pos[e[0]][0], pos[e[1]][0], None]
        edge_y += [pos[e[0]][1], pos[e[1]][1], None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
        hoverinfo="none",
        mode="lines",
    )

    initial_sizes = [max(8, 20 * df_time_norm.iloc[0].get(n, 0)) for n in node_order]
    if color_by_value:
        initial_colors = [df_time_norm.iloc[0].get(n, 0) for n in node_order]
    else:
        initial_colors = [color_map[G.nodes[n]["group"]] for n in node_order]

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(
            size=initial_sizes,
            color=initial_colors,
            colorscale="Turbo" if color_by_value else None,
            cmin=0 if color_by_value else None,
            cmax=1 if color_by_value else None,
            showscale=color_by_value,
            line=dict(width=1.5, color="DarkSlateGrey"),
        ),
        hoverinfo="text",
        hovertext=[
            f"<b>{G.nodes[n]['label']}</b><br>"
            f"Group: {G.nodes[n]['group']}<br>"
            f"Sub-group: {G.nodes[n]['subgroup']}<br>"
            f"{selected_var}: {df_time.iloc[0].get(n, 0):.2f}"
            for n in node_order
        ],
    )

    # --- Final figure ---
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Temporal evolution of {selected_var}",
                x=0.5,
                font=dict(size=22),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(l=10, r=10, b=10, t=60),
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {"frame": {"duration": 0}, "mode": "immediate"},
                            ],
                        },
                    ],
                    "x": 0.1,
                    "y": -0.1,
                    "xanchor": "right",
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"prefix": "Time: ", "font": {"size": 18}},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [
                                [str(ts)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": str(ts),
                            "method": "animate",
                        }
                        for ts in df_time.index
                    ],
                }
            ],
            height=800,
        ),
        frames=frames,
    )

    return fig


def plot_predictions_level(sales, preds, nodes, level="Node", selection=None):
    """
    Plot real vs predicted demand over time at different levels:
      - 'Node': individual product
      - 'Sub-Group': sum across all nodes in the selected sub-group
      - 'Group': sum across all nodes in the selected group

    Args:
        sales (pd.DataFrame): True sales values (index = date, columns = node names)
        preds (pd.DataFrame): Predicted values with 'split' column
        nodes (pd.DataFrame): Node metadata with columns ['Node', 'Group', 'Sub-Group']
        level (str): One of {'Node', 'Sub-Group', 'Group'}
        selection (str): Selected entity (node, group, or sub-group)
    Returns:
        fig (plotly.graph_objects.Figure)
    """
    # Validate
    if level not in ["Node", "Sub-Group", "Group"]:
        raise ValueError("Level must be 'Node', 'Sub-Group', or 'Group'.")

    # --- Aggregate data depending on level ---
    if level == "Node":
        label = selection
        real_series = sales[selection]
        pred_series_train = preds.loc[preds["split"] == "train", selection]
        pred_series_test = preds.loc[preds["split"] == "test", selection]

    elif level == "Sub-Group":
        subset_nodes = nodes.loc[nodes["Sub-Group"] == selection, "Node"].tolist()
        label = f"Sub-Group: {selection}"
        real_series = sales[subset_nodes].sum(axis=1)
        pred_series_train = preds.loc[preds["split"] == "train", subset_nodes].sum(
            axis=1
        )
        pred_series_test = preds.loc[preds["split"] == "test", subset_nodes].sum(axis=1)

    elif level == "Group":
        subset_nodes = nodes.loc[nodes["Group"] == selection, "Node"].tolist()
        label = f"Group: {selection}"
        real_series = sales[subset_nodes].sum(axis=1)
        pred_series_train = preds.loc[preds["split"] == "train", subset_nodes].sum(
            axis=1
        )
        pred_series_test = preds.loc[preds["split"] == "test", subset_nodes].sum(axis=1)

    # --- Plot ---
    fig = go.Figure()

    # Real values
    fig.add_trace(
        go.Scatter(
            x=sales.index,
            y=real_series,
            mode="lines+markers",
            name="Real",
            line=dict(color="black", width=2),
            marker=dict(size=6),
        )
    )

    # Predicted (train)
    fig.add_trace(
        go.Scatter(
            x=pred_series_train.index,
            y=pred_series_train,
            mode="lines+markers",
            name="Predicted (train)",
            line=dict(color="green", dash="solid"),
            marker=dict(size=6),
        )
    )

    # Predicted (test)
    fig.add_trace(
        go.Scatter(
            x=pred_series_test.index,
            y=pred_series_test,
            mode="lines+markers",
            name="Predicted (test)",
            line=dict(color="red", dash="dot"),
            marker=dict(size=6),
        )
    )

    # --- Layout ---
    fig.update_layout(
        title=f"📈 Real vs Predicted Demand ({label})",
        xaxis_title="Date",
        yaxis_title="Demand (weight)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1.1, orientation="h"),
        height=550,
    )

    return fig
