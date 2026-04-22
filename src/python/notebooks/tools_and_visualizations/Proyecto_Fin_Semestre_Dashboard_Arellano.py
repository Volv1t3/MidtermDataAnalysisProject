import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.pyplot import ticklabel_format
from plotly.subplots import make_subplots


# =============================================================================
# Data Preparation & Constants
# =============================================================================

COLOR_POSITIVE = "#A4A4A4"
COLOR_NEGATIVE = "#E01518"
COLOR_WARNING = '#F2AE30'
COLOR_ATTENTION = '#13678A'
COLOR_BG = "#F2F2F4"
COLOR_LINE = "#5A5A5C"
COLOR_GREEN_DARK = "#6E8C03"
COLOR_GREEN_LIGHT = "#C5D932"
COLOR_GREEN = "#6E8C03"
COLOR_RED = "#E01518"
COLOR_GREY = "#A4A4A4"

CATEGORY_COLORS = {
    "Furniture": "#636EFA",      # Blue
    "Office Supplies": "#FFA15A", # Orange
    "Technology": "#00CC96",      # Green
}

DISCOUNT_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
DISCOUNT_LABELS = [
    "0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-1.0",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add required derived columns."""
    df = df.copy()
    df["Profit_Margin"] = df["Profit"] / df["Sales"]
    df["Profit_Margin_Pct"] = df["Profit_Margin"] * 100
    df["Discount_Bin"] = pd.cut(
        df["Discount"], bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS, include_lowest=True
    )
    return df


def generate_color_gradient(n: int):
    """Generate gradient colors for sub-categories."""
    colors = []
    for i in range(n):
        ratio = i / (n - 1) if n > 1 else 0
        r1, g1, b1 = int(COLOR_GREEN_DARK[1:3], 16), int(COLOR_GREEN_DARK[3:5], 16), int(COLOR_GREEN_DARK[5:7], 16)
        r2, g2, b2 = int(COLOR_GREEN_LIGHT[1:3], 16), int(COLOR_GREEN_LIGHT[3:5], 16), int(COLOR_GREEN_LIGHT[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def _apply_standard_theme(fig, title, subtitle, x_label, y_label, x_rotation=30):
    """Unified styling for all dashboard graphs."""
    full_title = f"<b>{title}</b><br><i><span style='font-size:12px;font-weight:200;color:#666'>{subtitle}</span></i>"
    
    fig.update_layout(
        title={
            "text": full_title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.95,
            "font": {"size": 16, "family": "Arial"},
        },
        margin=dict(l=100, r=40, t=100, b=80),
        plot_bgcolor="white",
        paper_bgcolor=COLOR_BG,
        font=dict(family="Arial", size=11, color=COLOR_LINE),
    )
    
    # Hide grids and set standard axis appearance
    fig.update_xaxes(
        showgrid=False, 
        title="", 
        showline=True, 
        linewidth=2, 
        linecolor=COLOR_LINE,
        tickangle=x_rotation,
        ticks="outside",
        tickwidth=2,
        tickcolor=COLOR_LINE,
    )
    fig.update_yaxes(
        showgrid=False, 
        title="", 
        showline=True, 
        linewidth=2, 
        linecolor=COLOR_LINE,
        ticks="outside",
        tickwidth=2,
        tickcolor=COLOR_LINE,
    )
    
    # Bottom aligned Y title annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=-0.05, y=0,
        text=y_label,
        showarrow=False,
        textangle=-90,
        yanchor="bottom",
        font=dict(size=12, color=COLOR_LINE, family="Arial Black")
    )
    
    # Left aligned X title annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=-0.12,
        text=x_label,
        showarrow=False,
        xanchor="left",
        font=dict(size=12, color=COLOR_LINE, family="Arial Black")
    )
    
    return fig


# =============================================================================
# Filtering
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    date_range,
    time_field: str,
    region_sel,
    ship_sel,
    country_sel,
    state_sel,
    city_sel,
) -> pd.DataFrame:
    """Apply global sidebar filters to dataframe."""
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start, end = date_range
        if time_field in df.columns:
            col = df[time_field]
            mask &= (col >= pd.to_datetime(start)) & (col <= pd.to_datetime(end))
    if region_sel:
        mask &= df["Region"].isin(region_sel)
    if ship_sel:
        mask &= df["Ship Mode"].isin(ship_sel)
    if country_sel:
        mask &= df["Country"].isin(country_sel)
    if state_sel:
        mask &= df["State"].isin(state_sel)
    if city_sel:
        mask &= df["City"].isin(city_sel)
    return df.loc[mask]


# =============================================================================
# Aggregations
# =============================================================================

def get_aggregations(df: pd.DataFrame):
    """Compute all needed aggregations for dashboard graphs."""
    if df is None or df.empty:
        return {}

    # Profit by Discount Bin
    agg_profit_by_discount = df.groupby("Discount_Bin", observed=True)["Profit"].mean().reset_index()
    agg_profit_by_discount.columns = ["Discount_Bin", "Avg_Profit"]

    # Margin by Discount Bin
    agg_margin_by_discount = (
        df.groupby("Discount_Bin", observed=True)["Profit_Margin_Pct"].mean().reset_index()
    )
    agg_margin_by_discount.columns = ["Discount_Bin", "Avg_Profit_Margin_Pct"]

    # Margin by Discount & Category
    agg_margin_by_category = (
        df.groupby(["Discount_Bin", "Category"], observed=True)["Profit_Margin_Pct"].mean().reset_index()
    )
    agg_margin_by_category.columns = ["Discount_Bin", "Category", "Avg_Profit_Margin_Pct"]

    # Margin by Discount & Sub-Category
    agg_margin_by_subcategory = (
        df.groupby(["Discount_Bin", "Sub-Category"], observed=True)["Profit_Margin_Pct"]
        .mean()
        .reset_index()
    )
    agg_margin_by_subcategory.columns = ["Discount_Bin", "Sub_Category", "Avg_Profit_Margin_Pct"]

    # By Category (for scatter)
    agg_by_category = (
        df.groupby("Category")
        .agg({"Sales": "sum", "Profit": "sum", "Customer ID": "nunique"})
        .reset_index()
    )
    agg_by_category.columns = ["Category", "Total_Sales", "Total_Profit", "Unique_Customers"]
    agg_by_category["Profit_Margin_Pct"] = (
        agg_by_category["Total_Profit"] / agg_by_category["Total_Sales"]
    ) * 100
    agg_by_category["Avg_Sale_Per_Customer"] = (
        agg_by_category["Total_Sales"] / agg_by_category["Unique_Customers"]
    )

    # By Sub-Category (for scatter)
    agg_by_subcategory = (
        df.groupby(by=["Category", "Sub-Category"])
        .agg({"Sales": "sum", "Profit": "sum", "Customer ID": "nunique"})
        .reset_index()
    )
    agg_by_subcategory.columns = ["Category", "Sub_Category", "Total_Sales", "Total_Profit", "Unique_Customers"]
    agg_by_subcategory["Profit_Margin_Pct"] = (
        agg_by_subcategory["Total_Profit"] / agg_by_subcategory["Total_Sales"]
    ) * 100
    agg_by_subcategory["Avg_Sale_Per_Customer"] = (
        agg_by_subcategory["Total_Sales"] / agg_by_subcategory["Unique_Customers"]
    )

    return {
        "profit_by_discount": agg_profit_by_discount,
        "margin_by_discount": agg_margin_by_discount,
        "margin_by_category": agg_margin_by_category,
        "margin_by_subcategory": agg_margin_by_subcategory,
        "by_category": agg_by_category,
        "by_subcategory": agg_by_subcategory,
    }


# =============================================================================
# Charts - Waterfall
# =============================================================================

def plot_waterfall(agg_profit, agg_margin):
    """Waterfall charts for profit analysis."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Average Profit by Discount Range</b>",
            "<b>Profit Margin (%) by Discount Range</b>",
        ),
    )

    fig.add_trace(
        go.Waterfall(
            x=agg_profit["Discount_Bin"],
            y=agg_profit["Avg_Profit"],
            text=[f"${val:.2f}" for val in agg_profit["Avg_Profit"]],
            textposition="outside",
            increasing={"marker": {"color": COLOR_POSITIVE}},
            decreasing={"marker": {"color": COLOR_NEGATIVE}},
            connector={"line": {"color": COLOR_LINE}},
            name="Avg Profit",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Waterfall(
            x=agg_margin["Discount_Bin"],
            y=agg_margin["Avg_Profit_Margin_Pct"],
            text=[f"{val:.2f}%" for val in agg_margin["Avg_Profit_Margin_Pct"]],
            textposition="outside",
            increasing={"marker": {"color": COLOR_POSITIVE}},
            decreasing={"marker": {"color": COLOR_NEGATIVE}},
            connector={"line": {"color": COLOR_LINE}},
            name="Margin (%)",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=False,
        height=500,
    )
    
    # Special handling for subplots: titles are manual
    _apply_standard_theme(
        fig, 
        "Profit Analysis by Discount", 
        "Comparison of average profit and margin across discount segments",
        "Discount Range",
        "Economic Value ($ / %)"
    )
    return fig


# =============================================================================
# Charts - Lollipop
# =============================================================================

def plot_lollipop(agg_margin_by_cat, selected_categories=None):
    """Lollipop chart by category."""
    if selected_categories:
        data = agg_margin_by_cat[agg_margin_by_cat["Category"].isin(selected_categories)]
    else:
        data = agg_margin_by_cat

    fig = go.Figure()
    categories = data["Category"].unique()

    for idx, category in enumerate(categories):
        cat_data = data[data["Category"] == category]
        x_positions = []
        discount_bin_labels = []

        for i, discount_bin in enumerate(cat_data["Discount_Bin"]):
            offset = (idx - 1) * 0.25
            x_positions.append(i + offset)
            discount_bin_labels.append(str(discount_bin))

        # Draw stick
        for i, x in enumerate(x_positions):
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[0, cat_data["Avg_Profit_Margin_Pct"].iloc[i]],
                    mode="lines",
                    line=dict(color=CATEGORY_COLORS.get(category, "#333"), width=2),
                    hoverinfo="skip",
                    legendgroup=category,
                    name=category,
                    showlegend=False,
                )
            )

        # Draw candy
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=cat_data["Avg_Profit_Margin_Pct"],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=CATEGORY_COLORS.get(category, "#333"),
                    line=dict(color="white", width=1.5),
                ),
                text=[f"{val:.0f}%" for val in cat_data["Avg_Profit_Margin_Pct"]],
                textposition="middle center",
                textfont=dict(color="white", size=8, family="Arial Black"),
                name=category,
                legendgroup=category,
                customdata=discount_bin_labels,
                hovertemplate=f"<b>{category}</b><br>Margin: %{{y:.2f}}%<br>Discount: %{{customdata}}<extra></extra>",
            )
        )

    _apply_standard_theme(
        fig,
        "Profit Margin by Category",
        "Distribution of margins per category across discount ranges",
        "Discount Range",
        "Profit Margin (%)"
    )
    fig.update_layout(
        height=500,
        hovermode="closest",
        legend=dict(
            title="Category", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    return fig


# =============================================================================
# Charts - Line
# =============================================================================

def plot_heatmap(agg_margin_by_subcat, selected_subcats=None, top_n=None):
    """Heatmap for sub-categories profit margin."""
    data = agg_margin_by_subcat.copy()
    if selected_subcats:
        data = data[data["Sub_Category"].isin(selected_subcats)]
    elif top_n:
        overall = data.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().reset_index()
        top_subs = overall.nlargest(top_n, "Avg_Profit_Margin_Pct")["Sub_Category"].tolist()
        data = data[data["Sub_Category"].isin(top_subs)]

    if data.empty:
        return None

    # Transform for Heatmap
    pivot_df = data.pivot(index="Sub_Category", columns="Discount_Bin", values="Avg_Profit_Margin_Pct")
    # Ensure columns are in correct order from DISCOUNT_LABELS
    pivot_df = pivot_df.reindex(columns=DISCOUNT_LABELS)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Margin %"),
        hovertemplate="<b>%{y}</b><br>Discount: %{x}<br>Margin: %{z:.2f}%<extra></extra>",
        xgap=1,
        ygap=1
    ))

    _apply_standard_theme(
        fig,
        "Profitability Grid by Sub-Category",
        "Heatmap visualization of profit margins across all discount segments and sub-categories",
        "Discount Range",
        "Sub-Category"
    )
    
    fig.update_layout(height=600 if len(pivot_df) > 10 else 400)
    return fig


# =============================================================================
# Charts - Scatter
# =============================================================================

def plot_scatter_category(agg_data, min_sales=None, margin_range=None):
    """Scatter plot by category."""
    data = agg_data.copy()
    if min_sales:
        data = data[data["Total_Sales"] >= min_sales]
    if margin_range and len(margin_range) == 2:
        data = data[
            (data["Profit_Margin_Pct"] >= margin_range[0])
            & (data["Profit_Margin_Pct"] <= margin_range[1])
        ]

    if data.empty:
        return None

    avg_sales = data["Total_Sales"].mean()
    avg_margin = data["Profit_Margin_Pct"].mean()

    def assign_color(row):
        if row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] > avg_sales:
            return COLOR_GREEN  # High margin, high sales - optimal
        elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] < avg_sales:
            return COLOR_RED  # Low margin, low sales - poor performance
        elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] > avg_sales:
            return COLOR_WARNING  # High sales but low margin - needs attention
        else:  # row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] < avg_sales
            return COLOR_ATTENTION  # High margin but low sales - growth opportunity

    data["Color"] = data.apply(assign_color, axis=1)

    fig = go.Figure()
    fig.add_hline(
        y=avg_margin,
        line_dash='dash',
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Porcentaje de<br>Ganancia Promedio: <b>{avg_margin:.1f}%</b>",
        annotation_position="left",
        annotation_align='left',
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial",style='italic'),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.1
        )
    )

    fig.add_vline(
        x=avg_sales,
        line_dash="dash",
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Ventas Totales Promedio: <b>${avg_sales:,.0f}</b>",
        annotation_position="top",
        annotation_align='left',
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial", style='italic'),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )

    # Add color legend box in top right corner
    legend_x_start = 0.15
    legend_y_start = 1.06
    box_height = 0.03
    box_width = 0.04
    spacing = 0.01

    legend_items = [
        (COLOR_GREEN, "Margen Alto y Ventas Altas"),
        (COLOR_ATTENTION, "Margen Alto y Ventas Bajas"),
        (COLOR_WARNING, "Margen Bajo y Ventas Altas"),
        (COLOR_RED, "Margen Bajo y Ventas Bajas")
    ]

    current_x = legend_x_start
    for idx, (color, label) in enumerate(legend_items):
        # Add colored box
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=current_x, y0=legend_y_start - box_height,
            x1=current_x + box_width, y1=legend_y_start,
            fillcolor=color,
            line=dict(color=COLOR_LINE, width=1)
        )

        # Add text label
        fig.add_annotation(
            xref="paper", yref="paper",
            x=current_x + box_width + 0.005,
            y=legend_y_start - box_height / 2,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color=COLOR_LINE, family="Arial"),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=3
        )

        # Calculate width of text annotation (approximate)
        text_width = len(label) * 0.006  # Rough estimate
        current_x += box_width + text_width + spacing

    for _, row in data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Total_Sales"]],
                y=[row["Profit_Margin_Pct"]],
                mode="markers+text",
                marker=dict(
                    size=row["Avg_Sale_Per_Customer"] / 8,
                    color=row["Color"],
                    line=dict(color="white", width=1.5),
                    opacity=0.8,
                ),
                text=row["Category"],
                textposition="top center",
                textfont=dict(size=12, color="black", family="Arial Black"),
                name=row["Category"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Category']}</b><br>"
                    f"<b>Total de Ventas</b>: ${row['Total_Sales']:,.2f}<br>"
                    f"<b>Margen de Ganancia Promedio</b>: {row['Profit_Margin_Pct']:.2f}%<br>"
                    f"<b>Venta Promedio por Cliente</b>: ${row['Avg_Sale_Per_Customer']:.2f}<br>"
                    f"<b>Clientes Totales</b>: {row['Unique_Customers']}<extra></extra>"
                ),
            )
        )

    _apply_standard_theme(
        fig,
        "Comparativo de Rentabilidad en base al Porcentaje de Ganancia vs Ventas Totales",
        "Durante el periodo 2014-2017, <b>Technology</b> fue la categoría con las mayores ventas de $690K y el mejor margen ganancia promedio de 18.62%",
        "Ventas Totales ($)",
        "Margen de Ganancia Promedio (%)"
    )
    fig.update_layout(
        height=700,
        hovermode="closest",
        showlegend=True,
        xaxis={
            'zeroline':False
        },
        yaxis={
            'zeroline':False,
            'tickformat': '.0f',
            'ticksuffix': '%'

        }
    )
    return fig


def plot_scatter_subcategory(agg_data, cat_filter=None, min_sales=None, margin_range=None, top_n=None):
    """Scatter plot by sub-category.
    Recibimos como parámetro cat_filter:
    - Si es None, se grafican todas las subcategorías
    - Si es una lista de strings, se grafican solo las subcategorías que están en la lista"""
    agg_data_filtered = agg_data.copy()
    print(agg_data.columns)
    if cat_filter:
        agg_data_filtered = agg_data[agg_data["Category"].isin(cat_filter)]

        pass

    if min_sales:
        agg_data_filtered = agg_data_filtered[agg_data_filtered["Total_Sales"] >= min_sales]
    if margin_range and len(margin_range) == 2:
        agg_data_filtered = agg_data_filtered[
            (agg_data_filtered["Profit_Margin_Pct"] >= margin_range[0])
            & (agg_data_filtered["Profit_Margin_Pct"] <= margin_range[1])
        ]
    if top_n:
        agg_data_filtered = agg_data_filtered.nlargest(top_n, "Total_Sales")

    if agg_data_filtered.empty:
        return None

    avg_sales = agg_data_filtered["Total_Sales"].mean()
    avg_margin = agg_data_filtered["Profit_Margin_Pct"].mean()

    def assign_color(row):
        if row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] > avg_sales:
            return COLOR_GREEN  # High margin, high sales - optimal
        elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] < avg_sales:
            return COLOR_RED  # Low margin, low sales - poor performance
        elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] > avg_sales:
            return COLOR_WARNING  # High sales but low margin - needs attention
        else:  # row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] < avg_sales
            return COLOR_ATTENTION  # High margin but low sales - growth opportunity

    agg_data_filtered["Color"] = agg_data_filtered.apply(assign_color, axis=1)


    fig = go.Figure()
    fig.add_hline(
        y=avg_margin,
        line_dash='dash',
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Avg Margin: {avg_margin:.1f}%",
        annotation_position="left",
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial"),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.1
        )
    )

    fig.add_vline(
        x=avg_sales,
        line_dash="dash",
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Avg Sales: ${avg_sales:,.0f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial"),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )

    # Add color legend box in top right corner
    legend_x_start = 0.15
    legend_y_start = 1.05
    box_height = 0.03
    box_width = 0.04
    spacing = 0.01

    legend_items = [
        (COLOR_GREEN, "High Margin & High Sales"),
        (COLOR_ATTENTION, "High Margin & Low Sales"),
        (COLOR_WARNING, "Low Margin & High Sales"),
        (COLOR_RED, "Low Margin & Low Sales")
    ]

    current_x = legend_x_start
    for idx, (color, label) in enumerate(legend_items):
        # Add colored box
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=current_x, y0=legend_y_start - box_height,
            x1=current_x + box_width, y1=legend_y_start,
            fillcolor=color,
            line=dict(color=COLOR_LINE, width=1)
        )

        # Add text label
        fig.add_annotation(
            xref="paper", yref="paper",
            x=current_x + box_width + 0.005,
            y=legend_y_start - box_height / 2,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=9, color=COLOR_LINE, family="Arial"),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=3
        )

        # Calculate width of text annotation (approximate)
        text_width = len(label) * 0.006  # Rough estimate
        current_x += box_width + text_width + spacing

    for _, row in agg_data_filtered.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Total_Sales"]],
                y=[row["Profit_Margin_Pct"]],
                mode="markers+text",
                marker=dict(
                    size=row["Avg_Sale_Per_Customer"] / 20,
                    color=row["Color"],
                    line=dict(color="white", width=1),
                    opacity=0.7,
                ),
                text=row["Sub_Category"],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                name=row["Sub_Category"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Sub_Category']}</b><br>"
                    f"Sales: ${row['Total_Sales']:,.2f}<br>"
                    f"Margin: {row['Profit_Margin_Pct']:.2f}%<br>"
                    f"Avg Sale/Cust: ${row['Avg_Sale_Per_Customer']:.2f}<br>"
                    f"Customers: {row['Unique_Customers']}<extra></extra>"
                ),
            )
        )

    _apply_standard_theme(
        fig,
        "Granular Performance Analysis",
        "Detailed performance metrics for individual product types",
        "Total Sales ($)",
        "Profit Margin (%)"
    )
    fig.update_layout(
        height=700,
        hovermode="closest",
        showlegend=True,
        xaxis={'zeroline': False},
        yaxis={
            'zeroline': False,
            'tickformat': '.0f',
            'ticksuffix': '%'
        }
    )

    return fig


# =============================================================================
# Session State & Main Logic
# =============================================================================

def init_user_state() -> None:
    if "user_initialized" not in st.session_state:
        st.session_state.user_initialized = True
        st.session_state.active_filters = {}
        st.session_state.selected_view = "Overview"
        st.session_state.prepared_data = None
        st.session_state.filtered_data = None
        st.session_state.aggregations = {}


def load_data() -> pd.DataFrame:
    """Load dataset."""
    dataset_url = "https://github.com/Volv1t3/MidtermDataAnalysisProject/raw/refs/heads/Proyecto-Fin-Semestre/src/res/processed_data/ProyectoFinSemestreADM3083_SuperstoreProcessed.csv"
    dataset = pd.read_csv(dataset_url, sep=",")
    for column in ["Order Date", "Ship Date"]:
        dataset[column] = pd.to_datetime(dataset[column], errors="coerce")
    return dataset


def initialize_dashboard():
    init_user_state()
    data = load_data()
    data = prepare_data(data)
    st.session_state.prepared_data = data
    return data


def render_sidebar(data: pd.DataFrame | None):
    """Render sidebar controls."""
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("Configure filters and views here.")

    # --- Global Filters ---
    st.sidebar.subheader("Date Range Filters")
    time_field = st.sidebar.radio(
        "Filter Time By", ["Order Date", "Ship Date"], index=0, key="time_by"
    )
    enable_filter = st.sidebar.checkbox(
        "Enable date filter", value=True, key="enable_date_filter"
    )

    date_range = None
    if enable_filter and data is not None and time_field in data.columns:
        min_ts = data[time_field].min()
        max_ts = data[time_field].max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            min_date = (datetime.datetime.today() - timedelta(days=365 * 2)).date()
            max_date = datetime.datetime.today().date()
        else:
            min_date = min_ts.date()
            max_date = max_ts.date()
        default_start = max(min_date, max_date - timedelta(days=30))
        date_range = st.sidebar.date_input(
            f"Select {time_field} Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
    elif enable_filter:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(
                datetime.date.today(),
                datetime.date.today() + datetime.timedelta(days=1),
            ),
            key="date_range",
        )

    st.sidebar.markdown("---")

    # --- Categorical Filters ---
    if data is not None:
        region_opts = sorted(data["Region"].dropna().unique().tolist())
        region_sel = st.sidebar.multiselect(
            "Region", options=region_opts, default=region_opts, key="filter_region"
        )

        ship_opts = sorted(data["Ship Mode"].dropna().unique().tolist())
        ship_sel = st.sidebar.multiselect(
            "Ship Mode", options=ship_opts, default=ship_opts, key="filter_ship_mode"
        )

        st.sidebar.markdown("---")
        with st.sidebar.expander("Geographic Filters", expanded=False):
            country_opts = sorted(data["Country"].dropna().unique().tolist())
            country_sel = st.multiselect(
                "Country", options=country_opts, default=country_opts, key="filter_country"
            )

            if country_sel:
                state_opts = sorted(
                    data.loc[data["Country"].isin(country_sel), "State"
                ].dropna().unique().tolist()
                )
            else:
                state_opts = sorted(data["State"].dropna().unique().tolist())
            state_sel = st.multiselect(
                "State", options=state_opts, default=state_opts, key="filter_state"
            )

            if state_sel:
                city_opts = sorted(
                    data.loc[data["State"].isin(state_sel), "City"
                ].dropna().unique().tolist()
                )
            elif country_sel:
                city_opts = sorted(
                    data.loc[data["Country"].isin(country_sel), "City"
                ].dropna().unique().tolist()
                )
            else:
                city_opts = sorted(data["City"].dropna().unique().tolist())
            city_sel = st.multiselect(
                "City", options=city_opts, default=city_opts, key="filter_city"
            )
    else:
        region_sel = ship_sel = country_sel = state_sel = city_sel = []

    # Save to state
    st.session_state.active_filters["time_field"] = time_field
    st.session_state.active_filters["enable_date_filter"] = bool(enable_filter)
    st.session_state.active_filters["date_range"] = date_range
    st.session_state.active_filters["region"] = region_sel
    st.session_state.active_filters["ship_mode"] = ship_sel
    st.session_state.active_filters["country"] = country_sel
    st.session_state.active_filters["state"] = state_sel
    st.session_state.active_filters["city"] = city_sel

    return {
        "time_field": time_field,
        "date_range": date_range,
        "region": region_sel,
        "ship_mode": ship_sel,
        "country": country_sel,
        "state": state_sel,
        "city": city_sel,
    }


def render_tab_overview(data: pd.DataFrame, aggs: dict):
      """Tab 1: Overview with Waterfall, Lollipop, Line charts."""
      st.markdown("## Key Visualizations")

      # --- KPI Cards (3-Column) ---
      st.markdown("### Key Performance Indicators")
      c1, c2, c3 = st.columns(3)
      with c1:
          st.info("KPI Placeholder 1: Total Revenue")
      with c2:
          st.info("KPI Placeholder 2: Profit Margin (%)")
      with c3:
          st.info("KPI Placeholder 3: Unique Customers")

      # --- Row 1: Waterfall ---
      with st.expander("Waterfall Charts", expanded=True):
          st.info("Waterfall charts show average profit and profit margin by discount range.")
          if not aggs.get("profit_by_discount", pd.DataFrame()).empty:
              st.plotly_chart(
                  plot_waterfall(
                      aggs["profit_by_discount"], aggs["margin_by_discount"]
                  ),
                  use_container_width=True,
              )

      # --- Row 2: Lollipop ---
      with st.expander("Lollipop Charts", expanded=True):
          st.markdown("#### Filter by Category")
          # KPI Cards (2-Column)
          c1, c2 = st.columns(2)
          with c1:
              st.info("KPI Placeholder: Highest Margin Category")
          with c2:
              st.info("KPI Placeholder: Top Selling Category")

          # Local filter in mosaic layout
          f1, f2 = st.columns([2, 1])
          with f1:
              all_cats = sorted(data["Category"].dropna().unique().tolist())
              sel_cats = st.multiselect(
                  "Select Categories",
                  options=all_cats,
                  default=all_cats,
                  key="lollipop_cats",
              )
          with f2:
              st.caption("Detailed Category View")
              st.write(f"Showing **{len(sel_cats)}** of **{len(all_cats)}** categories.")
          if not aggs.get("margin_by_category", pd.DataFrame()).empty:
              st.plotly_chart(
                  plot_lollipop(aggs["margin_by_category"], sel_cats),
                  use_container_width=True,
              )

      # --- Row 3: Heatmap ---
      with st.expander("Profitability Heatmap", expanded=True):
          st.markdown("#### Filter by Sub-Category")
          # KPI Cards (2-Column)
          c1, c2 = st.columns(2)
          with c1:
              st.info("KPI Placeholder: Best Performing Sub-Category")
          with c2:
              st.info("KPI Placeholder: Slowest Growing Sub-Category")

          # Select All option in mosaic layout
          f1, f2 = st.columns([2, 1])
          with f1:
              all_subs = sorted(data["Sub-Category"].dropna().unique().tolist())
              if st.toggle("Select All Sub-Categories", value=True, key="line_select_all"):
                  sel_subs = all_subs
              else:
                  sel_subs = st.multiselect(
                      "Select Sub-Categories", options=all_subs, default=[], key="line_subs"
                  )
          with f2:
              top_n = st.number_input("Show Top N", min_value=1, value=5, key="line_top_n")

          if not aggs.get("margin_by_subcategory", pd.DataFrame()).empty:
              fig = plot_heatmap(aggs["margin_by_subcategory"], sel_subs, top_n if not st.session_state.get("line_select_all") else None)
              if fig:
                  st.plotly_chart(fig, use_container_width=True)


def render_tab_scatter(data: pd.DataFrame, aggs: dict):
    """Tab 2: Scatter Plots."""
    st.html("""
    
    <h2> Exploración de la Rentabilidad vs Volúmen de Ventas por Categoría y Subcategoría </h2>
    
    <small><i>Este dashboard permite explorar la relación entre la rentabilidad y el volumen de ventas de cada categoría y subcategoría 
    del minorista Superstore Giant. Se pueden aplicar filtros para analizar diferentes segmentos del negocio, mecanismos de envío, y datos geográficos
    de los clientes.</i></small>
    """)

    # --- Top KPI Cards: Profit Margin Analysis (3-Column) ---
    st.html("""
    <div style='display:flex; align-items:center; gap:10px; margin:5px; justify-content:center'>
        <h3>Resultados Generales del análisis de Márgenes de Ganancia</h3>
    </div>
    """)
    subcat_data = None

    if not aggs.get("by_subcategory", pd.DataFrame()).empty:
        subcat_data = aggs["by_subcategory"]

        # Best margin
        best_margin = subcat_data.loc[subcat_data["Profit_Margin_Pct"].idxmax()]
        # Average margin
        avg_margin = subcat_data["Profit_Margin_Pct"].mean()
        # Worst margin
        worst_margin = subcat_data.loc[subcat_data["Profit_Margin_Pct"].idxmin()]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="background-color: {COLOR_ATTENTION}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Ganancia Porcentual más Alta por Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">{best_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; color: white;"><strong>{best_margin['Sub_Category']}</strong></p>
                <p style="margin: 5px 0; color: white;">Categoría: {best_margin['Category']}</p>
                <p style="margin: 5px 0; color: white;">Ventas Totales: ${best_margin['Total_Sales']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background-color: {COLOR_GREY}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Ganancia Promedio sobre toda Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">{avg_margin:.2f}%</h2>
                <p style="margin: 5px 0; color: white;">Calculado sobre todas las Subcategorías</p>
                <p style="margin: 5px 0; color: white;">Total de Subcategorías: {len(subcat_data)}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background-color: {COLOR_RED}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Ganancia Porcentual más Baja por Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">{worst_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; color: white;"><strong>{worst_margin['Sub_Category']}</strong></p>
                <p style="margin: 5px 0; color: white;">Categoría: {worst_margin['Category']}</p>
                <p style="margin: 5px 0; color: white;">Ventas Totales: ${worst_margin['Total_Sales']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Second Row KPI Cards: Sales Volume Analysis (3-Column) ---
    st.html("""
    <div style='display:flex; align-items:center; gap:10px; margin:5px; justify-content:center'>
        <h3>Resultados Generales del análisis de Ventas Totales</h3>
    </div>
    """)

    if not aggs.get("by_subcategory", pd.DataFrame()).empty:
        # Highest sales
        highest_sales = subcat_data.loc[subcat_data["Total_Sales"].idxmax()]
        # Average sales
        avg_sales = subcat_data["Total_Sales"].mean()
        # Lowest sales
        lowest_sales = subcat_data.loc[subcat_data["Total_Sales"].idxmin()]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="background-color: {COLOR_ATTENTION}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Total de Ventas más Alto por Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">${highest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; color: white;"><strong>{highest_sales['Sub_Category']}</strong></p>
                <p style="margin: 5px 0; color: white;">Categoría: {highest_sales['Category']}</p>
                <p style="margin: 5px 0; color: white;">Margen de Ganancia: {highest_sales['Profit_Margin_Pct']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background-color: {COLOR_GREY}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Total de Ventas Promedio por Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">${avg_sales:,.2f}</h2>
                <p style="margin: 5px 0; color: white;">Calculado sobre todas las Subcategorías</p>
                <p style="margin: 5px 0; color: white;">Ganancia Total: ${subcat_data['Total_Sales'].sum():,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background-color: {COLOR_RED}; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="margin: 0; color: white;">Total de Ventas mas Bajo por Subcategoría</h4>
                <h2 style="margin: 10px 0; color: white;">${lowest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; color: white;"><strong>{lowest_sales['Sub_Category']}</strong></p>
                <p style="margin: 5px 0; color: white;">Categoría: {lowest_sales['Category']}</p>
                <p style="margin: 5px 0; color: white;">Margen de Ganancia: {lowest_sales['Profit_Margin_Pct']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # --- By Category ---
    st.html("""
    <h3>Exploración de Margen de Ganancia Porcentual vs Total de Ventas por Categoría</h3>
    """)

    cc1,cc2 = st.columns(2)

    with cc1:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            cat_data = aggs["by_category"]
            best_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmax()]
            st.markdown(f"""
            <div style="background-color: {COLOR_ATTENTION}; padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px">
                <p style="margin: 0; color: white;">La categoría más rentable del dataset corresponde a <strong>{best_cat['Category']}</strong> con un margen de <strong>{best_cat['Profit_Margin_Pct']:.2f}%</strong> y ventas totales de <strong>${best_cat['Total_Sales']:,.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    with cc2:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            cat_data = aggs["by_category"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.markdown(f"""
            <div style="background-color: {COLOR_WARNING}; padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px">
                <p style="margin: 0; color: white;">La categoría menos rentable del dataset corresponde a <strong>{worst_cat['Category']}</strong> con un margen de <strong>{worst_cat['Profit_Margin_Pct']:.2f}%</strong> y ventas totales de <strong>${worst_cat['Total_Sales']:,.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns(2)
    with f1:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            data = aggs.get("by_category", pd.DataFrame())
            min_sales = int(data["Total_Sales"].min())
            max_sales = int(data["Total_Sales"].max())
            min_sales = st.number_input(
                "Umbral de Ventas Totales Mínimas",
                value=0,
                key="scatter_cat_min_sales",
                min_value=0,
                max_value=max_sales,
                step=1000,
                format='%d',
                help="Filtro de ventas mínimas para inclusión de categorías en el gráfico"
            )
        else:
            min_sales = st.number_input("Umbral de Ventas Totales Mínimas", value=0, key="scatter_cat_min_sales", format='%d')
    with f2:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            data = aggs.get("by_category", pd.DataFrame())
            min_margin = int(data["Profit_Margin_Pct"].min()) *1.15
            max_margin = int(data["Profit_Margin_Pct"].max()) *1.15
            margin_range = st.slider(
                "Rango de Márgen de Ganancias ",
                min_value=-100, max_value=100,value=(-100,100), key="scatter_cat_margin",
                help="Rango de margen de ganancia para inclusión de categorías en el gráfico",

            )
        else:
            margin_range = st.slider("Rango de Márgen de Ganancias ", -100, 100, (-100, 100), key="scatter_cat_margin")

    # Chart
    if not aggs.get("by_category", pd.DataFrame()).empty:
        fig = plot_scatter_category(
            aggs["by_category"], min_sales, margin_range
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- By Sub-Category ---
    st.html("""
        <h3>Exploración de Margen de Ganancia Porcentual vs Total de Ventas por Subcategoría</h3>
        """)

    csub1, csub2 = st.columns(2)

    with csub1:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            cat_data = aggs["by_subcategory"]
            best_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmax()]
            st.markdown(f"""
                <div style="background-color: {COLOR_ATTENTION}; padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px">
                    <p style="margin: 0; color: white;">La subcategoría más rentable del dataset corresponde a <strong>{best_cat['Sub_Category']}</strong> con un margen de <strong>{best_cat['Profit_Margin_Pct']:.2f}%</strong> y ventas totales de <strong>${best_cat['Total_Sales']:,.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

    with csub2:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            cat_data = aggs["by_subcategory"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.markdown(f"""
                <div style="background-color: {COLOR_WARNING}; padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px">
                    <p style="margin: 0; color: white;">La subcategoría menos rentable del dataset corresponde a <strong>{worst_cat['Sub_Category']}</strong> con un margen de <strong>{worst_cat['Profit_Margin_Pct']:.2f}%</strong> y ventas totales de <strong>${worst_cat['Total_Sales']:,.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns(2)
    with f1:
        all_cats = sorted(data["Category"].dropna().unique().tolist())
        cat_filter = st.multiselect(
            "Filtrar por Categorías de Productos",
            options=all_cats,
            default=all_cats,
            help="Selección múltiple de categorías para incluir en el gráfico",
            key="scatter_subcat_cats"
        )
    with (f2):
        top_n = st.number_input(
            "Top N Subcategories por Ventas Totales",
            1,
            20,
            17,
            key="scatter_sub_top_n",
            help= "Mostrar las N subcategorías con mayores ventas totales"
        )

    f3, f4 = st.columns(2)
    with f3:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            data = aggs.get("by_subcategory", pd.DataFrame())
            min_sales = int(data["Total_Sales"].min())
            max_sales = int(data["Total_Sales"].max())
            min_sales_sub = st.number_input(
                "Umbral de Ventas Totales Mínimas",
                value=0,
                key="scatter_subcat_min_sales",
                min_value=0,
                max_value=max_sales,
                step=1000,
                format='%d',
                help="Filtro de ventas mínimas para inclusión de categorías en el gráfico"
            )
        else:
            min_sales_sub = st.number_input("Umbral de Ventas Totales Mínimas", value=0, key="scatter_subcat_min_sales",
                                        format='%d')
    with f4:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            data = aggs.get("by_subcategory", pd.DataFrame())
            min_margin = int(data["Profit_Margin_Pct"].min()) * 1.15
            max_margin = int(data["Profit_Margin_Pct"].max()) * 1.15
            margin_range_sub = st.slider(
                "Rango de Márgen de Ganancias ",
                min_value=-100, max_value=100, value=(-100, 100), key="scatter_subcat_margin",
                help="Rango de margen de ganancia para inclusión de categorías en el gráfico",

            )
        else:
            margin_range_sub = st.slider("Rango de Márgen de Ganancias ", -100, 100, (-100, 100), key="scatter_subcat_margin")

    # Chart
    if not aggs.get("by_subcategory", pd.DataFrame()).empty:
        fig = plot_scatter_subcategory(
            aggs["by_subcategory"], cat_filter, min_sales_sub, margin_range_sub, top_n
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_tab_data(data: pd.DataFrame, aggs: dict):
    """Tab 3: Data View."""
    st.markdown("## Raw Data")

    # Top: Full dataset
    st.subheader("Full Dataset")
    st.dataframe(data, use_container_width=True)

    st.subheader("Aggregated Datasets Used")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Profit by Discount")
        st.dataframe(aggs.get("profit_by_discount", pd.DataFrame()))
    with c2:
        st.markdown("#### Margin by Category")
        st.dataframe(aggs.get("margin_by_category", pd.DataFrame()))
    with c3:
        st.markdown("#### By Category (Scatter)")
        st.dataframe(aggs.get("by_category", pd.DataFrame()))

    st.markdown("#### By Sub-Category (Scatter)")
    st.dataframe(aggs.get("by_subcategory", pd.DataFrame()))


def main():
    st.set_page_config(page_title="Superstore Dashboard", layout="wide")
    data = initialize_dashboard()
    filters = render_sidebar(data)

    # Apply filters
    filtered = apply_filters(
        data,
        filters["date_range"],
        filters["time_field"],
        filters["region"],
        filters["ship_mode"],
        filters["country"],
        filters["state"],
        filters["city"],
    )

    # Compute aggregations on filtered data
    aggregations = get_aggregations(filtered)

    # Tabbed layout
    tabs = st.tabs(["Overview", "Scatter Plots", "Data"])

    with tabs[0]:
        render_tab_overview(filtered, aggregations)
    with tabs[1]:
        render_tab_scatter(filtered, aggregations)
    with tabs[2]:
        render_tab_data(filtered, aggregations)


if __name__ == "__main__":
    main()