import datetime
import json
import math
from datetime import timedelta
from logging import getLogger
from typing import Any, Iterable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.pyplot import ticklabel_format
from plotly.subplots import make_subplots

# =============================================================================
# Data Preparation & Constants
# =============================================================================

COLOR_POSITIVE = "#4A678C"
COLOR_NEGATIVE = "#731A06"
COLOR_WARNING = '#D78019'
COLOR_ATTENTION = '#0C1726'
COLOR_BG = "#F2F2F4"
COLOR_LINE = "#5A5A5C"
COLOR_GREEN_DARK = "#253B59"
COLOR_GREEN_LIGHT = "#4A678C"
COLOR_GREEN = "#6E8C03"
COLOR_RED = "#E01518"
COLOR_GREY = "#A4A4A4"
WHITE = "#FFFFFF"

CATEGORY_COLORS = {
    "Furniture": "#636EFA",      # Blue
    "Office Supplies": "#FFA15A", # Orange
    "Technology": "#00CC96",      # Green
}

DISCOUNT_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
DISCOUNT_LABELS = [
    "0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0"
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


def _apply_standard_theme(fig, title, subtitle):
    """Unified styling for all dashboard graphs."""
    full_title = f"<b>{title}</b><br><i><span style='font-size:12px;font-weight:200;color:#666'>{subtitle}</span></i>"
    
    fig.update_layout(
        title={
            "text": full_title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.92,
            "font": {"size": 16, "family": "Arial"},
        },
        autosize=False,
        width=1200,
        height=700,
        margin=dict(
            l=150,
            r=150,
            b=50,
            t=150,
            pad=4
        ),
    )

    return fig


def _apply_x_axis_customization(figure, x_label=None, x_ticks=None, x_tick_labels=None, tick_angle=30,
                                x_axis_style=None,  x_axis_label_distance: int=-0.12,row_id=None, column_id=None):
    """Apply x-axis customization to figure or subplot.

    Args:
        figure: Plotly figure object
        x_label: Label text for x-axis
        x_ticks: List of tick values
        x_tick_labels: List of tick labels (derived from x_ticks if not provided)
        tick_angle: Rotation angle for tick labels (default: 30)
        x_axis_style: Dictionary with x-axis style configuration
        row_id: Row index for subplot (1-indexed)
        column_id: Column index for subplot (1-indexed)
    """
    # Default x-axis style
    default_style = {
        "showgrid": False,
        "title": "",
        "showline": True,
        "linewidth": 2,
        "linecolor": COLOR_LINE,
        "tickangle": tick_angle,
        "ticks": "outside",
        "tickwidth": 2,
        "tickcolor": COLOR_LINE,
    }

    # Use provided style or default
    style = x_axis_style if x_axis_style else default_style

    # Add tick values and labels if provided
    if x_ticks is not None:
        style["tickvals"] = x_ticks
        style["ticktext"] = x_tick_labels if x_tick_labels else [str(t) for t in x_ticks]

    # Apply to specific subplot or entire figure
    if row_id is not None and column_id is not None:
        figure.update_xaxes(style, row=row_id, col=column_id)
    else:
        figure.update_xaxes(style)

    # Add x-axis label annotation if provided
    if x_label:
        annotation_config = {
            "xref": "paper", "yref": "y domain",
            "x": 0,
            "y": x_axis_label_distance,
            "text": x_label,
            "showarrow": False,
            "xanchor": "left",
            "yanchor":'bottom',
            "font": dict(size=12, color=COLOR_LINE, family="Arial Black")
        }

        # For subplots, adjust references
        if row_id is not None and column_id is not None:
            annotation_config["xref"] = f"x{column_id} domain" if column_id > 1 else "x domain"
            annotation_config["yref"] = f"y{row_id} domain" if row_id > 1 else "y domain"

        figure.add_annotation(**annotation_config)

    return figure


def _apply_y_axis_customization(figure,
                                y_label=None,
                                y_ticks=None,
                                y_tick_labels=None,
                                y_axis_style=None,
                                y_tick_style=None,
                                y_axis_lable_distance:float=-0.12,
                                row_id=None,
                                column_id=None):
    """Apply y-axis customization to figure or subplot.
    
    Args:
        figure: Plotly figure object
        y_label: Label text for y-axis
        y_ticks: List of tick values
        y_tick_labels: List of tick labels (derived from y_ticks if not provided)
        y_axis_style: Dictionary with y-axis style configuration
        row_id: Row index for subplot (1-indexed)
        column_id: Column index for subplot (1-indexed)
    """
    # Default y-axis style
    default_style = {
        "showgrid": False,
        "title": "",
        "showline": True,
        "linewidth": 2,
        "linecolor": COLOR_LINE,
        "ticks": "outside",
        "tickwidth": 2,
        "tickcolor": COLOR_LINE,
    }
    
    # Use provided style or default
    style = y_axis_style if y_axis_style else default_style
    
    # Add tick values and labels if provided
    if y_ticks is not None:
        style["tickvals"] = y_ticks
        style["ticktext"] = y_tick_labels if y_tick_labels else [str(t) for t in y_ticks]
    
    # Apply to specific subplot or entire figure
    if row_id is not None and column_id is not None:
        figure.update_yaxes(style, row=row_id, col=column_id)
    else:
        figure.update_yaxes(style)
    
    # Add y-axis label annotation if provided
    if y_label:
        annotation_config = {
            "xref": "x domain",
            "yref": "paper",
            "x": y_axis_lable_distance,
            "y": 0,
            "text": y_label,
            "showarrow": False,
            "textangle": -90,
            "yanchor": "bottom",
            "font": dict(size=12, color=COLOR_LINE, family="Arial Black")
        }
        
        # For subplots, adjust references
        if row_id is not None and column_id is not None:
            annotation_config["xref"] = f"x{column_id} domain" if column_id > 1 else "x domain"
            annotation_config["yref"] = f"y{row_id} domain" if row_id > 1 else "y domain"
        
        figure.add_annotation(**annotation_config)

    if y_tick_style:
        figure.update_layout(yaxis=y_tick_style,overwrite=False)


    return figure

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


def _return_color_for_cell_based_on_values(
        profit_margin_to_measure,
        sales_to_measure,
        avg_margin,
        avg_sales
):
    if profit_margin_to_measure > avg_margin and sales_to_measure > avg_sales:
        return COLOR_GREEN  # High margin, high sales - optimal
    elif profit_margin_to_measure < avg_margin and sales_to_measure < avg_sales:
        return COLOR_RED  # Low margin, low sales - poor performance
    elif profit_margin_to_measure < avg_margin and sales_to_measure > avg_sales:
        return COLOR_WARNING  # High sales but low margin - needs attention
    else:  # row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] < avg_sales
        return COLOR_ATTENTION  # High margin but low sales - growth opportunity


def _assign_color(row, avg_margin, avg_sales):
    if row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] > avg_sales:
        return COLOR_GREEN  # High margin, high sales - optimal
    elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] < avg_sales:
        return COLOR_RED  # Low margin, low sales - poor performance
    elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] > avg_sales:
        return COLOR_WARNING  # High sales but low margin - needs attention
    else:  # row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] < avg_sales
        return COLOR_ATTENTION  # High margin but low sales - growth opportunity

def _add_color_labels_for_scatter_segmentations(fig):
    # Add color legend box in top right corner
    legend_x_start = 0.05
    legend_y_start = 1.02
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

    return fig
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

    # Price-Volume-Profit by Category (bubble chart)
    agg_price_volume = df.groupby('Category').agg({
        'Sales': ['sum', 'mean'],
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    agg_price_volume.columns = ['Category', 'Total_Revenue', 'Avg_Price', 'Total_Profit', 'Transaction_Count']
    agg_price_volume['Profit_Margin_Pct'] = (agg_price_volume['Total_Profit'] / agg_price_volume['Total_Revenue']) * 100

    # Customer Purchase by Discount & Category (line chart)
    agg_customer_purchase = df.groupby(['Discount_Bin', 'Category'], observed=True).agg({
        'Sales': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    agg_customer_purchase.columns = ['Discount_Bin', 'Category', 'Total_Sales', 'Unique_Customers']
    agg_customer_purchase['Avg_Purchase_Per_Customer'] = (
        agg_customer_purchase['Total_Sales'] / agg_customer_purchase['Unique_Customers']
    )

    # Customer Value by Category (for pricing KPIs)
    agg_customer_value = df.groupby('Category').agg({
        'Sales': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    agg_customer_value.columns = ['Category', 'Total_Sales', 'Unique_Customers']
    agg_customer_value['Avg_Customer_Value'] = agg_customer_value['Total_Sales'] / agg_customer_value['Unique_Customers']

    return {
        "profit_by_discount": agg_profit_by_discount,
        "margin_by_discount": agg_margin_by_discount,
        "margin_by_category": agg_margin_by_category,
        "margin_by_subcategory": agg_margin_by_subcategory,
        "by_category": agg_by_category,
        "by_subcategory": agg_by_subcategory,
        "price_volume_by_category": agg_price_volume,
        "customer_purchase_by_discount": agg_customer_purchase,
        "customer_value_by_category": agg_customer_value,
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
            "<i style='font-size:10pt'>Ganancia Promedio por <b>Rango de Descuento</b> Aplicado</i>",
            "<i style='font-size:10pt'>Margen de Ganancia Porcentual Promedio por <b>Rango de Descuento</b> Aplicado</i>"
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
        autosize=False,
        showlegend=False,
        height=900,
        width=1200
    )

    #? Para definir la label definitiva, y responsive extraemos el ultimo valor positivo antes de caer en la rentabilidad
    #? y extraemos su idxmax() para sacar la bin a la que pertenece
    max_discount_label = agg_profit[agg_profit["Avg_Profit"] > 0]["Discount_Bin"].iloc[-1]

    _apply_standard_theme(
        fig,
        "Comparativa de Ganancia y Margen de Ganancia Promedio por cada Intervalo de Descuentos Aplicado",
f"Durante el periodo de análisis, las ventas generan ganancias y rentabilidad para la "
f"empresa mientras el descuento se mantenga por <b>debajo o hasta</b> un descuento en el rango máximo de <b>{float(max_discount_label.split('-')[0]) * 100:.0f} al {
float(max_discount_label.split('-')[1]) * 100:.0f}%</b>")

    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", row_id=1, column_id=1, x_axis_label_distance=-0.2)
    _apply_y_axis_customization(fig, y_label="Ganancia Promedio ($)", row_id=1, column_id=1, y_axis_style={
        'showgrid': False,
        'title': '',
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': '.0f',
        'tickprefix': '$'
    })
    

    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", row_id=1, column_id=2, x_axis_label_distance=-0.2)
    _apply_y_axis_customization(fig, y_label="Margen de Ganancia (%)", row_id=1, column_id=2, y_axis_style={
        'showgrid': False,
        'title': '',
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': '.0f',
        'ticksuffix': '%'
    })

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
                    size=40,
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
        "Margen de Ganancia por cada Categoría del Inventario",
        "Durante el periodo 2014-2017, todas las categorías comerciales de Superstore muestran un decrecimiento<br>del porcentaje de ganancia a medida que el descuento supera el 30%."
    )
    fig.update_layout(
        hovermode="closest",
        legend=dict(
            title="Categoría",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemdoubleclick=False,
            itemclick=False,
        ),
    )

    _apply_x_axis_customization(fig, x_label="Rango de Descuentos",x_tick_labels=DISCOUNT_LABELS, x_ticks=list(range(len(DISCOUNT_LABELS))), tick_angle=0)
    _apply_y_axis_customization(fig, y_label="Margen de Ganancia Promedio (%)", y_axis_style={
        'showgrid': False,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': '.0f',
        'ticksuffix': '%'
    }, y_axis_lable_distance=-0.07)
    return fig


# =============================================================================
# Charts - Heatmap
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


    text_array = []
    for i, row in enumerate(pivot_df.values):
        text_row = []
        for j, val in enumerate(row):
            if pd.isna(val):
                text_row.append("")  # Empty string for NaN
            else:
                text_row.append(f"{val:.0f}%")
        text_array.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Margin %"),
        hovertemplate="<b>%{y}</b><br>Discount: %{x}<br>Margin: %{z:.2f}%<extra></extra>",
        xgap=1,
        ygap=1,
        text=text_array,
        texttemplate="%{text}",
        showscale=True
    ))

    _apply_standard_theme(
        fig,
        "Tabulación del Margen de Ganancia por Subcategoría segmentado por Rango de Descuento",
        "Las 17 subcategorías de la compañia muestran una tendencia similar donde un descuento superior al 30% invierte<br> las ganancias porcentuales y resigna a pérdidas para cada venta realizada",
    )
    
    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", x_tick_labels=DISCOUNT_LABELS, x_ticks=list(range(len(DISCOUNT_LABELS))), tick_angle=0)
    _apply_y_axis_customization(fig, y_label="Sub-Categorías", y_axis_lable_distance=-0.1)
    return fig


# =============================================================================
# Charts - Pricing Strategy
# =============================================================================

def plot_price_volume_profit_bubble(agg_price_volume):
    """Bubble chart: Price-Volume-Profit by Category."""
    avg_margin = agg_price_volume['Profit_Margin_Pct'].mean()
    avg_price = agg_price_volume['Avg_Price'].mean()
    agg_price_volume = agg_price_volume.copy()
    agg_price_volume["Color"] = agg_price_volume.apply(func=lambda row: _return_color_for_cell_based_on_values(row['Profit_Margin_Pct'], row['Avg_Price'], avg_margin, avg_price),axis=1)

    fig = go.Figure()

    # ? Anadimos lineas de referencia por metrica
    avg_price = agg_price_volume['Avg_Price'].mean()
    fig.add_vline(x=avg_price,
                  line_dash="dash",
                  line_color=COLOR_LINE,
                  line_width=1,
                  opacity=0.5,
                  annotation_text=f"Precio Promedio: ${avg_price:,.0f}",
                   annotation_position="top",
                   annotation=dict(
                       font=dict(size=10, family="Arial", style='italic'),
                       bordercolor=COLOR_BG,
                       borderwidth=1,
                       borderpad=4,
                       y=0.95
                   )
    )
    avg_margin = agg_price_volume['Profit_Margin_Pct'].mean()
    fig.add_hline(
        y=avg_margin,
        line_dash="dash",
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Margen Promedio: {avg_margin:.1f}%",
        annotation_position="left",
        annotation= dict(
            font=dict(size=10, family="Arial", style='italic'),
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.1
        )
    )

    for idx, row in agg_price_volume.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Avg_Price']],
            y=[row['Profit_Margin_Pct']],
            mode='markers+text',
            marker=dict(
                size=math.log2(row['Total_Revenue']),
                color=row["Color"],
                line=dict(color=COLOR_LINE, width=1),
                opacity=0.7
            ),
            text=row['Category'],
            textposition='top center',
            textfont=dict(size=12, family="Arial Black"),
            name=row['Category'],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Category']}</b><br><br>"
                f"<b>Precio Promedio</b>: ${row['Avg_Price']:,.2f}<br>"
                f"<b>Margen de Utilidad</b>: {row['Profit_Margin_Pct']:.2f}%<br>"
                f"<b>Ingresos Totales</b>: ${row['Total_Revenue']:,.2f}<br>"
                f"<b>Transacciones</b>: {row['Transaction_Count']}<extra></extra>"
            )
        ))



    max_cat_by_average_sale = agg_price_volume.loc[agg_price_volume['Avg_Price'].idxmax(), 'Category']
    max_cat_by_average_sale_second_place = agg_price_volume.loc[agg_price_volume['Avg_Price'].nlargest(2).index[-1], 'Category']
    max_cat_by_average_sale_value = agg_price_volume.loc[agg_price_volume['Avg_Price'].idxmax(), 'Avg_Price']
    max_cat_by_margin = agg_price_volume.loc[agg_price_volume['Profit_Margin_Pct'].idxmax(), 'Category']
    max_cat_by_margin_value = agg_price_volume.loc[agg_price_volume['Profit_Margin_Pct'].idxmax(), 'Profit_Margin_Pct']

    _apply_standard_theme(
        fig,
        "Análisis Precio-Volumen-Utilidad por Categoría",
        f"Durante el periodo de análisis, <b>{max_cat_by_average_sale}</b> se consolidó como la categoría con el <b>Ticket Promedio Más Alto</b> con un ticket de <b>$ {max_cat_by_average_sale_value:,.2f} </b> seguido de <b>{max_cat_by_average_sale_second_place}</b>."
        f"<br>En términos de Margen de Ganancia <b>{max_cat_by_margin}</b> con <b>{max_cat_by_margin_value:.2f}%</b> es "
        f"la categoría más altas, reflejando que <b>{max_cat_by_margin}</b> es una categoría fuerte en ventas y en ingresos por venta."
    )
    
    _apply_x_axis_customization(fig, x_label="Precio Promedio ($)", x_axis_style={
        'showgrid': False,
        'gridcolor': 'lightgray',
        'gridwidth': 0.5,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'tickformat': ',.0f',
        'tickprefix': '$'
    }, x_axis_label_distance=-0.1)
    
    _apply_y_axis_customization(fig, y_label="Margen de Utilidad (%)", y_axis_style={
        'showgrid': False,
        'gridcolor': 'lightgray',
        'gridwidth': 0.5,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': '.0f',
        'ticksuffix': '%'
    }, y_axis_lable_distance=-0.07)
    
    fig.update_layout(
        hovermode='closest',
    )
    
    return fig


def plot_avg_customer_purchase_by_discount(agg_customer_purchase):
    """Line chart: Average Customer Purchase Value by Category across Discount Bins."""
    fig = go.Figure()
    
    for category in sorted(agg_customer_purchase['Category'].unique()):
        cat_data = agg_customer_purchase[agg_customer_purchase['Category'] == category]
        
        fig.add_trace(go.Scatter(
            x=cat_data['Discount_Bin'].astype(str),
            y=cat_data['Avg_Purchase_Per_Customer'],
            mode='lines+markers',
            name=category,
            line=dict(
                color=CATEGORY_COLORS.get(category, COLOR_GREY),
                width=1,
                dash='dash'
            ),
            marker=dict(
                size=10,
                color=CATEGORY_COLORS.get(category, COLOR_GREY),
                line=dict(color='white', width=2)
            ),
            hovertemplate=(
                f"<b>{category}</b><br><br>"
                "<b>Rango de Descuento</b>: %{x}<br>"
                "<b>Compra Promedio</b>: $%{y:,.2f}<br>"
                "<extra></extra>"
            )
        ))
    
    _apply_standard_theme(
        fig,
        "Valor Promedio de Compra por Cliente según Categoría y Rango de Descuento",
        "Durante el periodo 2014-2017, <b>Technology, Furniture y Office Supplies</b> muestran un <b>Valor de Compra Promedio</b> similar, demostrando una tendencia que coincide con el comportamiento general de las ganancias porcentuales por categoría<br>"
        "donde la ganancia se mantiene positiva antes de los descuentos mayores al 20%, demostrando rentabilidad y valor comercial para las categorías en base a descuentos menores."
    )
    
    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", tick_angle=30, x_axis_label_distance=-0.15)
    _apply_y_axis_customization(fig, y_label="Compra Promedio por Cliente ($)", y_axis_style={
        'showgrid': False,
        'gridcolor': 'lightgray',
        'gridwidth': 0.5,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'tickformat': ',.0f',
        'tickprefix': '$'
    }, y_axis_lable_distance=-0.07)
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(
            title="Categoría",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemdoubleclick=False,
            itemclick=False
        )
    )

    fig.add_annotation(
        text="Dependiendo del selector de fechas, algúnas de las líneas no tendran<br>registros para todos los rangos de descuentos",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align='left',
        font=dict(size=11, family="Arial Black"),
        bordercolor=COLOR_BG,
        borderwidth=1,
        borderpad=4,
        opacity=0.7)
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



    data["Color"] = data.apply(func= lambda row: _assign_color(row, avg_margin, avg_sales), axis=1)

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
            font=dict(size=10, family="Arial",style='italic'),
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.15
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
            font=dict(size=10, family="Arial", style='italic'),
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )

    _add_color_labels_for_scatter_segmentations(fig)

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
                textfont=dict(size=12, family="Arial Black"),
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
        f"Durante el periodo de análisis, <b>{data.loc[data['Profit_Margin_Pct'].idxmax(), 'Category']}</b> "
        f"fue la categoría con las mayores ventas de <b>${data.loc[data['Profit_Margin_Pct'].idxmax(), 'Total_Sales']:,.0f}</b> y el mejor margen ganancia promedio de <b>{data.loc[data['Profit_Margin_Pct'].idxmax(), 'Profit_Margin_Pct'].max():.2f}%</b>"
    )

    _apply_x_axis_customization(fig, x_label="Ventas Totales ($)", x_axis_style={
        'showgrid': False,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': ',.0f',
        'title_standoff': 15
    }, x_axis_label_distance=-0.06)
    _apply_y_axis_customization(fig, y_label='Margen de Ganancia Promedio (%)', y_axis_style={
        'showgrid': False,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': ',.0f',
        'title_standoff': 15}, y_axis_lable_distance=-0.05)

    fig.update_layout(
        hovermode="closest",
        height=1000,
        showlegend=True,
        xaxis={
            'zeroline':False,
            'tickformat':',.0f',
            'tickprefix':'$'
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



    agg_data_filtered["Color"] = agg_data_filtered.apply(func = lambda row: _assign_color(row,avg_margin, avg_sales), axis=1)


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
            font=dict(size=10, family="Arial"),
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
            font=dict(size=10, family="Arial"),
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )

    _add_color_labels_for_scatter_segmentations(fig)

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
                textfont=dict(size=11),
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
        "Comparativo de Rentabilidad en base al Porcentaje de Ganancia vs Ventas Totales por Subcategoría",
        f"Durante el periodo de análisis, la subcategoría <b>{agg_data_filtered.loc[agg_data_filtered['Profit_Margin_Pct'].idxmax()]['Sub_Category']}</b> "
        f"dentro de la categoría <b>{agg_data_filtered.loc[agg_data_filtered['Total_Sales'].idxmax()]['Category']}</b> "
        f"muestra un desempeño óptimo con ventas totales de ${agg_data_filtered['Total_Sales'].max():,.0f} "
        f"y un margen de ganancia promedio del {agg_data_filtered['Profit_Margin_Pct'].max():.2f}%."
    )

    _apply_x_axis_customization(fig, x_label="Ventas Totales ($)", x_axis_style={
        'showgrid': False,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': ',.0f',
        'title_standoff': 15
    }, x_axis_label_distance=-0.07)
    _apply_y_axis_customization(fig, y_label='Margen de Ganancia (%)', y_axis_style={
        'showgrid': False,
        'showline': True,
        'linewidth': 2,
        'linecolor': COLOR_LINE,
        'ticks': 'outside',
        'tickwidth': 2,
        'tickcolor': COLOR_LINE,
        'zeroline': True,
        'tickformat': ',.0f',
        'title_standoff': 15
    }, y_axis_lable_distance=-0.05)
    fig.update_layout(
        height=1000,
        hovermode="closest",
        showlegend=True,
        xaxis={'zeroline': False, 'tickformat':',.0f','tickprefix':'$'},
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
        st.session_state.selected_view = "Visión Rentabilidad por Descuentos"
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
    st.sidebar.title("Controles de Filtros Generales")
    st.sidebar.markdown("Filtros generales para toda la dashboardoard. Estos filtros afectan a todas las visualizaciones y permiten segmentar los datos por tiempo, región, mecanismo de envío, y ubicación geográfica del cliente.")

    # --- Global Filters ---
    st.sidebar.subheader("Filtros Temporales")
    time_field = st.sidebar.radio(
        "Filtrar Fechas Mediante", ["Order Date", "Ship Date"], index=0, key="time_by"
    )
    enable_filter = st.sidebar.checkbox(
        "Activar/Desactivar Filtro de Fechas", value=True, key="enable_date_filter"
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
            f"Seleccione un rango de {time_field}",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
    elif enable_filter:
        date_range = st.sidebar.date_input(
            "Seleccione un rango de fechas",
            value=(
                datetime.date.today(),
                datetime.date.today() + datetime.timedelta(days=1),
            ),
            key="date_range",
        )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Filtros Categóricos")

    # --- Categorical Filters ---
    if data is not None:
        region_opts = sorted(data["Region"].dropna().unique().tolist())
        region_sel = st.sidebar.multiselect(
            "Región", options=region_opts, default=region_opts, key="filter_region"
        )

        ship_opts = sorted(data["Ship Mode"].dropna().unique().tolist())
        ship_sel = st.sidebar.multiselect(
            "Modo de Envio", options=ship_opts, default=ship_opts, key="filter_ship_mode"
        )

        st.sidebar.markdown("---")
        with st.sidebar.expander("Filtros Geográficos", expanded=False):
            country_opts = sorted(data["Country"].dropna().unique().tolist())
            country_sel = st.multiselect(
                "País del Registro", options=country_opts, default=country_opts, key="filter_country"
            )

            if country_sel:
                state_opts = sorted(
                    data.loc[data["Country"].isin(country_sel), "State"
                ].dropna().unique().tolist()
                )
            else:
                state_opts = sorted(data["State"].dropna().unique().tolist())
            state_sel = st.multiselect(
                "Estado del Registro (Si Aplica)", options=state_opts, default=state_opts, key="filter_state"
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
                "Ciudad del Registro", options=city_opts, default=city_opts, key="filter_city"
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
      st.html(
        """
        <h2> Exploración de la Rentabilidad por Descuentos Aplicados </h2>
        <small><i>Este dashboard permite explorar la relación entre la rentabilidad y el porcentaje de descuento categoría y subcategoría 
        del minorista Superstore Giant. Se pueden aplicar filtros para analizar diferentes segmentos del negocio, mecanismos de envío, y datos geográficos
        de los clientes.</i></small>
        """
      )

      # --- KPI Cards (2 rows layout) ---
      # First row: 2 KPI cards
      c1, c2 = st.columns(2)
      with c1:
          st.html(f"""
          <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">GANANCIA POR RANGO DE DESCUENTO</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">Límite Estratégico: 30%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Umbral Crítico de Rentabilidad</p>
                <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    La rentabilidad operativa experimenta una <b>contracción acelerada</b> al superar el 30% de descuento, erosionando el margen neto.
                </p>
            </div>
          """)
      with c2:
          # Calculate profit margin stats
          avg_margin = aggs.get("by_subcategory", pd.DataFrame())
          if not avg_margin.empty:
              overall_margin = avg_margin["Profit_Margin_Pct"].mean()
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_WARNING}, #f4c060); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">MARGEN DE GANANCIA PROMEDIO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{overall_margin:.2f}%</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Benchmark de Rendimiento Global</p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        El rendimiento porcentual promedio se mantiene saludable bajo una política de descuentos controlada y segmentada. Pero aplicado descuentos altos, el márgen es negativo.
                    </p>
                </div>
              """)
      
      # Second row: 3 KPI cards
      c3, c4, c5 = st.columns(3)
      with c3:
          # Calculate unique customers
          unique_customers = data["Customer ID"].nunique()
          total_orders = len(data)
          st.html(f"""
          <div style="background: linear-gradient(135deg, {COLOR_GREEN}, #89ae04); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">CLIENTES ÚNICOS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{unique_customers:,}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Total de órdenes: <b>{total_orders:,}</b></p>
                <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Representa una base consolidada con un promedio de <b>{total_orders/unique_customers:.1f} órdenes</b> por cliente.
                </p>
            </div>
          """)
      with c4:
          # Calculate average order value
          avg_order_value = data["Sales"].mean()
          total_sales = data["Sales"].sum()
          st.html(f"""
          <div style="background: linear-gradient(135deg, {COLOR_GREY}, #c0c0c0); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VALOR PROMEDIO POR ORDEN</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_order_value:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ventas totales: <b>${total_sales:,.2f}</b></p>
                <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Este indicador refleja el <b>ticket promedio</b> por cada transacción procesada en el sistema.
                </p>
            </div>
          """)
      with c5:
          # Calculate total profit
          total_profit = data["Profit"].sum()
          profit_margin_overall = (total_profit / total_sales * 100) if total_sales > 0 else 0
          st.html(f"""
          <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">GANANCIA TOTAL ACUMULADA</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${total_profit:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Neto: <b>{profit_margin_overall:.2f}%</b></p>
                <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Resultado financiero neto consolidado durante el periodo de análisis seleccionado.
                </p>
            </div>
          """)

      # --- Waterfall ---
      st.markdown("---")
      st.html("""
      <h3>Exploración de la tendencia general de la Ganancia Promedio y el Margen de Ganancia Porcentual por Rango de Descuento</h3>
      """)
      if not aggs.get("profit_by_discount", pd.DataFrame()).empty:
          st.plotly_chart(
              plot_waterfall(
                  aggs["profit_by_discount"], aggs["margin_by_discount"]
              ),
              use_container_width=True,
              config={
                  "displaylogo": False,
                  "displayModeBar": False}
          )

      # --- Lollipop ---
      st.markdown("---")
      st.html("""
      <h3>Desglose del Margen de Ganancia Promedio por Categoría</h3>
      """)
      # KPI Cards (2-Column)
      c1, c2 = st.columns(2)
      with c1:
          # Highest margin category
          margin_by_cat = aggs.get("margin_by_category", pd.DataFrame())
          if not margin_by_cat.empty:
              best_cat = margin_by_cat.groupby("Category")["Avg_Profit_Margin_Pct"].mean().idxmax()
              best_margin = margin_by_cat.groupby("Category")["Avg_Profit_Margin_Pct"].mean().max()
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_GREEN}, #89ae04); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER DE RENTABILIDAD POR CATEGORÍA</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{best_cat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{best_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        La categoría <b>{best_cat}</b> demuestra la mayor eficiencia en la conversión de ventas a beneficios netos.
                    </p>
                </div>
              """)
      with c2:
          # Top selling category by total sales
          by_cat = aggs.get("by_category", pd.DataFrame())
          if not by_cat.empty:
              top_cat = by_cat.loc[by_cat["Total_Sales"].idxmax(), "Category"]
              top_sales = by_cat["Total_Sales"].max()
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER EN VOLUMEN DE VENTAS</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{top_cat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ingresos Totales: <b>${top_sales:,.2f}</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        <b>{top_cat}</b> se consolida como el principal generador de flujo de caja para la organización.
                    </p>
                </div>
              """)

      # Local filter in mosaic layout
      f1, f2 = st.columns([2, 1])
      with f1:
          all_cats = sorted(data["Category"].dropna().unique().tolist())
          sel_cats = st.multiselect(
              "Seleccione las Categorías a Mostrar",
              options=all_cats,
              default=all_cats,
              key="lollipop_cats",
              help="Permite definir una o varias categorías a mostrar en el gráfico exploratorio del márgen de ganancia por nivel de descuento."
          )
      with f2:
          st.caption("Vista de Margen de Ganancia por Categoría")
          st.write(f"Mostrando **{len(sel_cats)}** de **{len(all_cats)}** categorías.")
      if not aggs.get("margin_by_category", pd.DataFrame()).empty:
          st.plotly_chart(
              plot_lollipop(aggs["margin_by_category"], sel_cats),
              use_container_width=True,
              config={
                  "displaylogo": False,
                  "displayModeBar": False}
          )

      # --- Heatmap ---
      st.markdown("---")
      st.html(
          """
          <h3>Exploración del Margen de Ganancia Promedio por Subcategoría</h3>
          """
      )
      # KPI Cards (2-Column)
      c1, c2 = st.columns(2)
      with c1:
          # Best performing sub-category
          margin_by_subcat = aggs.get("by_subcategory", pd.DataFrame())
          if not aggs.get("by_subcategory", pd.DataFrame()).empty:
              margin_by_subcat = aggs["by_subcategory"]
              best_subcat = margin_by_subcat.loc[margin_by_subcat["Profit_Margin_Pct"].idxmax()]["Sub_Category"]
              best_subcat_margin = margin_by_subcat.loc[margin_by_subcat["Profit_Margin_Pct"].idxmax()]["Profit_Margin_Pct"]
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_GREEN}, #89ae04); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CON MEJOR DESEMPEÑO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{best_subcat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{best_subcat_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        Identificada como la unidad de negocio más rentable dentro de su respectiva categoría.
                    </p>
                </div>
              """)
      with c2:
          # Worst performing sub-category
          if not margin_by_subcat.empty:
              worst_subcat = margin_by_subcat.loc[margin_by_subcat["Profit_Margin_Pct"].idxmin()]["Sub_Category"]
              worst_subcat_margin = margin_by_subcat.loc[margin_by_subcat["Profit_Margin_Pct"].idxmin()][
                  "Profit_Margin_Pct"]
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_RED}, #ff4d4d); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CRÍTICA (BAJO MARGEN)</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{worst_subcat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{worst_subcat_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        Requiere <b>atención inmediata</b> y revisión de la estructura de costos o estrategia de precios.
                    </p>
                </div>
              """)

      # Select All option in mosaic layout
      f1, f2 = st.columns([2, 1])
      with f1:
          all_subs = sorted(data["Sub-Category"].dropna().unique().tolist())
          if st.toggle("Mostrar todas las Subcategorías Comerciales", value=True, key="line_select_all", help="Permite definir una o más subcategorías para explorar la tendencia de su margen de ganancia por nivel de descuento."):
              sel_subs = all_subs
          else:
              sel_subs = st.multiselect(
                  "Selecccionar Subcategorías a Mostrar", options=all_subs, default=all_subs, key="line_subs"
              )
      with f2:
          top_n = st.number_input("Mostrar Top N Subcategorías", min_value=1, value=5, key="line_top_n", help="Permite definir el número de subcategorías a mostrar en el gráfico exploratorio del margen de ganancia por nivel de descuento. Si se activa la opción 'Mostrar todas las Subcategorías Comerciales', este filtro se deshabilita automáticamente.")

      if not aggs.get("margin_by_subcategory", pd.DataFrame()).empty:
          fig = plot_heatmap(aggs["margin_by_subcategory"], sel_subs, top_n if not st.session_state.get("line_select_all") else None)
          if fig:
              st.plotly_chart(fig, use_container_width=True, config={
                  "displaylogo": False,
                  "displayModeBar": False,
                  'modeBarButtonsToRemove': ['toImage','resetScale2d','fullscreen']
              })


def render_tab_scatter(data: pd.DataFrame, aggs: dict):
    """Tab 2: Scatter Plots."""
    st.html("""
    
    <h2> Exploración de la Rentabilidad vs Volúmen de Ventas por Categoría y Subcategoría </h2>
    
    <small><i>Este dashboard permite explorar la relación entre la rentabilidad y el volumen de ventas de cada categoría y subcategoría 
    del minorista Superstore Giant. Se pueden aplicar filtros para analizar diferentes segmentos del negocio, mecanismos de envío, y datos geográficos
    de los clientes.</i></small>
    """)

    # --- Top KPI Cards: Profit Margin Analysis (3-Column) ---

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
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÁXIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{best_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{best_margin['Sub_Category']}</b> ({best_margin['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Ventas Totales de ${best_margin['Total_Sales']:,.2f}.</p>
            </div>
            """)

        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, #c0c0c0); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD PROMEDIO GLOBAL</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{avg_margin:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Calculado sobre {len(subcat_data)} subcategorías analizadas.</p>
            </div>
            """)

        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED}, #ff4d4d); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÍNIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{worst_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{worst_margin['Sub_Category']}</b> ({worst_margin['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Requiere revisión estratégica de márgenes.</p>
            </div>
            """)

    # --- Second Row KPI Cards: Sales Volume Analysis (3-Column) ---


    if not aggs.get("by_subcategory", pd.DataFrame()).empty:
        # Highest sales
        highest_sales = subcat_data.loc[subcat_data["Total_Sales"].idxmax()]
        # Average sales
        avg_sales = subcat_data["Total_Sales"].mean()
        # Lowest sales
        lowest_sales = subcat_data.loc[subcat_data["Total_Sales"].idxmin()]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÁXIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${highest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{highest_sales['Sub_Category']}</b> ({highest_sales['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Margen Operativo: {highest_sales['Profit_Margin_Pct']:.2f}%.</p>
            </div>
            """)

        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, #c0c0c0); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VENTAS PROMEDIO POR UNIDAD</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_sales:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media Transaccional del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Ingresos Totales Brutos: ${subcat_data['Total_Sales'].sum():,.2f}.</p>
            </div>
            """)

        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED}, #ff4d4d); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÍNIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${lowest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{lowest_sales['Sub_Category']}</b> ({lowest_sales['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Segmento con baja tracción comercial.</p>
            </div>
            """)

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
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>{best_cat['Category']}</b> con un margen de <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b> es la categoría líder en rentabilidad dentro del dataset.
                </p>
            </div>
            """)

    with cc2:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            cat_data = aggs["by_category"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            average_margin_by_cat = cat_data["Profit_Margin_Pct"].mean()
            average_sales_by_cat = cat_data["Total_Sales"].mean()
            st.html(f"""
            <div style="background: linear-gradient(135deg, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'],worst_cat['Total_Sales'],average_margin_by_cat, average_sales_by_cat)},{_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'],worst_cat['Total_Sales'],average_margin_by_cat, average_sales_by_cat)}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>{worst_cat['Category']}</b> con un margen de <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f}</b>es la categoría sugerida para su revisión y seguimiento.
                </p>
            </div>
            """)

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
            st.plotly_chart(fig, use_container_width=True, config={
                      "displaylogo": False,
                      "displayModeBar": False})

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
            st.html(f"""
                <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>{best_cat['Sub_Category']}</b> con margen del <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b> hija de la categoría <b>{best_cat['Category']}</b> es la subcategoría más rentable dentro del periodo de análisis.
                    </p>
                </div>
                """)

    with csub2:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            cat_data = aggs["by_subcategory"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.html(f"""
                <div style="background: linear-gradient(135deg, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'],worst_cat['Total_Sales'],average_margin_by_cat, average_sales_by_cat)}, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'],worst_cat['Total_Sales'],average_margin_by_cat, average_sales_by_cat)}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>{worst_cat['Sub_Category']}</b> con margen del <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f} </b> dentro de la categoría <b>{worst_cat['Category']} </b>corresponde a la categoría con el más bajo rendimiento en el periodo de análisis.
                    </p>
                </div>
                """)

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
            st.plotly_chart(fig, use_container_width=True, config={
                      "displaylogo": False,
                      "displayModeBar": False})


def render_tab_pricing(data: pd.DataFrame, aggs: dict):
    """Tab 3: Pricing Strategy Analysis."""
    st.html("""
    <h2 style='text-align: left; '> Estrategias de Precios y Optimización del Valor de Vida del Cliente </h2>
    <small><i>
        Este panel ofrece una visión analítica profunda sobre la arquitectura de precios y su impacto directo en la rentabilidad 
        y la lealtad de gasto del cliente. A través de la correlación de volúmenes, márgenes y comportamientos transaccionales, 
        se identifican los pilares estratégicos que sustentan el crecimiento del minorista Superstore Giant.
    </i></small>
    """)


    c1, c2, c3 = st.columns(3)
    with c1:
        # Highest average price category (from price_volume_by_category)
        agg_pv = aggs.get("price_volume_by_category", pd.DataFrame())
        if not agg_pv.empty:
            highest_price_cat = agg_pv.loc[agg_pv["Avg_Price"].idxmax(), "Category"]
            highest_price = agg_pv["Avg_Price"].max()
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_ATTENTION}, #1a8ab8); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LIDERAZGO EN PRECIOS PREMIUM</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">{highest_price_cat}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ticket Promedio: <b>${highest_price:,.2f}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    La categoría <b>{highest_price_cat}</b> se posiciona como el segmento de mayor valor transaccional, liderando la captura de ingresos por unidad vendida.
                  </p>
              </div>
            """)
    
    with c2:
        # Average customer purchase value (from customer_value_by_category)
        agg_cv = aggs.get("customer_value_by_category", pd.DataFrame())
        if not agg_cv.empty:
            total_sales = agg_cv["Total_Sales"].sum()
            unique_customers = agg_cv["Unique_Customers"].sum()
            avg_customer_value = total_sales / unique_customers
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREEN}, #89ae04); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">CUSTOMER LIFETIME VALUE</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_customer_value:,.2f}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Base de Clientes: <b>{unique_customers:,}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Este KPI establece el valor promedio generado por cada cliente único, sirviendo como benchmark crítico para estrategias de fidelización.
                  </p>
              </div>
            """)
    
    with c3:
        # Most efficient pricing category (highest margin) - from price_volume_by_category
        if not agg_pv.empty:
            best_margin_cat = agg_pv.loc[agg_pv["Profit_Margin_Pct"].idxmax(), "Category"]
            best_margin = agg_pv["Profit_Margin_Pct"].max()
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_WARNING}, #f4c060); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">EFICIENCIA OPERATIVA DE PRECIOS</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">{best_margin_cat}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Operativo: <b>{best_margin:.2f}%</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    <b>{best_margin_cat}</b> demuestra la estructura de precios más saludable, optimizando el retorno sobre cada dólar invertido por el cliente.
              </p>
          </div>
        """)

    # --- Price-Volume-Profit Bubble Chart ---
    st.markdown("---")
    st.html("<h3>Análisis de la relación entre Precio-Volumen-Márgen por Categoría</h3>")
    
    c1, c2 = st.columns(2)
    with c1:
        # Highest revenue category (from aggs)
        agg_pv = aggs.get("price_volume_by_category", pd.DataFrame())
        if not agg_pv.empty:
            top_revenue_cat = agg_pv.loc[agg_pv["Total_Revenue"].idxmax(), "Category"]
            top_revenue = agg_pv["Total_Revenue"].max()
            st.html(f"""
            <div style="background: linear-gradient(90deg, {COLOR_GREEN_DARK}, {COLOR_GREEN_LIGHT}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                  <h4 style="margin: 0; color: {WHITE}; font-weight: 400; letter-spacing: 0.5px;">GENERACIÓN DE INGRESOS BRUTOS</h4>
                  <h2 style="margin: 10px 0; color: {WHITE}; font-size: 2em;">${top_revenue:,.2f}</h2>
                  <p style="margin: 5px 0; color: {WHITE};">Líder: <b>{top_revenue_cat}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; color: {WHITE}; line-height: 1.4;">Aporta el motor principal de flujo de caja para la operación global.</p>
              </div>
            """)
    
    with c2:
        # Transaction count
        total_transactions = len(data)
        avg_transaction_value = data['Sales'].mean()
        st.html(f"""
        <div style="background: linear-gradient(90deg, {COLOR_LINE}, {COLOR_GREY}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {WHITE}; font-weight: 400; letter-spacing: 0.5px;">VELOCIDAD Y TRACCIÓN COMERCIAL</h4>
              <h2 style="margin: 10px 0; color: {WHITE}; font-size: 2em;">{total_transactions:,} Órdenes</h2>
              <p style="margin: 5px 0; color: {WHITE};">Ticket Medio: <b>${avg_transaction_value:,.2f}</b></p>
              <p style="margin: 10px 0 0 0; font-size: 0.9em; color: {WHITE}; line-height: 1.4;">Refleja la intensidad de la demanda y la recurrencia operativa del portafolio.</p>
          </div>
        """)
    
    # Filters for bubble chart
    f1, f2 = st.columns(2)
    with f1:
        agg_pv = aggs.get("price_volume_by_category", pd.DataFrame())
        max_revenue = int(agg_pv["Total_Revenue"].max()) if not agg_pv.empty else 0
        min_revenue_filter = st.number_input(
            "Umbral de Ingresos Mínimos",
            value=0,
            key="bubble_min_revenue",
            min_value=0,
            max_value=max_revenue,
            step=10000,
            format='%d',
            help="Filtra categorías por volumen de ventas para limpiar el análisis visual."
        )
    with f2:
        margin_range_bubble = st.slider(
            "Rango de Margen de Utilidad Objetivo (%)",
            min_value=-100,
            max_value=100,
            value=(-50, 100),
            key="bubble_margin_range",
            help="Define el espectro de rentabilidad para el análisis comparativo."
        )
    
    # Apply filters and plot
    agg_bubble = aggs.get("price_volume_by_category", pd.DataFrame())
    if not agg_bubble.empty:
        filtered_bubble = agg_bubble[
            (agg_bubble['Total_Revenue'] >= min_revenue_filter) &
            (agg_bubble['Profit_Margin_Pct'] >= margin_range_bubble[0]) &
            (agg_bubble['Profit_Margin_Pct'] <= margin_range_bubble[1])
        ]
        
        if not filtered_bubble.empty:
            st.plotly_chart(
                plot_price_volume_profit_bubble(filtered_bubble),
                use_container_width=True,
                config={"displaylogo": False, "displayModeBar": False}
            )
        else:
            st.warning("No se encontraron categorías que cumplan con los criterios de filtrado seleccionados.")


    st.markdown("---")
    st.html("<h3 >Impacto de Descuentos en la Compra Promedio de un Cliente</h3>")

    c2 = st.columns(1)
    with c2[0]:
        # Category with highest customer value (from aggs)
        agg_cv = aggs.get("customer_value_by_category", pd.DataFrame())
        if not agg_cv.empty:
            best_cat = agg_cv.loc[agg_cv["Avg_Customer_Value"].idxmax(), "Category"]
            best_value = agg_cv["Avg_Customer_Value"].max()
            st.html(f"""
            <div style="background-color: {COLOR_ATTENTION}; padding: 25px; border-radius: 15px; color: white;">
                  <h4 style="margin: 0; font-weight: 300;">LÍDER DE SEGMENTO DE ALTO VALOR</h4>
                  <h2 style="margin: 10px 0; font-size: 2em;">{best_cat}</h2>
                  <p style="margin: 5px 0;">Lealtad de Gasto: <b>${best_value:,.2f}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.85em; opacity: 0.9;">Segmento con mayor disposición de pago y potencial de rentabilidad.</p>
              </div>
            """)
    
    # Filters for line chart
    f1, f2 = st.columns(2)
    with f1:
        all_cats = sorted(data["Category"].dropna().unique().tolist())
        sel_cats_line = st.multiselect(
            "Selección de Categorías Estratégicas",
            options=all_cats,
            default=all_cats,
            key="line_chart_cats",
            help="Seleccione las categorías a incluir en el análisis de dinámica de valor."
        )
    with f2:
        agg_cv = aggs.get("customer_value_by_category", pd.DataFrame())
        max_avg = int(agg_cv["Avg_Customer_Value"].max()) if not agg_cv.empty else 1000
        customer_value_range = st.slider(
            "Umbral de Valor Promedio por Cliente ($)",
            min_value=0,
            max_value=max_avg + 100,
            value=(0, max_avg + 100),
            key="line_customer_value_range",
            help="Filtrar categorías según su valor promedio por cliente."
        )
    
    # Apply filters and plot
    agg_line = aggs.get("customer_purchase_by_discount", pd.DataFrame())
    if not agg_line.empty:
        filtered_line = agg_line.copy()
        if sel_cats_line:
            filtered_line = filtered_line[filtered_line['Category'].isin(sel_cats_line)]
        
        # Filter by customer value range using per-category averages
        cat_avg = filtered_line.groupby('Category')['Avg_Purchase_Per_Customer'].mean()
        valid_cats = [cat for cat in cat_avg.index
                      if customer_value_range[0] <= cat_avg[cat] <= customer_value_range[1]]
        filtered_line = filtered_line[filtered_line['Category'].isin(valid_cats)]
        
        if not filtered_line.empty:
            st.plotly_chart(
                plot_avg_customer_purchase_by_discount(filtered_line),
                use_container_width=True,
                config={"displaylogo": False, "displayModeBar": False}
            )
        else:
            st.info("Ajuste los filtros para visualizar la dinámica de valor.")



def render_tab_data(data: pd.DataFrame, aggs: dict):
    """Tab 3: Data View."""
    st.html("""
    <h2>Registro de las tablas de datos generales usadas para el análisis de datos</h2>
    """)

    # Top: Full dataset
    st.subheader("Dataset Completo")
    st.dataframe(data, use_container_width=True)

    st.subheader("Tablas de Datos Agregadas y Modificadas")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Descuento vs Ganancia ")
        st.dataframe(aggs.get("profit_by_discount", pd.DataFrame()))
    with c2:
        st.markdown("#### Margen de Ganancia por Categoria y Rangos de Descuentos")
        st.dataframe(aggs.get("margin_by_category", pd.DataFrame()))
    with c3:
        st.markdown("#### Margen de Ganancia por Categoría ")
        st.dataframe(aggs.get("by_category", pd.DataFrame()))

    st.markdown("#### Margen de Ganancia por Subcategoría")
    st.dataframe(aggs.get("by_subcategory", pd.DataFrame()))

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("#### Precio-Volumen-Utilidad por Categoría")
        st.dataframe(aggs.get("price_volume_by_category", pd.DataFrame()))
    with c5:
        st.markdown("#### Compra Promedio por Cliente según Descuento y Categoría")
        st.dataframe(aggs.get("customer_purchase_by_discount", pd.DataFrame()))
    with c6:
        st.markdown("#### Valor Promedio por Cliente por Categoría")
        st.dataframe(aggs.get("customer_value_by_category", pd.DataFrame()))


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
    tabs = st.tabs([
        "Visión Rentabilidad por Descuentos", 
        "Visión Rentabilidad por Categoría y Subcategoría", 
        "Visión Estrategia de Precios",
        "Visión Datos Usados"
    ])

    with tabs[0]:
        render_tab_overview(filtered, aggregations)
    with tabs[1]:
        render_tab_scatter(filtered, aggregations)
    with tabs[2]:
        render_tab_pricing(filtered, aggregations)
    with tabs[3]:
        render_tab_data(filtered, aggregations)


if __name__ == "__main__":
    main()