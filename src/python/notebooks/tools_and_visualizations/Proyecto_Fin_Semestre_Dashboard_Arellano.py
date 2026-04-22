import datetime
from datetime import timedelta
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
            "y": 0.9,
            "font": {"size": 16, "family": "Arial"},
        },
        autosize=False,
        width=1200,
        height=700,
        margin=dict(
            l=150,
            r=150,
            b=150,
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


    _apply_standard_theme(fig, "Comparativa de Ganancia y Margen de Ganancia Promedio por cada Intervalo de Descuentos Aplicado",
                          "Durante el periodo 2014-2017, las ventas generan ganancias y rentabilidad para la empresa mientras el descuento<br>se mantenga debajo del 30%")

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

def plot_price_volume_profit_bubble(data):
    """Bubble chart: Price-Volume-Profit by Category."""
    agg_price_volume = data.groupby('Category').agg({
        'Sales': ['sum', 'mean'],
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    agg_price_volume.columns = ['Category', 'Total_Revenue', 'Avg_Price', 'Total_Profit', 'Transaction_Count']
    agg_price_volume['Profit_Margin_Pct'] = (agg_price_volume['Total_Profit'] / agg_price_volume['Total_Revenue']) * 100
    
    fig = go.Figure()
    
    for idx, row in agg_price_volume.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Avg_Price']],
            y=[row['Profit_Margin_Pct']],
            mode='markers+text',
            marker=dict(
                size=row['Total_Revenue'] / 5000,
                color=CATEGORY_COLORS.get(row['Category'], COLOR_GREY),
                line=dict(color=COLOR_LINE, width=1),
                opacity=0.7
            ),
            text=row['Category'],
            textposition='top center',
            textfont=dict(size=12, color=COLOR_LINE, family="Arial Black"),
            name=row['Category'],
            showlegend=True,
            hovertemplate=(
                f"<b>{row['Category']}</b><br>"
                f"Precio Promedio: ${row['Avg_Price']:,.2f}<br>"
                f"Margen de Utilidad: {row['Profit_Margin_Pct']:.2f}%<br>"
                f"Ingresos Totales: ${row['Total_Revenue']:,.2f}<br>"
                f"Transacciones: {row['Transaction_Count']}<extra></extra>"
            )
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_LINE, line_width=1, opacity=0.5)
    
    _apply_standard_theme(
        fig,
        "Análisis Precio-Volumen-Utilidad por Categoría",
        "El tamaño de las burbujas representa los ingresos totales. Las categorías con mayor precio promedio<br>y margen positivo indican estrategias de precios efectivas."
    )
    
    _apply_x_axis_customization(fig, x_label="Precio Promedio ($)", x_axis_style={
        'showgrid': True,
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
    }, x_axis_label_distance=-0.15)
    
    _apply_y_axis_customization(fig, y_label="Margen de Utilidad (%)", y_axis_style={
        'showgrid': True,
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
        legend=dict(
            title="Categoría",
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            itemdoubleclick=False,
            itemclick=False
        )
    )
    
    return fig


def plot_pricing_efficiency_quadrant(data):
    """Quadrant analysis: Pricing Strategy Efficiency by Category."""
    agg_pricing_efficiency = data.groupby('Category').agg({
        'Sales': ['sum', 'mean'],
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    agg_pricing_efficiency.columns = ['Category', 'Total_Revenue', 'Avg_Price', 'Total_Profit', 'Transaction_Volume']
    agg_pricing_efficiency['Profit_Margin_Pct'] = (agg_pricing_efficiency['Total_Profit'] / agg_pricing_efficiency['Total_Revenue']) * 100
    
    avg_price = agg_pricing_efficiency['Avg_Price'].mean()
    avg_margin = agg_pricing_efficiency['Profit_Margin_Pct'].mean()
    
    fig = go.Figure()
    
    fig.add_hline(
        y=avg_margin,
        line_dash='dash',
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Margen Promedio: {avg_margin:.1f}%",
        annotation_position="left",
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial", style='italic'),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.1
        )
    )
    
    fig.add_vline(
        x=avg_price,
        line_dash="dash",
        line_color=COLOR_LINE,
        line_width=1,
        opacity=0.5,
        annotation_text=f"Precio Promedio: ${avg_price:,.0f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=10, color=COLOR_LINE, family="Arial", style='italic'),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )
    
    for idx, row in agg_pricing_efficiency.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Avg_Price']],
            y=[row['Profit_Margin_Pct']],
            mode='markers+text',
            marker=dict(
                size=row['Transaction_Volume'] / 50,
                color=CATEGORY_COLORS.get(row['Category'], COLOR_GREY),
                line=dict(color=COLOR_LINE, width=1),
                opacity=0.7
            ),
            text=row['Category'],
            textposition='top center',
            textfont=dict(size=12, color=COLOR_LINE, family="Arial Black"),
            name=row['Category'],
            showlegend=True,
            hovertemplate=(
                f"<b>{row['Category']}</b><br>"
                f"Precio Promedio: ${row['Avg_Price']:,.2f}<br>"
                f"Margen de Utilidad: {row['Profit_Margin_Pct']:.2f}%<br>"
                f"Volumen de Transacciones: {row['Transaction_Volume']}<br>"
                f"Ingresos Totales: ${row['Total_Revenue']:,.2f}<extra></extra>"
            )
        ))
    
    _apply_standard_theme(
        fig,
        "Análisis de Cuadrantes: Eficiencia de Estrategia de Precios",
        "El tamaño de las burbujas representa el volumen de transacciones. Los cuadrantes dividen las categorías<br>según precio y margen promedio para identificar oportunidades de optimización."
    )
    
    _apply_x_axis_customization(fig, x_label="Precio Promedio ($)", x_axis_style={
        'showgrid': True,
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
    }, x_axis_label_distance=-0.15)
    
    _apply_y_axis_customization(fig, y_label="Margen de Utilidad (%)", y_axis_style={
        'showgrid': True,
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
        legend=dict(
            title="Categoría",
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            itemdoubleclick=False,
            itemclick=False
        )
    )
    
    return fig


def plot_avg_customer_purchase_by_discount(data):
    """Line chart: Average Customer Purchase Value by Category across Discount Bins."""
    agg_customer_purchase = data.groupby(['Discount_Bin', 'Category']).agg({
        'Sales': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    
    agg_customer_purchase['Avg_Purchase_Per_Customer'] = (
        agg_customer_purchase['Sales'] / agg_customer_purchase['Customer ID']
    )
    
    fig = go.Figure()
    
    for category in sorted(data['Category'].unique()):
        cat_data = agg_customer_purchase[agg_customer_purchase['Category'] == category]
        
        fig.add_trace(go.Scatter(
            x=cat_data['Discount_Bin'].astype(str),
            y=cat_data['Avg_Purchase_Per_Customer'],
            mode='lines+markers',
            name=category,
            line=dict(
                color=CATEGORY_COLORS.get(category, COLOR_GREY),
                width=3
            ),
            marker=dict(
                size=10,
                color=CATEGORY_COLORS.get(category, COLOR_GREY),
                line=dict(color='white', width=2)
            ),
            hovertemplate=(
                f"<b>{category}</b><br>"
                "Rango de Descuento: %{x}<br>"
                "Compra Promedio: $%{y:,.2f}<br>"
                "<extra></extra>"
            )
        ))
    
    _apply_standard_theme(
        fig,
        "Valor Promedio de Compra por Cliente según Categoría y Rango de Descuento",
        "Análisis del comportamiento de compra promedio por cliente en cada categoría a medida que aumenta<br>el descuento aplicado. Identifica el punto óptimo de descuento para maximizar el valor por cliente."
    )
    
    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", tick_angle=30, x_axis_label_distance=-0.15)
    _apply_y_axis_customization(fig, y_label="Compra Promedio por Cliente ($)", y_axis_style={
        'showgrid': True,
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
            font=dict(size=10, color=COLOR_LINE, family="Arial", style='italic'),
            bgcolor="white",
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            y=0.95
        )
    )

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
        "Durante el periodo 2014-2017, <b>Technology</b> fue la categoría con las mayores ventas de $690K y el mejor margen ganancia promedio de 18.62%"
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
    }, x_axis_label_distance=-0.1)
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
    legend_x_start = 0.1
    legend_y_start = 1.02
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
        "Comparativo de Rentabilidad en base al Porcentaje de Ganancia vs Ventas Totales por Subcategoría",
        "Durante el periodo 2014-2017, la subcategoría <b>Paper</b> dentro de la categoría Office Supplies muestra un desempeño óptimo con ventas totales de $65K y un margen de ganancia promedio del 43.58%."
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
    }, x_axis_label_distance=-0.1)
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
      st.html("""
          <div style='display:flex; align-items:center; gap:10px; margin:5px; justify-content:center'>
              <h3>Resultados Generales del Análisis de Márgenes y Porcentajes de Ganancia por Porcentaje de Descuento</h3>
          </div>
          """)
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
          avg_margin = aggs.get("margin_by_discount", pd.DataFrame())
          if not avg_margin.empty:
              overall_margin = avg_margin["Avg_Profit_Margin_Pct"].mean()
              st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_WARNING}, #f4c060); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">MARGEN DE GANANCIA PROMEDIO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{overall_margin:.2f}%</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Benchmark de Rendimiento Global</p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        El rendimiento porcentual promedio se mantiene saludable bajo una política de descuentos controlada y segmentada.
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
          if not data.empty:
              top_cat = data.groupby("Category")["Sales"].sum().idxmax()
              top_sales = data.groupby("Category")["Sales"].sum().max()
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
          margin_by_subcat = aggs.get("margin_by_subcategory", pd.DataFrame())
          if not margin_by_subcat.empty:
              best_subcat = margin_by_subcat.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().idxmax()
              best_subcat_margin = margin_by_subcat.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().max()
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
              worst_subcat = margin_by_subcat.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().idxmin()
              worst_subcat_margin = margin_by_subcat.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().min()
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
                    Líder de Rentabilidad: <b>{best_cat['Category']}</b> con un margen de <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b>.
                </p>
            </div>
            """)

    with cc2:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            cat_data = aggs["by_category"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_WARNING}, #f4c060); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    Punto de Revisión: <b>{worst_cat['Category']}</b> con un margen de <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f}</b>.
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
                        Máximo Rendimiento: <b>{best_cat['Sub_Category']}</b> con margen del <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b>.
                    </p>
                </div>
                """)

    with csub2:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            cat_data = aggs["by_subcategory"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.html(f"""
                <div style="background: linear-gradient(135deg, {COLOR_WARNING}, #f4c060); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        Mínimo Rendimiento: <b>{worst_cat['Sub_Category']}</b> con margen del <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f}</b>.
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
    <h2 style='text-align: left; color: #13678A;'> Estrategias de Precios y Optimización del Valor de Vida del Cliente </h2>
    <p style='font-size: 1.1em; color: #5A5A5C;'>
        Este panel ofrece una visión analítica profunda sobre la arquitectura de precios y su impacto directo en la rentabilidad 
        y la lealtad de gasto del cliente. A través de la correlación de volúmenes, márgenes y comportamientos transaccionales, 
        se identifican los pilares estratégicos que sustentan el crecimiento del minorista Superstore Giant.
    </p>
    """)

    # --- KPI Cards (3 columns) ---
    st.html("""
        <div style='display:flex; align-items:center; gap:10px; margin:20px 0; justify-content:flex-start'>
            <h3 style='color: #13678A; border-left: 5px solid #13678A; padding-left: 15px;'>Resultados Ejecutivos: Optimización de Precios</h3>
        </div>
        """)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        # Highest average price category
        cat_avg_price = data.groupby('Category')['Sales'].mean()
        highest_price_cat = cat_avg_price.idxmax()
        highest_price = cat_avg_price.max()
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
        # Average customer purchase value
        total_sales = data['Sales'].sum()
        unique_customers = data['Customer ID'].nunique()
        avg_customer_value = total_sales / unique_customers
        st.html(f"""
        <div style="background: linear-gradient(135deg, {COLOR_GREEN}, #89ae04); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
              <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VALOR DE VIDA DEL CLIENTE (ALV)</h4>
              <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_customer_value:,.2f}</h2>
              <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Base de Clientes: <b>{unique_customers:,}</b></p>
              <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                Este KPI establece el valor promedio generado por cada cliente único, sirviendo como benchmark crítico para estrategias de fidelización.
              </p>
          </div>
        """)
    
    with c3:
        # Most efficient pricing category (highest margin)
        cat_margin = data.groupby('Category').apply(
            lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100)
        )
        best_margin_cat = cat_margin.idxmax()
        best_margin = cat_margin.max()
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
    st.html("<h3 style='color: #13678A;'>Análisis Avanzado de Rentabilidad: Sinergia Precio-Volumen-Margen</h3>")
    st.markdown("""
    *Visualización tridimensional que correlaciona la inversión del cliente con el retorno operativo. 
    Las esferas superiores revelan productos con alta elasticidad y márgenes optimizados.*
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        # Highest revenue category
        cat_revenue = data.groupby('Category')['Sales'].sum()
        top_revenue_cat = cat_revenue.idxmax()
        top_revenue = cat_revenue.max()
        st.html(f"""
        <div style="background: linear-gradient(135deg, white, #f9f9f9); border-left: 8px solid {COLOR_GREEN}; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_LINE}; font-weight: 400; letter-spacing: 0.5px;">GENERACIÓN DE INGRESOS BRUTOS</h4>
              <h2 style="margin: 10px 0; color: {COLOR_GREEN}; font-size: 2em;">${top_revenue:,.2f}</h2>
              <p style="margin: 5px 0; color: {COLOR_LINE};">Líder: <b>{top_revenue_cat}</b></p>
              <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; line-height: 1.4;">Aporta el motor principal de flujo de caja para la operación global.</p>
          </div>
        """)
    
    with c2:
        # Transaction count
        total_transactions = len(data)
        avg_transaction_value = data['Sales'].mean()
        st.html(f"""
        <div style="background: linear-gradient(135deg, white, #f9f9f9); border-left: 8px solid {COLOR_GREY}; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_LINE}; font-weight: 400; letter-spacing: 0.5px;">VELOCIDAD Y TRACCIÓN COMERCIAL</h4>
              <h2 style="margin: 10px 0; color: {COLOR_GREY}; font-size: 2em;">{total_transactions:,} Órdenes</h2>
              <p style="margin: 5px 0; color: {COLOR_LINE};">Ticket Medio: <b>${avg_transaction_value:,.2f}</b></p>
              <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; line-height: 1.4;">Refleja la intensidad de la demanda y la recurrencia operativa del portafolio.</p>
          </div>
        """)
    
    # Filters for bubble chart
    f1, f2 = st.columns(2)
    with f1:
        cat_agg = data.groupby('Category')['Sales'].sum()
        max_revenue = int(cat_agg.max()) if not cat_agg.empty else 0
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
    if not data.empty:
        filtered_bubble = data.copy()
        cat_revenue_check = filtered_bubble.groupby('Category')['Sales'].sum()
        cat_margin_check = filtered_bubble.groupby('Category').apply(
            lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100)
        )
        
        valid_cats = []
        for cat in filtered_bubble['Category'].unique():
            if (cat_revenue_check[cat] >= min_revenue_filter and 
                margin_range_bubble[0] <= cat_margin_check[cat] <= margin_range_bubble[1]):
                valid_cats.append(cat)
        
        filtered_bubble = filtered_bubble[filtered_bubble['Category'].isin(valid_cats)]
        
        if not filtered_bubble.empty:
            st.plotly_chart(
                plot_price_volume_profit_bubble(filtered_bubble),
                use_container_width=True,
                config={"displaylogo": False, "displayModeBar": False}
            )
        else:
            st.warning("No se encontraron categorías que cumplan con los criterios de filtrado seleccionados.")

    # --- Pricing Efficiency Quadrant ---
    st.markdown("---")
    st.html("<h3 style='color: #13678A;'>Matriz de Posicionamiento Estratégico y Eficiencia de Portafolio</h3>")
    st.markdown("""
    *Segmentación de categorías basada en desviaciones de precio y margen. El cuadrante **Superior Derecho** identifica activos estrella, 
    mientras el **Inferior Izquierdo** señala áreas críticas de reestructuración táctica.*
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.html(f"""
        <div style="background: linear-gradient(135deg, white, #f9f9f9); border-top: 6px solid {COLOR_ATTENTION}; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_ATTENTION}; font-weight: 400; letter-spacing: 0.5px;">DIAGNÓSTICO DE CUADRANTES</h4>
              <p style="margin: 15px 0 5px 0; font-size: 0.95em;"><b>Superior Derecho:</b> Alto Valor, Alto Retorno (Zona Óptima)</p>
              <p style="margin: 5px 0; font-size: 0.95em;"><b>Inferior Izquierdo:</b> Commodities / Riesgo de Margen (Zona de Revisión)</p>
          </div>
        """)
    
    with c2:
        # Calculate quadrant distribution
        cat_data = data.groupby('Category').agg({'Sales': 'mean'})
        avg_price_overall = cat_data['Sales'].mean() if not cat_data.empty else 0
        st.html(f"""
        <div style="background: linear-gradient(135deg, white, #f9f9f9); border-top: 6px solid {COLOR_WARNING}; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_WARNING}; font-weight: 400; letter-spacing: 0.5px;">PRECIO DE REFERENCIA (BENCHMARK)</h4>
              <h2 style="margin: 10px 0; color: {COLOR_LINE}; font-size: 2em;">${avg_price_overall:,.2f}</h2>
              <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #666; line-height: 1.4;">Punto de equilibrio del sistema para evaluación de competitividad.</p>
          </div>
        """)
    
    # Filters for quadrant chart
    f1, f2 = st.columns(2)
    with f1:
        cat_price = data.groupby('Category')['Sales'].mean()
        max_price = int(cat_price.max()) if not cat_price.empty else 1000
        price_range_quadrant = st.slider(
            "Filtro de Espectro de Precios ($)",
            min_value=0,
            max_value=max_price + 100,
            value=(0, max_price + 100),
            key="quadrant_price_range"
        )
    with f2:
        cat_transactions = data.groupby('Category')['Order ID'].count()
        max_trans = int(cat_transactions.max()) if not cat_transactions.empty else 1000
        min_transactions = st.number_input(
            "Mínimo de Transacciones para Análisis",
            value=0,
            key="quadrant_min_trans",
            min_value=0,
            max_value=max_trans,
            step=50,
            format='%d'
        )
    
    # Apply filters and plot
    if not data.empty:
        filtered_quadrant = data.copy()
        cat_price_check = filtered_quadrant.groupby('Category')['Sales'].mean()
        cat_trans_check = filtered_quadrant.groupby('Category')['Order ID'].count()
        
        valid_cats = []
        for cat in filtered_quadrant['Category'].unique():
            if (price_range_quadrant[0] <= cat_price_check[cat] <= price_range_quadrant[1] and
                cat_trans_check[cat] >= min_transactions):
                valid_cats.append(cat)
        
        filtered_quadrant = filtered_quadrant[filtered_quadrant['Category'].isin(valid_cats)]
        
        if not filtered_quadrant.empty:
            st.plotly_chart(
                plot_pricing_efficiency_quadrant(filtered_quadrant),
                use_container_width=True,
                config={"displaylogo": False, "displayModeBar": False}
            )
        else:
            st.warning("Ajuste los filtros para visualizar la distribución de cuadrantes.")

    # --- Average Customer Purchase by Discount ---
    st.markdown("---")
    st.html("<h3 style='color: #13678A;'>Dinámica de Valor al Cliente: Impacto de Descuentos en la Lealtad de Gasto</h3>")
    st.markdown("""
    *Análisis longitudinal del ticket promedio frente a la presión promocional. Identifica el **'Sweet Spot'** 
    donde el incentivo maximiza la captura de valor sin erosionar la percepción de marca.*
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        # Optimal discount range
        discount_analysis = data.groupby('Discount_Bin').agg({'Sales': 'sum', 'Customer ID': 'nunique'})
        discount_analysis['Avg_Per_Customer'] = discount_analysis['Sales'] / discount_analysis['Customer ID']
        if not discount_analysis.empty:
            optimal_discount = discount_analysis['Avg_Per_Customer'].idxmax()
            optimal_value = discount_analysis['Avg_Per_Customer'].max()
            st.html(f"""
            <div style="background-color: {COLOR_GREEN}; padding: 25px; border-radius: 15px; color: white;">
                  <h4 style="margin: 0; font-weight: 300;">PUNTO DE EQUILIBRIO PROMOCIONAL</h4>
                  <h2 style="margin: 10px 0; font-size: 2em;">{optimal_discount}</h2>
                  <p style="margin: 5px 0;">Captura Máxima: <b>${optimal_value:,.2f}</b> por cliente</p>
                  <p style="margin: 10px 0 0 0; font-size: 0.85em; opacity: 0.9;">Incentivo óptimo para maximizar el ticket sin sacrificar volumen.</p>
              </div>
            """)
    
    with c2:
        # Category with highest customer value
        cat_customer_value = data.groupby('Category').agg({'Sales': 'sum', 'Customer ID': 'nunique'})
        cat_customer_value['Avg_Per_Customer'] = cat_customer_value['Sales'] / cat_customer_value['Customer ID']
        if not cat_customer_value.empty:
            best_cat = cat_customer_value['Avg_Per_Customer'].idxmax()
            best_value = cat_customer_value['Avg_Per_Customer'].max()
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
            key="line_chart_cats"
        )
    with f2:
        cat_customer_avg = data.groupby('Category').agg({'Sales': 'sum', 'Customer ID': 'nunique'})
        cat_customer_avg['Avg_Per_Customer'] = cat_customer_avg['Sales'] / cat_customer_avg['Customer ID']
        max_avg = int(cat_customer_avg['Avg_Per_Customer'].max()) if not cat_customer_avg.empty else 1000
        customer_value_range = st.slider(
            "Umbral de Valor Promedio por Cliente ($)",
            min_value=0,
            max_value=max_avg + 100,
            value=(0, max_avg + 100),
            key="line_customer_value_range"
        )
    
    # Apply filters and plot
    if not data.empty:
        filtered_line = data.copy()
        if sel_cats_line:
            filtered_line = filtered_line[filtered_line['Category'].isin(sel_cats_line)]
        
        cat_value_check = filtered_line.groupby('Category').agg({'Sales': 'sum', 'Customer ID': 'nunique'})
        cat_value_check['Avg_Per_Customer'] = cat_value_check['Sales'] / cat_value_check['Customer ID']
        
        valid_cats = []
        for cat in filtered_line['Category'].unique():
            if cat in cat_value_check.index:
                if customer_value_range[0] <= cat_value_check.loc[cat, 'Avg_Per_Customer'] <= customer_value_range[1]:
                    valid_cats.append(cat)
        
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