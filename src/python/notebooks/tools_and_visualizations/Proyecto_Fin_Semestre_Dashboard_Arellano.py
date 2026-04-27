#? ---------------------------------------------------------------------------------------------------------------------
#? Proyecto de Fin de Semestre ADM 3083 Herramientas y Visualizacion
#? Dashboard Analitica y de Resultados para Superstore Giant
#? Santiago Arellano 00328370
#? Lunes, 20 de abril de 2026
#?
#? ---------------------------------------------------------------------------------------------------------------------



#? Importe de librerias locales necesarias para el trabajo
import datetime
import math
from datetime import timedelta
from typing import Iterable, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import os

#? Definicion de colores para la visualizacion, esta seccion define todos los tonos usados para las
#? visualizaciones, y los KPIs
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
COLOR_WHITE = "#FFFFFF"
COLOR_POS_DARK = "#516D73"
COLOR_POS_LIGHT = '#C1D4D9'
COLOR_BROWN_DARK = "#BA9B65"
COLOR_BROWN_LIGHT = "#F4EFDC"
COLOR_PURPLE_DARK = "#6F7DA6"
COLOR_PURPLE_LIGHT = "#C1B3F2"
COLOR_RED_DARK = "#D9043D"
COLOR_RED_LIGHT = "#F291A3"

# ? Recently Modified -> https://color.adobe.com/search?q=business
CATEGORY_COLORS = {
    "Furniture": COLOR_BROWN_DARK,  # Blue
    "Office Supplies": "#025940",  # Green!
    "Technology": "#3A5898",  # Green
}

DISCOUNT_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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


#? 1. Definimos la primera funcion base de la aplicacion, que prepara os datos al realizar calculos necesarios base como
#? el margen de ganancia y su contraparte porcentual para cada fila.
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset base realizando calculos y asignaciones necesarias para la visualizacion.

    Operaciones realizadas:
    - Calcula Profit_Margin (Profit / Sales) asignando 0 cuando Sales es menor o igual a 0 para evitar divisiones por cero.
    - Calcula Profit_Margin_Pct como porcentaje del margen.
    - Asigna una etiqueta de rango de descuento (Discount_Bin) usando DISCOUNT_BINS y DISCOUNT_LABELS con pd.cut.

    Args:
        data_df: DataFrame original cargado desde el CSV.

    Returns:
        DataFrame con las columnas nuevas Profit_Margin, Profit_Margin_Pct y Discount_Bin.
    """
    data_df = df.copy()
    #? Margen de Ganancia
    data_df["Profit_Margin"] = np.where(data_df["Sales"] > 0, data_df["Profit"] / data_df["Sales"], 0)
    data_df["Profit_Margin_Pct"] = data_df["Profit_Margin"] * 100

    #? Discount bin usando pd.cut para asignarle a cada fila un discount bin label en base de donde cae su valor de
    #? descuento
    data_df["Discount_Bin"] = pd.cut(
        data_df["Discount"], bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS, include_lowest=True
    )
    return data_df


#? 2. Funcion auxiliar definida para aplicar un formato estandar de titulo, subtitulo, margen y tamano. Esta funcion
#? fue extraida de la funcion para modificacion del eje x e y para tener el titulo aislado y modificable por su cuenta
def _apply_standard_theme(fig, title, subtitle) -> Figure:
    """
    Aplica el formato estandar de titulo, tamano y margenes a una figura de plotly.

    Args:
        fig: Instancia de plotly Figure a modificar.
        title: Texto principal del titulo (se muestra en negrita).
        subtitle: Texto de subtitulo/insight (se coloca en fuente mas pequena y estilo italic).

    Returns:
        La misma figura modificada con layout actualizado.
    """

    #? Armamos la cadena del titulo con el formato adecuado
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

    #? Retornamos la instancia ingresada
    return fig


def _apply_x_axis_customization(figure, x_label=None, x_ticks=None, x_tick_labels=None, tick_angle=30,
                                x_axis_style=None, x_axis_label_distance: int = -0.12, row_id=None, column_id=None):
    """Apply x-axis customization to figure or subplot.

    Args:
        figure: Figura de plotly a modificar.
        x_label: Texto de la etiqueta del eje X (opcional).
        x_ticks: Valores de tick para el eje X (opcional).
        x_tick_labels: Etiquetas para los ticks (opcional).
        tick_angle: Angulo de rotacion de los ticks.
        x_axis_style: Diccionario con estilo para el eje.
        x_axis_label_distance: Distancia de la etiqueta respecto al eje.
        row_id: Fila de subplot (1-indexed) si aplica.
        column_id: Columna de subplot (1-indexed) si aplica.

    Returns:
        La figura modificada.
    """


    #? Estilo de eje base
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
    style = x_axis_style if x_axis_style else default_style

    #? Anadimos las ticks si las tenemos, y si tenemos tick labels las definimos, sino se derivan de los ticks
    if x_ticks is not None:
        style["tickvals"] = x_ticks
        style["ticktext"] = x_tick_labels if x_tick_labels else [str(t) for t in x_ticks]

    #? Actualizamos el eje x de la figura con el estilo aplicado, si tenemos row y col id a ese subplot sino a toda la figura
    if row_id is not None and column_id is not None:
        figure.update_xaxes(style, row=row_id, col=column_id)
    else:
        figure.update_xaxes(style)

    #? Aplicamos la label del eje x si tenemos un x_label
    if x_label:
        annotation_config = {
            "xref": "paper", "yref": "y domain",
            "x": 0,
            "y": x_axis_label_distance,
            "text": x_label,
            "showarrow": False,
            "xanchor": "left",
            "yanchor": 'bottom',
            "font": dict(size=12, color=COLOR_LINE, family="Arial Black")
        }

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
                                y_axis_lable_distance: float = -0.12,
                                row_id=None,
                                column_id=None):
    """Apply y-axis customization to figure or subplot.

    Args:
        figure: Figura de plotly a modificar.
        y_label: Texto de la etiqueta del eje Y (opcional).
        y_ticks: Valores de tick para el eje Y (opcional).
        y_tick_labels: Etiquetas para los ticks del eje Y (opcional).
        y_axis_style: Diccionario con estilo para el eje Y.
        y_tick_style: Estilo adicional para ticks aplicado al layout.
        y_axis_lable_distance: Distancia de la etiqueta respecto al eje.
        row_id: Fila de subplot (1-indexed) si aplica.
        column_id: Columna de subplot (1-indexed) si aplica.

    Returns:
        La figura modificada.
    """
    #? Definoms un estilo base en donde no tenemos titulo  los dtos del eje se definen normalmente
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
        if row_id is not None and column_id is not None:
            annotation_config["xref"] = f"x{column_id} domain" if column_id > 1 else "x domain"
            annotation_config["yref"] = f"y{row_id} domain" if row_id > 1 else "y domain"

        figure.add_annotation(**annotation_config)

    if y_tick_style:
        figure.update_layout(yaxis=y_tick_style, overwrite=False)

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
) -> pd.DataFrame | None:
    """
    Aplica los filtros seleccionados en la barra lateral sobre el dataframe recibido.

    Soporta filtrado por rango de fechas (usando el campo time_field), region, modo de envio, pais, estado y ciudad.

    Args:
        df: DataFrame original.
        date_range: Tupla (start_date, end_date) o None.
        time_field: Nombre de la columna fecha a usar ('Order Date' o 'Ship Date').
        region_sel, ship_sel, country_sel, state_sel, city_sel: Listas de valores seleccionados para filtrar.

    Returns:
        DataFrame filtrado (misma referencia si no aplica ningun filtro).
    """

    #? Si no tenemos datos retornamos el dataframe directamente
    if df is None or df.empty:
        return df

    #? Aplicamos una mascara inicializada en true para todos los datos, la idea es que las operaciones binarias son
    #? mas eficientes y rapidas que hacer operaciones de filtro en el dataframe directamente
    mask = pd.Series(True, index=df.index)

    #? Si tenemos un date range y si tenemos dos fechas registradas y no son nulas entonces extraemos las dos fechas y
    #? aplicamos el filtro de rango sobre el campo de fecha seleccionado, esto nos permite tener un filtro de tiempo dinamico
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start, end = date_range
        if time_field in df.columns:
            col = df[time_field]
            mask &= (col >= pd.to_datetime(start)) & (col <= pd.to_datetime(end))
    #? Aplicamos los filtros de las columnas
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

    #? Al final retornamos el dataframe aplicada la mascara binaria
    return df.loc[mask]


def _return_color_for_cell_based_on_values(
        profit_margin_to_measure,
        sales_to_measure,
        avg_margin,
        avg_sales
):
    """
    Devuelve un color de celda segun la comparacion entre las metricas actuales y sus promedios.

    Logica:
    - margen y ventas por encima del promedio -> COLOR_ATTENTION
    - margen y ventas por debajo del promedio -> COLOR_RED
    - ventas altas y margen bajo -> COLOR_PURPLE_DARK
    - margen alto y ventas bajas -> COLOR_POS_LIGHT
    """
    if profit_margin_to_measure > avg_margin and sales_to_measure > avg_sales:
        return COLOR_ATTENTION
    elif profit_margin_to_measure < avg_margin and sales_to_measure < avg_sales:
        return COLOR_RED
    elif profit_margin_to_measure < avg_margin and sales_to_measure > avg_sales:
        return COLOR_PURPLE_DARK
    else:
        return COLOR_POS_LIGHT


def _assign_color(row, avg_margin, avg_sales):
    """
    Asigna un color a una fila segun su margen y ventas comparados con los promedios.

    Args:
        row: Fila con las columnas Profit_Margin_Pct y Total_Sales.
        avg_margin: Margen promedio de referencia.
        avg_sales: Ventas promedio de referencia.

    Returns:
        Codigo de color para usar en graficos o tarjetas.
    """
    if row["Profit_Margin_Pct"] > avg_margin and row["Total_Sales"] > avg_sales:
        return COLOR_ATTENTION
    elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] < avg_sales:
        return COLOR_RED
    elif row["Profit_Margin_Pct"] < avg_margin and row["Total_Sales"] > avg_sales:
        return COLOR_PURPLE_DARK
    else:
        return COLOR_POS_LIGHT


def _add_color_labels_for_scatter_segmentations(fig, legend_items=None) -> Figure:
    """
    Funcion que permite anadir la leyenda directamente a la figura debajo del titulo y el insight de cada grafico que
    trabaja con segmentaciones por margen y ventas.
    :param fig: figura de plotly ingresada directamente a la funcion
    :param legend_items: items de la leyenda que queremos mostrar en la grafica si no se registra se asume margen y ventas
    :return:
    """
    if legend_items is None:
        legend_items = [
            (COLOR_ATTENTION, "Margen Alto y Ventas Altas"),
            (COLOR_POS_LIGHT, "Margen Alto y Ventas Bajas"),
            (COLOR_PURPLE_DARK, "Margen Bajo y Ventas Altas"),
            (COLOR_RED, "Margen Bajo y Ventas Bajas")
        ]

    #? Definimos constantes relativas en el grafico, basadas en la formulacion del titulo del grafico
    legend_x_start = 0.05
    legend_y_start = 1.05
    box_height = 0.03
    box_width = 0.04
    spacing = 0.01

    #? Iteramos sobre cada uno de los items de la leyenda y vamos dibujando un rectangulo con su color y un texto al lado
    current_x = legend_x_start
    for idx, (color, label) in enumerate(legend_items):
        #? 1. Anadimos la figura coloreada por cada clase, basado en el color definido en el legend_items
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            #? Definimos que inicie en 0.05 x y que se mueva en y hacia abajo el tamano de la caja, y a la derecha el tamano de la caja
            x0=current_x, y0=legend_y_start - box_height,
            x1=current_x + box_width, y1=legend_y_start,
            fillcolor=color,
            opacity=0.7, #Modificamos la opacidad del color para que cuadre con la opacidad del grafico
            line=dict(color=COLOR_LINE, width=1)
        )

        #? 2. Anadimos una label del tipo de clase que estamos presentando
        fig.add_annotation(
            xref="paper", yref="paper",
            x=current_x + box_width + 0.005,
            y=legend_y_start - box_height / 2,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color=COLOR_LINE, family="Arial"),
            borderwidth=1,
            borderpad=3
        )

        #? Calculamos una aproximacion del espacio del texto y usamos la longitud del texto estimada con un spacing para
        #? avanzar el eje x
        text_width = len(label) * 0.006
        current_x += box_width + text_width + spacing

    return fig


def get_aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Funcion encargada de precalcular todas las agregaciones necesarias para los graficos a presentar. Esta funcion realiza todos los
    group by requeridos y los guarda en un diccionario que puede ser usado dentro de la apliacacion para llamar a las
    agrupaciones por nombres y su subsequente caching.
    :param df:
    :return:
    """

    #? 1. Si no tenemos data entonces retornamos un diccionario vacio
    if df is None or df.empty:
        return {}

    #? 1.1 Calculamos la agregacion del promedio de ganancia por rango de descuento para la visualizacion 1 y
    #? resetamos el indice para tener las bins y el profit como clumnas y no eje.
    agg_profit_by_discount = (
        df.groupby("Discount_Bin", observed=True)["Profit"].mean().reset_index()
    )
    #? 1.1.1 Renombramos las columnas para su facil uso en el analisis
    agg_profit_by_discount.columns = ["Discount_Bin", "Avg_Profit"]

    #? 1.2 Calculamos la agregacion de margen de ganancia porcntual promedio por cada discount bin de la visualizacion 2
    agg_margin_by_discount = (
        df.groupby("Discount_Bin", observed=True)["Profit_Margin_Pct"].mean().reset_index()
    )
    agg_margin_by_discount.columns = ["Discount_Bin", "Avg_Profit_Margin_Pct"]

    #? 1.3 Calculamos el margen de ganancia promedio por categoria para la visualizacion 3 lollipop
    agg_margin_by_category = (
        df.groupby(["Discount_Bin", "Category"], observed=True)["Profit_Margin_Pct"].mean().reset_index()
    )
    agg_margin_by_category.columns = ["Discount_Bin", "Category", "Avg_Profit_Margin_Pct"]

    #? 1.4 Calculamos el margen de ganancia promedio por discount bin y subcategorya para el heatmap
    agg_margin_by_subcategory = (
        df.groupby(["Discount_Bin", "Sub-Category"], observed=True)["Profit_Margin_Pct"]
        .mean()
        .reset_index()
    )
    agg_margin_by_subcategory.columns = ["Discount_Bin", "Sub_Category", "Avg_Profit_Margin_Pct"]

    #? 1.5 Calculamos el total de clientes, el total de ingresos y el total de ganancia por categoria para KPIs
    agg_by_category = (
        df.groupby("Category")
        .agg({"Sales": "sum", "Profit": "sum", "Customer ID": "nunique"})
        .reset_index()
    )
    agg_by_category.columns = ["Category", "Total_Sales", "Total_Profit", "Unique_Customers"]
    #? 1.5.1 Calculmoas el margen de ganancia porcentaual por categoria
    agg_by_category["Profit_Margin_Pct"] = np.where(
        agg_by_category["Total_Sales"] > 0,
        (agg_by_category["Total_Profit"] / agg_by_category["Total_Sales"]) * 100,
        0
    )
    #? 1.5.2 Calculamos el total de venta promedio por cliente
    agg_by_category["Avg_Sale_Per_Customer"] = np.where(
        agg_by_category["Unique_Customers"] > 0,
        agg_by_category["Total_Sales"] / agg_by_category["Unique_Customers"],
        0
    )

    #? 1.6 Calculamos el total de ventas, clientes unicos y ganancia por categoria y subcategoria
    agg_by_subcategory = (
        df.groupby(by=["Category", "Sub-Category"])
        .agg({"Sales": "sum", "Profit": "sum", "Customer ID": "nunique"})
        .reset_index()
    )
    agg_by_subcategory.columns = ["Category", "Sub_Category", "Total_Sales", "Total_Profit", "Unique_Customers"]
    agg_by_subcategory["Profit_Margin_Pct"] = np.where(
        agg_by_subcategory["Total_Sales"] > 0,
        (agg_by_subcategory["Total_Profit"] / agg_by_subcategory["Total_Sales"]) * 100,
        0
    )
    agg_by_subcategory["Avg_Sale_Per_Customer"] = np.where(
        agg_by_subcategory["Unique_Customers"] > 0,
        agg_by_subcategory["Total_Sales"] / agg_by_subcategory["Unique_Customers"],
        0
    )

    #? 1.7 Calculamos el total de ordenes, ventas promedio y ganancia total para cada categoria para los scatter plots
    agg_price_volume = df.groupby('Category').agg({
        'Sales': ['sum', 'mean'],
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    agg_price_volume.columns = ['Category', 'Total_Revenue', 'Avg_Price', 'Total_Profit', 'Transaction_Count']
    agg_price_volume['Profit_Margin_Pct'] = np.where(
        agg_price_volume['Total_Revenue'] > 0,
        (agg_price_volume['Total_Profit'] / agg_price_volume['Total_Revenue']) * 100,
        0
    )

    #? 1.8 Calculamos el promedio de compra por cliente para cada categoria y por cada rango de descuento para el analisis
    #? de compra
    agg_customer_purchase = df.groupby(['Discount_Bin', 'Category'], observed=True).agg({
        'Sales': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    agg_customer_purchase.columns = ['Discount_Bin', 'Category', 'Total_Sales', 'Unique_Customers']
    agg_customer_purchase['Avg_Purchase_Per_Customer'] = np.where(
        agg_customer_purchase['Unique_Customers'] > 0,
        agg_customer_purchase['Total_Sales'] / agg_customer_purchase['Unique_Customers'],
        0
    )

    agg_customer_value = df.groupby('Category').agg({
        'Sales': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    agg_customer_value.columns = ['Category', 'Total_Sales', 'Unique_Customers']
    agg_customer_value['Avg_Customer_Value'] = np.where(
        agg_customer_value['Unique_Customers'] > 0,
        agg_customer_value['Total_Sales'] / agg_customer_value['Unique_Customers'],
        0
    )

    #? 1.8 Retornamos todo en un diccionario
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



def plot_waterfall(agg_profit: pd.DataFrame, agg_margin: pd.DataFrame) -> Figure:
    """
    Grafico de waterfall para la demostracion de contenido del promedio de ganancia y margen porcetual por rango de
    decuentos
    :param agg_profit: Dataframe con los datos agregados de profit
    :param agg_margin: Daatframe con los datos agregados del margen porcentual
    :return:
    """

    #? 1. Creamos dos subplots para cada uno de los elementos del waterfall chart con un titulo especifico
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<i style='font-size:10pt'>Ganancia Promedio por <b>Rango de Descuento</b> Aplicado</i>",
            "<i style='font-size:10pt'>Margen de Ganancia Porcentual Promedio por <b>Rango de Descuento</b> Aplicado</i>"
        ),

    )

    #? 2. Generamos una trace, es decir un grafico, basado en Graphic Objects.Waterfall definimos en x los discount bins
    #? en y el promedio de ganancia real y definimos que colores usar para el grafico y sus componentes
    fig.add_trace(
        go.Waterfall(
            x=agg_profit["Discount_Bin"],
            y=agg_profit["Avg_Profit"],
            text=[f"${val:.2f}" for val in agg_profit["Avg_Profit"]],
            textposition="outside",
            increasing={"marker": {"color": COLOR_ATTENTION}},
            decreasing={"marker": {"color": COLOR_NEGATIVE}},
            connector={"line": {"color": COLOR_LINE}},
            name="Avg Profit"
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
            increasing={"marker": {"color": COLOR_ATTENTION}},
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

    # ? Para definir la label definitiva, y responsive extraemos el ultimo valor positivo antes de caer en la rentabilidad
    # ? y extraemos su idxmax() para sacar la bin a la que pertenece
    positive_profit = agg_profit[agg_profit["Avg_Profit"] > 0]
    if not positive_profit.empty:
        max_discount_label = positive_profit["Discount_Bin"].iloc[-1]
        discount_text = f"mientras el descuento se mantenga por <b>debajo o hasta</b> un descuento en el rango máximo de <b>{float(max_discount_label.split('-')[0]) * 100:.0f} al {float(max_discount_label.split('-')[1]) * 100:.0f}%</b>"
    else:
        discount_text = "ya que <b>todos los rangos de descuento generan pérdidas</b> en el periodo seleccionado"


    #? Aplicamos los estilos del grafico general y los ejes para cada par de figuras
    _apply_standard_theme(
        fig,
        "Comparativa de Ganancia y Margen de Ganancia Promedio por cada Intervalo de Descuentos Aplicado",
        f"Durante el periodo de análisis, las ventas generan ganancias y rentabilidad para la empresa {discount_text}")

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



def plot_lollipop(agg_margin_by_cat: pd.DataFrame, selected_categories: None | Iterable[Any]) -> Figure:
    """
    Grafico de lollipop para mostrar el margen de ganancia promedio por categoria y por rango de descuento
    :param agg_margin_by_cat:
    :param selected_categories:
    :return:
    """

    #? Filtramos los datos dependiendo de si teneomos categorias o no definidas en las categorias seleccionadas
    if selected_categories:
        data = agg_margin_by_cat[agg_margin_by_cat["Category"].isin(selected_categories)]
    else:
        data = agg_margin_by_cat

    fig = go.Figure()
    categories = data["Category"].unique()

    #? Por cada categoria procedemos a dibuar tanto el scatter de un solo eje para el palo del lollipop chart
    for idx, category in enumerate(categories):

        cat_data = data[data["Category"] == category]
        x_positions = []
        discount_bin_labels = []

        #? OJO: calculamos los offsets de donde van a ir los palitos de cada una de las categorias en base a un offset especifico
        #? definido para todas las columnas por categoria en base a el numero de la categoria y todos los steps de discount bin existentes.
        #? Esto porque al tener varias categorias en el mismo grafico y con varios pasos por cada discount bin, los desfasamos
        #? para que se vean bien
        for i, discount_bin in enumerate(cat_data["Discount_Bin"]):
            offset = (idx - 1) * 0.25
            x_positions.append(i + offset)
            discount_bin_labels.append(str(discount_bin))

        #? Dibujamos por cada categoria el palito en la posicion correspondiente definida por el calculo del offset
        #? para cada discount bin
        for i, x in enumerate(x_positions):
            fig.add_trace(
                go.Scatter(
                    x=[x, x], #Esto le indica al programa que la x no cambia de lugar, es una sola posicion
                    y=[0, cat_data["Avg_Profit_Margin_Pct"].iloc[i]], #Esto le indica a plotly que el final del grafico es el punto
                    # encima que representa el valor del margen porcentual, lo que genera una linea recta
                    mode="lines",
                    line=dict(color=COLOR_LINE, width=2),
                    hoverinfo="skip",
                    legendgroup=category,
                    name=category,
                    showlegend=False,
                )
            )

        #? Aqui creamos el scatter de la bolita para el lollipop chart, definido en la misma posicio de x que la posicion calculada
        #? para cada uno de los puntos de x_positions (puntos de cada discount bin) y donde la altura varia con respecto
        #? al valor del porcentaje
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
                textfont=dict(size=10, family="Arial Black"),
                name=category,
                legendgroup=category,
                customdata=discount_bin_labels,
                hovertemplate=f"<b>{category}</b><br>Margin: %{{y:.2f}}%<br>Discount: %{{customdata}}<extra></extra>",
            )
        )

    #? Aplicamos los temas generales del grafico y sus ejes
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

    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", x_tick_labels=DISCOUNT_LABELS,
                                x_ticks=list(range(len(DISCOUNT_LABELS))), tick_angle=0)
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



def plot_heatmap(agg_margin_by_subcat: pd.DataFrame, selected_subcats: Iterable[Any], top_n: int) -> Figure | None:
    """
    Grafico de heatmap para mostrar el margen de ganancia promedio por subcategoria y por rango de descuento
    :param agg_margin_by_subcat:
    :param selected_subcats:
    :param top_n:
    :return:
    """

    #? Realizamos una  copia de los datos originales para la modificacion interna, dado que vamos a modificar la data
    #? dependiendo de los filtros que tengamos. Si tenemos subcategorias seleccioandas entoncs ifltramos los datos para esas
    #?categorias, si tenemos top_n entncones realizamos otro proceso
    data = agg_margin_by_subcat.copy()
    if selected_subcats:
        data = data[data["Sub_Category"].isin(selected_subcats)]
    elif top_n:
        #? Para obtener l top_n hacemos una agrupacion en donde obtenemos el promedio del margen de ganancia porcentual para
        #? cada subcateogria y obtenemos el top_n de ese promedio.
        overall = data.groupby("Sub_Category")["Avg_Profit_Margin_Pct"].mean().reset_index()
        top_subs = overall.nlargest(top_n, "Avg_Profit_Margin_Pct")["Sub_Category"].tolist()
        data = data[data["Sub_Category"].isin(top_subs)]

    if data.empty:
        return None

    #? Dado que tenemos una tabla de datos y no una tabla pivote para hacer el heatmap, transformamos a una tabla pivote
    #? cruzada con las categorias como indice y las columnas como discount bins con el margen de ganancia promedio
    #? como el  dato de cada celda
    pivot_df = data.pivot(index="Sub_Category", columns="Discount_Bin", values="Avg_Profit_Margin_Pct")

    #? Aqui nos aseguramos de que las columnas se encuentren en el orden requerido por el orden natural de los descuentos
    pivot_df = pivot_df.reindex(columns=DISCOUNT_LABELS)


    #? Aqui generaoms las labels de los datos para el grafico, la idea esq que no queremos mostrar NaN en celdas vacias,
    #? por lo que generamos el arreglo de texto, es decr una matrix de dos dimensiones que tiene n filas como subcategorias
    #? y m columna spor discount bins donde el valor se registra si no es NaN y si lo es se esconde.
    text_array = []
    for i, row in enumerate(pivot_df.values):
        text_row = []
        for j, val in enumerate(row):
            if pd.isna(val):
                text_row.append("")  # Empty string for NaN
            else:
                text_row.append(f"{val:.0f}%")
        text_array.append(text_row)

    #? Generamos el heatmap con los datos ya procesados
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values, #Esto define los valores en el dataset
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Margin %"),
        hovertemplate="<b>%{y}</b><br>Discount: %{x}<br>Margin: %{z:.2f}%<extra></extra>",
        xgap=1,
        ygap=1,
        text=text_array, #Aqui reistramos el texto generado
        texttemplate="%{text}",
        showscale=True
    ))

    #? Aplicamos el foramto normal
    _apply_standard_theme(
        fig,
        "Tabulación del Margen de Ganancia por Subcategoría segmentado por Rango de Descuento",
        "Las 17 subcategorías de la compañia muestran una tendencia similar donde un descuento superior al 30% invierte<br> las ganancias porcentuales y resigna a pérdidas para cada venta realizada",
    )

    _apply_x_axis_customization(fig, x_label="Rango de Descuentos", x_tick_labels=DISCOUNT_LABELS,
                                x_ticks=list(range(len(DISCOUNT_LABELS))), tick_angle=0)
    _apply_y_axis_customization(fig, y_label="Sub-Categorías", y_axis_lable_distance=-0.1)
    return fig



def plot_price_volume_profit_bubble(agg_price_volume: pd.DataFrame) -> Figure | None:
    """
    Grafico de burbujas que muestra el analisis precio volumen utilidad por categoria
    :param agg_price_volume:
    :return:
    """

    #? Aqui extraemos las constantes requeridas para los ejes del scatter plot y las lineas de segmentos.
    avg_margin = agg_price_volume['Profit_Margin_Pct'].mean()
    avg_price = agg_price_volume['Avg_Price'].mean()
    agg_price_volume = agg_price_volume.copy()


    #? Aqui aplicamos una categorizacion en base condiciones definidas mediante Numpy, dado que tenemos varias condiciones
    #? usamos Numpy.select para estas multiples condiciones basados en los colores ya que el proceso de seleccion en base
    #? a matematica de vectores y la implmentacion de numpy es mucho mas rapida y eficiente que una aplicacio por apply
    #? o condiciones
    conditions = [
        (agg_price_volume['Profit_Margin_Pct'] > avg_margin) & (agg_price_volume['Avg_Price'] > avg_price),
        (agg_price_volume['Profit_Margin_Pct'] < avg_margin) & (agg_price_volume['Avg_Price'] < avg_price),
        (agg_price_volume['Profit_Margin_Pct'] < avg_margin) & (agg_price_volume['Avg_Price'] > avg_price),
    ]
    choices = [COLOR_ATTENTION, COLOR_RED, COLOR_PURPLE_DARK]
    agg_price_volume["Color"] = np.select(conditions, choices, default=COLOR_POS_LIGHT)

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
        annotation=dict(
            font=dict(size=10, family="Arial", style='italic'),
            bordercolor=COLOR_BG,
            borderwidth=1,
            borderpad=4,
            x=0.1
        )
    )

    #? Anadimos cada una de las burbujas por categoria en base del color correspondiente y los datos de su instancia
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

    #? Agregamos las etiquetas de los segmentos del scatter plot
    _add_color_labels_for_scatter_segmentations(fig, [
            (COLOR_ATTENTION, "Margen Alto y Precio Prom. Alto"),
            (COLOR_POS_LIGHT, "Margen Alto y Precio Prom. Bajo"),
            (COLOR_PURPLE_DARK, "Margen Bajo y Precio Prom. Alto"),
            (COLOR_RED, "Margen Bajo y Precio Prom. Bajo")
        ])

    #? Anadimos labels dinamicas al insight del grafico
    max_cat_by_average_sale = agg_price_volume.loc[agg_price_volume['Avg_Price'].idxmax(), 'Category']
    max_cat_by_average_sale_second_place = agg_price_volume.loc[
        agg_price_volume['Avg_Price'].nlargest(2).index[-1], 'Category']
    max_cat_by_average_sale_value = agg_price_volume.loc[agg_price_volume['Avg_Price'].idxmax(), 'Avg_Price']
    max_cat_by_margin = agg_price_volume.loc[agg_price_volume['Profit_Margin_Pct'].idxmax(), 'Category']
    max_cat_by_margin_value = agg_price_volume.loc[agg_price_volume['Profit_Margin_Pct'].idxmax(), 'Profit_Margin_Pct']

    #? Aplicamos el tema general al grafico
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


def plot_avg_customer_purchase_by_discount(agg_customer_purchase: pd.DataFrame) -> Figure | None:
    """
    Grafico de lineas que muestra la compra promedio por cliente segun rango de descuento por categoria
    :param agg_customer_purchase:
    :return:
    """
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



def plot_scatter_category(agg_data, min_sales=None, margin_range=None):
    """
    Grafico de dispersion que muestra la relacion entre ventas totales y margen de ganancia agrupadas por categoria
    :param agg_data:
    :param min_sales:
    :param margin_range:
    :return:
    """
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

    # Vectorized color assignment
    conditions = [
        (data["Profit_Margin_Pct"] > avg_margin) & (data["Total_Sales"] > avg_sales),
        (data["Profit_Margin_Pct"] < avg_margin) & (data["Total_Sales"] < avg_sales),
        (data["Profit_Margin_Pct"] < avg_margin) & (data["Total_Sales"] > avg_sales),
    ]
    choices = [COLOR_ATTENTION, COLOR_RED, COLOR_PURPLE_DARK]
    data["Color"] = np.select(conditions, choices, default=COLOR_POS_LIGHT)

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
            font=dict(size=10, family="Arial", style='italic'),
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
            'zeroline': False,
            'tickformat': ',.0f',
            'tickprefix': '$'
        },
        yaxis={
            'zeroline': False,
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

    # Vectorized color assignment
    conditions = [
        (agg_data_filtered["Profit_Margin_Pct"] > avg_margin) & (agg_data_filtered["Total_Sales"] > avg_sales),
        (agg_data_filtered["Profit_Margin_Pct"] < avg_margin) & (agg_data_filtered["Total_Sales"] < avg_sales),
        (agg_data_filtered["Profit_Margin_Pct"] < avg_margin) & (agg_data_filtered["Total_Sales"] > avg_sales),
    ]
    choices = [COLOR_ATTENTION, COLOR_RED, COLOR_PURPLE_DARK]
    agg_data_filtered["Color"] = np.select(conditions, choices, default=COLOR_POS_LIGHT)

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
        xaxis={'zeroline': False, 'tickformat': ',.0f', 'tickprefix': '$'},
        yaxis={
            'zeroline': False,
            'tickformat': '.0f',
            'ticksuffix': '%'
        }
    )

    return fig



def init_user_state() -> None:
    if "user_initialized" not in st.session_state:
        st.session_state.user_initialized = True
        st.session_state.active_filters = {}
        st.session_state.selected_view = "Visión Rentabilidad por Descuentos"
        st.session_state.prepared_data = None
        st.session_state.filtered_data = None
        st.session_state.aggregations = {}


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Funcion encargada de la carga de datos desde un archivo CSV o una URL remota.
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(
        script_dir,
        "..", "..", "..", "res", "processed_data",
        "ProyectoFinSemestreADM3083_SuperstoreProcessed.csv"
    )

    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path, sep=",")
    else:
        # Fallback to remote URL
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
    st.sidebar.markdown(
        "Filtros generales para toda la dashboard Estos filtros afectan a todas las visualizaciones y permiten segmentar los datos por tiempo, región, mecanismo de envío, y ubicación geográfica del cliente.")

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
          <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
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
              <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">MARGEN DE GANANCIA PROMEDIO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{overall_margin:.2f}%</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Benchmark de Rendimiento Global</p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        El rendimiento porcentual promedio se mantiene saludable bajo una política de descuentos controlada y segmentada. Pero aplicado descuentos altos, el márgen es negativo.
                    </p>
                </div>
              """)
        else:
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">MARGEN DE GANANCIA PROMEDIO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">--%</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Benchmark de Rendimiento Global</p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        No hay datos disponibles para los filtros seleccionados.
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
          <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">CLIENTES ÚNICOS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{unique_customers:,}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Total de órdenes: <b>{total_orders:,}</b></p>
                <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Representa una base consolidada con un promedio de <b>{total_orders / unique_customers if unique_customers > 0 else 0:.1f} órdenes</b> por cliente.
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
          <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
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
              <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER DE RENTABILIDAD POR CATEGORÍA</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{best_cat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{best_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        La categoría <b>{best_cat}</b> demuestra la mayor eficiencia en la conversión de ventas a beneficios netos.
                    </p>
                </div>
              """)
        else:
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER DE RENTABILIDAD POR CATEGORÍA</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>--%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        No hay datos disponibles para los filtros seleccionados.
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
              <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER EN VOLUMEN DE VENTAS</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{top_cat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ingresos Totales: <b>${top_sales:,.2f}</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        <b>{top_cat}</b> se consolida como el principal generador de flujo de caja para la organización.
                    </p>
                </div>
              """)
        else:
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LÍDER EN VOLUMEN DE VENTAS</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ingresos Totales: <b>$--</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        No hay datos disponibles para los filtros seleccionados.
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
            best_subcat_margin = margin_by_subcat.loc[margin_by_subcat["Profit_Margin_Pct"].idxmax()][
                "Profit_Margin_Pct"]
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_BROWN_DARK}, {COLOR_BROWN_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CON MEJOR DESEMPEÑO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{best_subcat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{best_subcat_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        Identificada como la unidad de negocio más rentable dentro de su respectiva categoría.
                    </p>
                </div>
              """)
        else:
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_BROWN_DARK}, {COLOR_BROWN_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CON MEJOR DESEMPEÑO</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>--%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        No hay datos disponibles para los filtros seleccionados.
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
              <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CRÍTICA (BAJO MARGEN)</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">{worst_subcat}</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>{worst_subcat_margin:.2f}%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        Requiere <b>atención inmediata</b> y revisión de la estructura de costos o estrategia de precios.
                    </p>
                </div>
              """)
        else:
            st.html(f"""
              <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">SUBCATEGORÍA CRÍTICA (BAJO MARGEN)</h4>
                    <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                    <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Promedio: <b>--%</b></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                        No hay datos disponibles para los filtros seleccionados.
                    </p>
                </div>
              """)

    # Select All option in mosaic layout
    f1, f2 = st.columns([2, 1])
    with f1:
        all_subs = sorted(data["Sub-Category"].dropna().unique().tolist())
        if st.toggle("Mostrar todas las Subcategorías Comerciales", value=True, key="line_select_all",
                     help="Permite definir una o más subcategorías para explorar la tendencia de su margen de ganancia por nivel de descuento."):
            sel_subs = all_subs
        else:
            sel_subs = st.multiselect(
                "Selecccionar Subcategorías a Mostrar", options=all_subs, default=all_subs, key="line_subs"
            )
    with f2:
        top_n = st.number_input("Mostrar Top N Subcategorías", min_value=1, value=5, key="line_top_n",
                                help="Permite definir el número de subcategorías a mostrar en el gráfico exploratorio del margen de ganancia por nivel de descuento. Si se activa la opción 'Mostrar todas las Subcategorías Comerciales', este filtro se deshabilita automáticamente.")

    if not aggs.get("margin_by_subcategory", pd.DataFrame()).empty:
        fig = plot_heatmap(aggs["margin_by_subcategory"], sel_subs,
                           top_n if not st.session_state.get("line_select_all") else None)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                "displaylogo": False,
                "displayModeBar": False,
                'modeBarButtonsToRemove': ['toImage', 'resetScale2d', 'fullscreen']
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
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÁXIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{best_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{best_margin['Sub_Category']}</b> ({best_margin['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Ventas Totales de ${best_margin['Total_Sales']:,.2f}.</p>
            </div>
            """)

        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD PROMEDIO GLOBAL</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{avg_margin:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Calculado sobre {len(subcat_data)} subcategorías analizadas.</p>
            </div>
            """)

        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÍNIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">{worst_margin['Profit_Margin_Pct']:.2f}%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{worst_margin['Sub_Category']}</b> ({worst_margin['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Requiere revisión estratégica de márgenes.</p>
            </div>
            """)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÁXIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">--%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>--</b> (--)</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
            </div>
            """)
        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD PROMEDIO GLOBAL</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">--%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
            </div>
            """)
        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">RENTABILIDAD MÍNIMA (SUBCATEGORÍA)</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">--%</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>--</b> (--)</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
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
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÁXIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${highest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{highest_sales['Sub_Category']}</b> ({highest_sales['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Margen Operativo: {highest_sales['Profit_Margin_Pct']:.2f}%.</p>
            </div>
            """)

        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VENTAS PROMEDIO POR UNIDAD</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_sales:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media Transaccional del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Ingresos Totales Brutos: ${subcat_data['Total_Sales'].sum():,.2f}.</p>
            </div>
            """)

        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÍNIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">${lowest_sales['Total_Sales']:,.2f}</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>{lowest_sales['Sub_Category']}</b> ({lowest_sales['Category']})</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">Segmento con baja tracción comercial.</p>
            </div>
            """)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÁXIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">$--</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>--</b> (--)</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
            </div>
            """)
        with c2:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_GREY}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VENTAS PROMEDIO POR UNIDAD</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">$--</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Media Transaccional del Portafolio</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
            </div>
            """)
        with c3:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_RED_DARK}, {COLOR_RED_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">VOLUMEN MÍNIMO DE VENTAS</h4>
                <h2 style="margin: 15px 0; font-size: 2.2em;">$--</h2>
                <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;"><b>--</b> (--)</p>
                <p style="margin: 10px 0 0 0; font-size: 0.85em; line-height: 1.4;">No hay datos disponibles.</p>
            </div>
            """)

    st.markdown("---")

    # --- By Category ---
    st.html("""
    <h3>Exploración de Margen de Ganancia Porcentual vs Total de Ventas por Categoría</h3>
    """)

    cc1, cc2 = st.columns(2)

    with cc1:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            cat_data = aggs["by_category"]
            best_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmax()]
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>{best_cat['Category']}</b> con un margen de <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b> es la categoría líder en rentabilidad dentro del dataset.
                </p>
            </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>--</b> con un margen de <b>--%</b> y facturación de <b>$--</b> es la categoría líder en rentabilidad dentro del dataset.
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
            <div style="background: linear-gradient(135deg, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'], worst_cat['Total_Sales'], average_margin_by_cat, average_sales_by_cat)},{_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'], worst_cat['Total_Sales'], average_margin_by_cat, average_sales_by_cat)}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>{worst_cat['Category']}</b> con un margen de <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f}</b> es la categoría sugerida para su revisión y seguimiento.
                </p>
            </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                    <b>--</b> con un margen de <b>--%</b> y facturación de <b>$--</b> es la categoría sugerida para su revisión y seguimiento.
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
            min_sales = st.number_input("Umbral de Ventas Totales Mínimas", value=0, key="scatter_cat_min_sales",
                                        format='%d')
    with f2:
        if not aggs.get("by_category", pd.DataFrame()).empty:
            data = aggs.get("by_category", pd.DataFrame())
            min_margin = int(data["Profit_Margin_Pct"].min()) * 1.15
            max_margin = int(data["Profit_Margin_Pct"].max()) * 1.15
            margin_range = st.slider(
                "Rango de Márgen de Ganancias ",
                min_value=-100, max_value=100, value=(-100, 100), key="scatter_cat_margin",
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
                <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>{best_cat['Sub_Category']}</b> con margen del <b>{best_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${best_cat['Total_Sales']:,.2f}</b> hija de la categoría <b>{best_cat['Category']}</b> es la subcategoría más rentable dentro del periodo de análisis.
                    </p>
                </div>
                """)
        else:
            st.html(f"""
                <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>--</b> con margen del <b>--%</b> y facturación de <b>$--</b> hija de la categoría <b>--</b> es la subcategoría más rentable dentro del periodo de análisis.
                    </p>
                </div>
                """)

    with csub2:
        if not aggs.get("by_subcategory", pd.DataFrame()).empty:
            cat_data = aggs["by_subcategory"]
            worst_cat = cat_data.loc[cat_data["Profit_Margin_Pct"].idxmin()]
            st.html(f"""
                <div style="background: linear-gradient(135deg, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'], worst_cat['Total_Sales'], average_margin_by_cat, average_sales_by_cat)}, {_return_color_for_cell_based_on_values(worst_cat['Profit_Margin_Pct'], worst_cat['Total_Sales'], average_margin_by_cat, average_sales_by_cat)}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>{worst_cat['Sub_Category']}</b> con margen del <b>{worst_cat['Profit_Margin_Pct']:.2f}%</b> y facturación de <b>${worst_cat['Total_Sales']:,.2f} </b> dentro de la categoría <b>{worst_cat['Category']} </b>corresponde a la categoría con el más bajo rendimiento en el periodo de análisis.
                    </p>
                </div>
                """)
        else:
            st.html(f"""
                <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_GREY}); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 0.95em; line-height: 1.5;">
                        <b>--</b> con margen del <b>--%</b> y facturación de <b>$--</b> dentro de la categoría <b>--</b> corresponde a la categoría con el más bajo rendimiento en el periodo de análisis.
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
            help="Mostrar las N subcategorías con mayores ventas totales"
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
            margin_range_sub = st.slider("Rango de Márgen de Ganancias ", -100, 100, (-100, 100),
                                         key="scatter_subcat_margin")

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
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LIDERAZGO EN PRECIOS PREMIUM</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">{highest_price_cat}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ticket Promedio: <b>${highest_price:,.2f}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    La categoría <b>{highest_price_cat}</b> se posiciona como el segmento de mayor valor transaccional, liderando la captura de ingresos por unidad vendida.
                  </p>
              </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">LIDERAZGO EN PRECIOS PREMIUM</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Ticket Promedio: <b>$--</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    No hay datos disponibles para los filtros seleccionados.
                  </p>
              </div>
            """)

    with c2:
        # Average customer purchase value (from customer_value_by_category)
        agg_cv = aggs.get("customer_value_by_category", pd.DataFrame())
        if not agg_cv.empty:
            total_sales = agg_cv["Total_Sales"].sum()
            unique_customers = agg_cv["Unique_Customers"].sum()
            avg_customer_value = total_sales / unique_customers if unique_customers > 0 else 0
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_PURPLE_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">CUSTOMER LIFETIME VALUE</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">${avg_customer_value:,.2f}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Base de Clientes: <b>{unique_customers:,}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    Este KPI establece el valor promedio generado por cada cliente único, sirviendo como benchmark crítico para estrategias de fidelización.
                  </p>
              </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_PURPLE_DARK}, {COLOR_PURPLE_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">CUSTOMER LIFETIME VALUE</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">$--</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Base de Clientes: <b>--</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    No hay datos disponibles para los filtros seleccionados.
                  </p>
              </div>
            """)

    with c3:
        # Most efficient pricing category (highest margin) - from price_volume_by_category
        if not agg_pv.empty:
            best_margin_cat = agg_pv.loc[agg_pv["Profit_Margin_Pct"].idxmax(), "Category"]
            best_margin = agg_pv["Profit_Margin_Pct"].max()
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">EFICIENCIA OPERATIVA DE PRECIOS</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">{best_margin_cat}</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Operativo: <b>{best_margin:.2f}%</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    <b>{best_margin_cat}</b> demuestra la estructura de precios más saludable, optimizando el retorno sobre cada dólar invertido por el cliente.
              </p>
          </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(135deg, {COLOR_POS_DARK}, {COLOR_POS_LIGHT}); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                  <h4 style="margin: 0; font-weight: 300; letter-spacing: 1px;">EFICIENCIA OPERATIVA DE PRECIOS</h4>
                  <h2 style="margin: 15px 0; font-size: 2.2em;">--</h2>
                  <p style="margin: 5px 0; font-size: 1.1em; opacity: 0.9;">Margen Operativo: <b>--%</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; line-height: 1.4;">
                    No hay datos disponibles para los filtros seleccionados.
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
            <div style="background: linear-gradient(90deg, {COLOR_BROWN_DARK}, {COLOR_BROWN_LIGHT}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                  <h4 style="margin: 0; color: {COLOR_WHITE}; font-weight: 400; letter-spacing: 0.5px;">GENERACIÓN DE INGRESOS BRUTOS</h4>
                  <h2 style="margin: 10px 0; color: {COLOR_WHITE}; font-size: 2em;">${top_revenue:,.2f}</h2>
                  <p style="margin: 5px 0; color: {COLOR_WHITE};">Líder: <b>{top_revenue_cat}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; color: {COLOR_WHITE}; line-height: 1.4;">Aporta el motor principal de flujo de caja para la operación global.</p>
              </div>
            """)
        else:
            st.html(f"""
            <div style="background: linear-gradient(90deg, {COLOR_BROWN_DARK}, {COLOR_BROWN_LIGHT}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                  <h4 style="margin: 0; color: {COLOR_WHITE}; font-weight: 400; letter-spacing: 0.5px;">GENERACIÓN DE INGRESOS BRUTOS</h4>
                  <h2 style="margin: 10px 0; color: {COLOR_WHITE}; font-size: 2em;">$--</h2>
                  <p style="margin: 5px 0; color: {COLOR_WHITE};">Líder: <b>--</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.9em; color: {COLOR_WHITE}; line-height: 1.4;">No hay datos disponibles para los filtros seleccionados.</p>
              </div>
            """)

    with c2:
        # Transaction count
        total_transactions = len(data)
        avg_transaction_value = data['Sales'].mean()
        st.html(f"""
        <div style="background: linear-gradient(90deg, {COLOR_BROWN_DARK}, {COLOR_BROWN_LIGHT}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_WHITE}; font-weight: 400; letter-spacing: 0.5px;">VELOCIDAD Y TRACCIÓN COMERCIAL</h4>
              <h2 style="margin: 10px 0; color: {COLOR_WHITE}; font-size: 2em;">{total_transactions:,} Órdenes</h2>
              <p style="margin: 5px 0; color: {COLOR_WHITE};">Ticket Medio: <b>${avg_transaction_value:,.2f}</b></p>
              <p style="margin: 10px 0 0 0; font-size: 0.9em; color: {COLOR_WHITE}; line-height: 1.4;">Refleja la intensidad de la demanda y la recurrencia operativa del portafolio.</p>
          </div>
        """)

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
            <div style="background-color: {COLOR_POS_DARK}; padding: 25px; border-radius: 15px; color: white;">
                  <h4 style="margin: 0; font-weight: 300;">LÍDER DE SEGMENTO DE ALTO VALOR</h4>
                  <h2 style="margin: 10px 0; font-size: 2em;">{best_cat}</h2>
                  <p style="margin: 5px 0;">Lealtad de Gasto: <b>${best_value:,.2f}</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.85em; opacity: 0.9;">Segmento con mayor disposición de pago y potencial de rentabilidad.</p>
              </div>
            """)
        else:
            st.html(f"""
            <div style="background-color: {COLOR_POS_DARK}; padding: 25px; border-radius: 15px; color: white;">
                  <h4 style="margin: 0; font-weight: 300;">LÍDER DE SEGMENTO DE ALTO VALOR</h4>
                  <h2 style="margin: 10px 0; font-size: 2em;">--</h2>
                  <p style="margin: 5px 0;">Lealtad de Gasto: <b>$--</b></p>
                  <p style="margin: 10px 0 0 0; font-size: 0.85em; opacity: 0.9;">No hay datos disponibles para los filtros seleccionados.</p>
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

    aggregations = get_aggregations(filtered)

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
