import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# 1. Imports y configuración
# =========================
import os
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","..","..","res","data", "coffee_shop_sales.xlsx")

MONTH_ORDER = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
DAY_ORDER = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
TIME_ORDER = ["Mañana", "Mediodía", "Tarde", "Noche"]

COLORS = {
    "coffee_brown": "#6F4E37",
    "roast_brown": "#A67B5B",
    "cream": "#F5E6D3",
    "light_beige": "#FAF7F2",
    "dark_espresso": "#3E2723",
    "accent_blue": "#3B82F6",
    "success_green": "#10B981",
    "warning_orange": "#F59E0B",
    "danger_red": "#EF4444",
}

COLOR_WHITE = "#FFFFFF"


def _axis_label_for_metric(metric: str) -> str:
    """Human-friendly labels for metrics (avoid raw column names)."""
    return {
        "revenue": "Ingresos ($)",
        "transaction_id": "Transacciones",
        "transaction_qty": "Unidades",
        "transacciones": "Transacciones",
        "ticket_promedio": "Ticket promedio ($)",
        "items": "Items",
        "revenue_txn": "Ingreso por transaccion ($)",
        "transacciones_high_value": "Transacciones (alto valor)",
        "percentage": "% crecimiento",
        "absolute": "Crecimiento absoluto ($)",
        "indexed": "Indice (enero=100)",
        "pct_growth": "% crecimiento",
        "abs_growth": "Crecimiento absoluto ($)",
    }.get(metric, metric)


def _contrast_text_colors(values: pd.Series, threshold: float = .3) -> list[str]:
    """Return per-point text colors for a light->dark continuous scale.

    Assumes low values map to lighter colors (needs darker text) and high values
    map to darker colors (needs white text).
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.empty:
        return []
    vmin = float(np.nanmin(v.to_numpy())) if np.isfinite(np.nanmin(v.to_numpy())) else np.nan
    vmax = float(np.nanmax(v.to_numpy())) if np.isfinite(np.nanmax(v.to_numpy())) else np.nan
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return [COLOR_WHITE] * len(v)

    t = (v - vmin) / (vmax - vmin)
    # Lower quantiles are lighter on YlOrBr -> use near-black.
    dark = "#111827"
    return [dark if (float(x) < threshold) else COLOR_WHITE for x in t.fillna(1.0).to_list()]


def _parse_rgb(rgb: str) -> tuple[int, int, int]:
    # Expect strings like "rgb(255, 0, 0)".
    rgb = rgb.strip()
    if rgb.startswith("rgb(") and rgb.endswith(")"):
        parts = rgb[4:-1].split(",")
        r, g, b = (int(float(p.strip())) for p in parts[:3])
        return r, g, b
    if rgb.startswith("#") and len(rgb) == 7:
        return int(rgb[1:3], 16), int(rgb[3:5], 16), int(rgb[5:7], 16)
    # Fallback to white.
    return 255, 255, 255


def _relative_luminance(r: int, g: int, b: int) -> float:
    # Simple luminance proxy on [0,1].
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _heatmap_text_colors(z: np.ndarray, colorscale, light_cutoff: float = 0.62) -> list[str]:
    """Choose black/white text per cell based on sampled colorscale luminance."""
    import plotly.colors as pc

    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return []

    valid = np.isfinite(z)
    if not valid.any():
        return [COLOR_WHITE] * int(z.size)

    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))
    if vmin == vmax:
        return [COLOR_WHITE] * int(z.size)

    norm = (z - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    dark = "#111827"
    out: list[str] = []
    for v, ok in zip(norm.flatten(order="C"), valid.flatten(order="C")):
        if not ok:
            out.append(dark)
            continue
        color = pc.sample_colorscale(colorscale, [float(v)], colortype="rgb")[0]
        r, g, b = _parse_rgb(color)
        lum = _relative_luminance(r, g, b)
        out.append(dark if lum >= light_cutoff else COLOR_WHITE)
    return out


def _fmt_cell_value(metric: str, value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if metric in {"revenue", "revenue_txn", "ticket_promedio", "absolute", "abs_growth"}:
        return _fmt_currency(float(value), 0)
    if metric in {"pct_growth", "percentage"}:
        return f"{float(value):+.1f}%"
    if metric in {"indexed"}:
        return f"{float(value):.0f}"
    return _fmt_number(float(value), 0)


def _st_html(html: str) -> None:
    # Streamlit added st.html in newer versions; fall back to markdown.
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def _fmt_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    fmt = f"{{:,.{decimals}f}}" if decimals > 0 else "{:,.0f}"
    return fmt.format(value)


def _fmt_currency(value: float | int | None, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "$--"
    fmt = f"${{:,.{decimals}f}}" if decimals > 0 else "${:,.0f}"
    return fmt.format(value)


def render_kpi_card(
    title: str,
    value_text: str,
    leader_text: str,
    body_text: str,
    color_a: str = COLORS["coffee_brown"],
    color_b: str = COLORS["roast_brown"],
) -> None:
    _st_html(
        f"""
        <div style="background: linear-gradient(90deg, {color_a}, {color_b}); padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
              <h4 style="margin: 0; color: {COLOR_WHITE}; font-weight: 400; letter-spacing: 0.5px;">{title}</h4>
              <h2 style="margin: 10px 0; color: {COLOR_WHITE}; font-size: 2em;">{value_text}</h2>
              <p style="margin: 5px 0; color: {COLOR_WHITE};">{leader_text}</p>
              <p style="margin: 10px 0 0 0; font-size: 0.95em; color: {COLOR_WHITE}; line-height: 1.4;">{body_text}</p>
          </div>
        """
    )

CATEGORY_COLORS = {
    "Coffee": COLORS["coffee_brown"],
    "Tea": COLORS["roast_brown"],
    "Bakery": COLORS["accent_blue"],
    "Drinking Chocolate": COLORS["warning_orange"],
    "Coffee beans": COLORS["dark_espresso"],
    "Flavours": COLORS["success_green"],
    "Loose Tea": "#8B5E3C",
    "Packaged Chocolate": "#B98E68",
    "Branded": "#9CA3AF",
}

# Ensure 'Other' has a color
CATEGORY_COLORS.setdefault("Other", "#9CA3AF")

def _group_top_n_categories(df: pd.DataFrame, category_col: str = "product_category", value_col: str = "revenue", top_n: int = 10) -> pd.DataFrame:
    """Return a copy of df where categories outside the top_n by value_col are replaced with 'Other'.

    The top_n is computed within the provided dataframe df.
    """
    top_n = int(max(1, top_n))
    if category_col not in df.columns or value_col not in df.columns:
        return df.copy()
    base = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    if base.empty:
        return df.copy()
    # If top_n covers all categories, nothing to do
    if len(base) <= top_n:
        return df.copy()
    top_cats = set(base.head(top_n).index.tolist())
    out = df.copy()
    out[category_col] = out[category_col].where(out[category_col].isin(top_cats), other="Other")
    return out


# =========================
# 2. CSS styling
# =========================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {COLORS['light_beige']};
                color: {COLORS['dark_espresso']};
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {COLORS['dark_espresso']};
            }}
            .kpi-card {{
                background: white;
                border-left: 5px solid {COLORS['coffee_brown']};
                padding: 0.8rem 1rem;
                border-radius: 12px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            }}
            .finding-box {{
                background: {COLORS['cream']};
                border: 1px solid {COLORS['roast_brown']};
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }}
            .rec-card {{
                background: white;
                border: 1px solid #E5E7EB;
                border-left: 6px solid {COLORS['coffee_brown']};
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 3. Helper functions
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga y prepara datos con tipado e ingeniería de variables."""
    df = pd.read_excel(path)

    month_map = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }
    day_map = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves", 4: "viernes", 5: "sábado", 6: "domingo"}

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["transaction_time"] = pd.to_datetime(df["transaction_time"].astype(str), format="%H:%M:%S", errors="coerce")

    df["transaction_qty"] = pd.to_numeric(df["transaction_qty"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    df["month"] = df["transaction_date"].dt.month
    df["month_name"] = pd.Categorical(df["month"].map(month_map), categories=MONTH_ORDER, ordered=True)
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["day_name"] = pd.Categorical(df["day_of_week"].map(day_map), categories=DAY_ORDER, ordered=True)
    df["hour"] = df["transaction_time"].dt.hour
    df["week"] = df["transaction_date"].dt.isocalendar().week.astype("Int64")

    conds = [
        (df["hour"] >= 6) & (df["hour"] < 11),
        (df["hour"] >= 11) & (df["hour"] < 14),
        (df["hour"] >= 14) & (df["hour"] < 17),
        (df["hour"] >= 17),
    ]
    labels = ["Mañana", "Mediodía", "Tarde", "Noche"]
    df["time_of_day"] = pd.Categorical(np.select(conds, labels, default="Fuera de horario"), categories=TIME_ORDER, ordered=True)

    return df.dropna(subset=["transaction_date", "transaction_time", "transaction_qty", "unit_price", "revenue"])


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def calculate_kpis(df: pd.DataFrame) -> dict:
    by_month = df.groupby(["month", "month_name"], observed=False)["revenue"].sum().reset_index().sort_values("month")
    jan = by_month.loc[by_month["month"] == 1, "revenue"].sum()
    jun = by_month.loc[by_month["month"] == 6, "revenue"].sum()
    mom_growth = ((jun / jan) - 1) * 100 if jan > 0 else np.nan

    store_rev = df.groupby("store_location", observed=False)["revenue"].sum().sort_values(ascending=False)
    cat_rev = df.groupby("product_category", observed=False)["revenue"].sum().sort_values(ascending=False)
    hour_rev = df.groupby("hour", observed=False)["revenue"].sum().sort_values(ascending=False)

    txn_items = df.groupby("transaction_id", observed=False)["transaction_qty"].sum()
    multi_item_pct = (txn_items >= 2).mean() * 100
    morning_txn_pct = (df["time_of_day"] == "Mañana").mean() * 100

    return {
        "total_revenue": df["revenue"].sum(),
        "total_transactions": int(df["transaction_id"].nunique()),
        "avg_ticket": df["revenue"].sum() / df["transaction_id"].nunique(),
        "best_store": store_rev.index[0] if not store_rev.empty else "N/A",
        "top_category": cat_rev.index[0] if not cat_rev.empty else "N/A",
        "top_category_share": (cat_rev.iloc[0] / cat_rev.sum() * 100) if not cat_rev.empty else np.nan,
        "peak_hour": int(hour_rev.index[0]) if not hour_rev.empty else np.nan,
        "best_month": by_month.sort_values("revenue", ascending=False).iloc[0]["month_name"] if not by_month.empty else "N/A",
        "mom_growth": mom_growth,
        "multi_item_pct": multi_item_pct,
        "morning_txn_pct": morning_txn_pct,
    }


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("☕ Filtros Globales")

    min_date = df["transaction_date"].min().date()
    max_date = df["transaction_date"].max().date()
    selected_dates = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    stores = sorted(df["store_location"].dropna().unique().tolist())
    categories = sorted(df["product_category"].dropna().unique().tolist())
    product_types = sorted(df["product_type"].dropna().unique().tolist())
    times = [x for x in TIME_ORDER if x in df["time_of_day"].dropna().unique().tolist()]

    selected_stores = st.sidebar.multiselect("Tienda", stores, default=stores, key="sidebar_stores")
    selected_categories = st.sidebar.multiselect("Categoría de producto", categories, default=categories, key="sidebar_categories")
    selected_types = st.sidebar.multiselect("Tipo de producto", product_types, default=product_types, key="sidebar_types")
    selected_times = st.sidebar.multiselect("Franja horaria (opcional)", times, default=times, key="sidebar_times")

    mask = pd.Series(True, index=df.index)

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        mask &= (df["transaction_date"].dt.date >= start_date) & (df["transaction_date"].dt.date <= end_date)

    if selected_stores:
        mask &= df["store_location"].isin(selected_stores)
    if selected_categories:
        mask &= df["product_category"].isin(selected_categories)
    if selected_types:
        mask &= df["product_type"].isin(selected_types)
    if selected_times:
        mask &= df["time_of_day"].isin(selected_times)

    return df.loc[mask].copy()


# =========================
# 4. Chart functions
# =========================
def style_fig(
    fig,
    title: str,
    subtitle: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
):
    """Apply standard theme to figure matching the other dashboard's styling"""
    # Armamos la cadena del titulo con el formato adecuado
    full_title = f"<b>{title}</b><br><i><span style='font-size:12px;font-weight:200;color:#666'>{subtitle}</span></i>"
    
    fig.update_layout(
        title={
            "text": full_title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.92,
            "font": {"size": 16, "family": "Arial"},
        },
        autosize=True,
        height=500,  # Keep consistent height; Streamlit container controls width
        margin=dict(
            l=100,
            r=50,
            b=50,
            t=100,
            pad=4
        ),
        # Remove grid lines but keep styling similar to other dashboard
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="#5A5A5C",
            ticks='outside',
            tickwidth=2,
            tickcolor="#5A5A5C",
            tickangle=30,
            tickfont={"size": 11, "family": "Arial"},
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            title="",
            automargin=True,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Axis titles: keep professional labels (even if tick labels are hidden).
    if x_title:
        fig.update_xaxes(title_text=x_title, title_standoff=14, automargin=True)
    if y_title:
        fig.update_yaxes(title_text=y_title, title_standoff=14, automargin=True)

    # Storytelling: keep labels inside bars when present.
    try:
        fig.update_traces(textposition="inside", selector=dict(type="bar"))
        # Allow HTML in bar labels (for bold/italic).
        fig.update_traces(texttemplate="%{text}", selector=dict(type="bar"))
    except Exception:
        pass

    # Preserve per-bar text colors when caller sets them.
    try:
        for tr in fig.data:
            if getattr(tr, "type", None) != "bar":
                continue
            tf = getattr(tr, "textfont", None)
            current = getattr(tf, "color", None) if tf is not None else None
            if current is None:
                tr.textfont = dict(color=COLOR_WHITE)
    except Exception:
        pass
    # Enforce month order if month_name present on x-axis as categories
    try:
        xaxis = fig.layout.xaxis
        if getattr(xaxis, "categoryarray", None) is None and getattr(xaxis, "tickvals", None) is None:
            # if x-axis contains month names, set categoryorder
            fig.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    except Exception:
        pass
    return fig


def chart_monthly_revenue(df: pd.DataFrame):
    d = df.groupby(["month", "month_name"], observed=False)["revenue"].sum().reset_index().sort_values("month")
    # Filter out months with zero revenue to avoid connecting unrelated months in the line chart
    d = d[d["revenue"] > 0]
    fig = px.line(
        d,
        x="month_name",
        y="revenue",
        markers=True,
        color_discrete_sequence=[COLORS["coffee_brown"]],
        labels={"month_name": "Mes", "revenue": _axis_label_for_metric("revenue")},
    )
    # Calculate insight for monthly revenue
    if len(d) >= 2:
        max_month = d.loc[d['revenue'].idxmax()]
        min_month = d.loc[d['revenue'].idxmin()]
        insight = f"Mes más alto: {max_month['month_name']} (${max_month['revenue']:,.0f}). Mes más bajo: {min_month['month_name']} (${min_month['revenue']:,.0f})."
    else:
        insight = "Datos insuficientes para generar insights."
    return style_fig(fig, "Ingresos por mes", insight, x_title="Mes", y_title=_axis_label_for_metric("revenue"))


def chart_store_revenue(df: pd.DataFrame):
    d = df.groupby("store_location", observed=False)["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    fig = px.bar(
        d,
        x="store_location",
        y="revenue",
        color="store_location",
        color_discrete_sequence=[COLORS["coffee_brown"], COLORS["roast_brown"], COLORS["accent_blue"]],
        text=d["revenue"].apply(lambda x: f"${x:,.0f}"),
        labels={"store_location": "Tienda", "revenue": _axis_label_for_metric("revenue")},
    )
    fig.update_traces(textposition='inside')
    # Update y-axis to better showcase differences while keeping minimum at zero
    max_revenue = d["revenue"].max()
    fig.update_yaxes(range=[0, max_revenue * 1.03])  # Add 3% padding above max value
    # Calculate insight for store revenue
    if len(d) >= 1:
        max_store = d.iloc[0]
        min_store = d.iloc[-1]
        insight = f"Tienda líder: {max_store['store_location']} (${max_store['revenue']:,.0f}). Diferencia: ${max_store['revenue'] - min_store['revenue']:,.0f}."
    else:
        insight = "Datos insuficientes para generar insights."
    return style_fig(fig, "Ingresos por tienda", insight, x_title="Tienda", y_title=_axis_label_for_metric("revenue"))


def chart_category_donut(df: pd.DataFrame):
    return chart_category_donut_topn(df, top_n=10)


def chart_category_donut_topn(df: pd.DataFrame, top_n: int = 10):
    """Donut chart with Top N categories + Other bucket."""
    base = (
        df.groupby("product_category", observed=False)["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    if base.empty:
        d = pd.DataFrame({"product_category": [], "revenue": []})
    else:
        top_n = int(max(1, top_n))
        top = base.head(top_n)
        other_val = float(base.iloc[top_n:].sum()) if len(base) > top_n else 0.0
        d = top.reset_index()
        if other_val > 0:
            d = pd.concat(
                [d, pd.DataFrame({"product_category": ["Other"], "revenue": [other_val]})],
                ignore_index=True,
            )

    # Build a color map that includes "Other".
    color_map = dict(CATEGORY_COLORS)
    color_map.setdefault("Other", "#9CA3AF")

    fig = px.pie(
        d,
        names="product_category",
        values="revenue",
        hole=0.55,
        color="product_category",
        color_discrete_map=color_map,
        labels={"product_category": "Categoría", "revenue": _axis_label_for_metric("revenue")},
    )
    fig.update_layout(legend_title_text="Categoría")
    subtitle = f"Top {min(top_n, len(base))} categorías y 'Other'" if not base.empty else ""
    return style_fig(fig, "Participación de ingresos por categoría", subtitle)


def create_pareto(df: pd.DataFrame, top_n: int = 20):
    d = df.groupby("product_type", observed=False)["revenue"].sum().sort_values(ascending=False).reset_index()
    d["cum_pct"] = d["revenue"].cumsum() / d["revenue"].sum() * 100
    d = d.head(top_n)
    fig = px.bar(
        d,
        x="product_type",
        y="revenue",
        color_discrete_sequence=[COLORS["coffee_brown"]],
        text=d["revenue"].apply(lambda x: f"${x:,.0f}"),
        labels={"product_type": "Producto", "revenue": _axis_label_for_metric("revenue")},
    )
    fig.update_traces(textposition='inside')
    fig.add_scatter(x=d["product_type"], y=d["cum_pct"], mode="lines+markers", name="% acumulado", yaxis="y2")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="% acumulado"))
    return style_fig(fig, "Gráfico Pareto de productos", x_title="Producto", y_title=_axis_label_for_metric("revenue"))


def create_heatmap_day_hour(df: pd.DataFrame, metric: str = "revenue"):
    pivot = df.pivot_table(index="day_name", columns="hour", values=metric, aggfunc="sum").reindex(DAY_ORDER)
    colorscale = [[0, COLORS["cream"]], [1, COLORS["coffee_brown"]]]
    fig = px.imshow(
        pivot,
        color_continuous_scale=colorscale,
        aspect="auto",
        labels={"x": "Hora", "y": "Día de semana", "color": _axis_label_for_metric(metric)},
    )

    # Add per-cell annotations with contrast-aware text colors.
    z = np.asarray(pivot.to_numpy(dtype=float))
    flat_colors = _heatmap_text_colors(z, colorscale)
    flat_text = [_fmt_cell_value(metric, v) for v in z.flatten(order="C")]
    xs = list(pivot.columns)
    ys = list(pivot.index)
    xi, yi = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    fig.add_trace(
        go.Scatter(
            x=[xs[i] for i in xi.flatten(order="C")],
            y=[ys[j] for j in yi.flatten(order="C")],
            mode="text",
            text=flat_text,
            textfont=dict(color=flat_colors, size=11),
            hoverinfo="skip",
            showlegend=False,
        )
    )


    style_fig(fig, "Heatmap: día de semana × hora", x_title="Hora", y_title="Día de semana")

    return fig.update_layout(yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True))


def create_heatmap_store_category(df: pd.DataFrame, metric: str = "revenue"):
    pivot = df.pivot_table(index="store_location", columns="product_category", values=metric, aggfunc="sum").fillna(0)
    # try to keep category ordering from CATEGORY_COLORS when available
    cols = [c for c in CATEGORY_COLORS.keys() if c in pivot.columns]
    if cols:
        pivot = pivot.reindex(columns=cols)
    colorscale = "RdYlGn"
    fig = px.imshow(
        pivot,
        color_continuous_scale=colorscale,
        aspect="auto",
        labels={"x": "Categoría", "y": "Tienda", "color": _axis_label_for_metric(metric)},
    )

    z = np.asarray(pivot.to_numpy(dtype=float))
    flat_colors = _heatmap_text_colors(z, colorscale)
    flat_text = [_fmt_cell_value(metric, v) for v in z.flatten(order="C")]
    xs = list(pivot.columns)
    ys = list(pivot.index)
    xi, yi = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    fig.add_trace(
        go.Scatter(
            x=[xs[i] for i in xi.flatten(order="C")],
            y=[ys[j] for j in yi.flatten(order="C")],
            mode="text",
            text=flat_text,
            textfont=dict(color=flat_colors, size=11),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    return style_fig(fig, "Heatmap: tienda × categoría", x_title="Categoría", y_title="Tienda")


def render_overview(df: pd.DataFrame, full_df: pd.DataFrame):
    st.subheader("Resumen Ejecutivo")
    kpi = calculate_kpis(df)
    full_kpi = calculate_kpis(full_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingresos Totales", format_currency(kpi["total_revenue"]))
    c2.metric("Transacciones Totales", f"{kpi['total_transactions']:,}")
    c3.metric("Ticket Promedio", format_currency(kpi["avg_ticket"]))
    c4.metric("Mejor Tienda", kpi["best_store"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Categoría Top", f"{kpi['top_category']} ({kpi['top_category_share']:.1f}%)")
    c6.metric("Hora Pico", f"{kpi['peak_hour']}:00")
    c7.metric("Mes de Mayor Ingreso", str(kpi["best_month"]).capitalize())
    c8.metric("Crecimiento Ene-Jun", f"{kpi['mom_growth']:.2f}%")

    st.markdown("#### Hallazgos Clave")
    findings = [
        f"<b>{full_kpi['morning_txn_pct']:.2f}%</b> de transacciones ocurren en horario matutino.",
        "<b>10 productos</b> generan el <b>80%</b> de los ingresos (Pareto).",
        "<b>Lower Manhattan</b> tiene el ticket promedio más alto (<b>$4.81</b>).",
        "<b>41.5%</b> de transacciones incluyen múltiples items.",
        "Se observa **crecimiento sostenido** mes a mes.",
    ]
    for text in findings:
        st.markdown(f"<div class='finding-box'>{text}</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        monthly = (
            df.groupby(["month", "month_name"], observed=False)["revenue"]
            .sum()
            .reset_index()
            .sort_values("month")
        )
        monthly = monthly[monthly["revenue"] > 0]
        if monthly.empty:
            render_kpi_card(
                "INGRESOS POR MES",
                "$--",
                "Líder: <b>--</b>",
                "No hay datos disponibles para los filtros seleccionados.",
            )
        else:
            peak = monthly.loc[monthly["revenue"].idxmax()]
            floor = monthly.loc[monthly["revenue"].idxmin()]
            render_kpi_card(
                "INGRESOS POR MES",
                _fmt_currency(peak["revenue"], 0),
                f"Líder: <b>{peak['month_name']}</b>",
                f"Rango del periodo: {floor['month_name']} ({_fmt_currency(floor['revenue'], 0)}) a {peak['month_name']} ({_fmt_currency(peak['revenue'], 0)}).",
            )
        st.plotly_chart(chart_monthly_revenue(df), use_container_width=True, key="overview_monthly")
    with col_b:
        store_rev = df.groupby("store_location", observed=False)["revenue"].sum().sort_values(ascending=False)
        if store_rev.empty:
            render_kpi_card(
                "INGRESOS POR TIENDA",
                "$--",
                "Líder: <b>--</b>",
                "No hay datos disponibles para los filtros seleccionados.",
            )
        else:
            leader_store = store_rev.index[0]
            leader_value = float(store_rev.iloc[0])
            spread = float(store_rev.iloc[0] - store_rev.iloc[-1]) if len(store_rev) > 1 else 0.0
            render_kpi_card(
                "INGRESOS POR TIENDA",
                _fmt_currency(leader_value, 0),
                f"Líder: <b>{leader_store}</b>",
                f"Brecha líder vs. último: {_fmt_currency(spread, 0)}.",
            )
        st.plotly_chart(chart_store_revenue(df), use_container_width=True, key="overview_store")
    cat_rev = df.groupby("product_category", observed=False)["revenue"].sum().sort_values(ascending=False)
    if cat_rev.empty:
        render_kpi_card(
            "MIX DE INGRESOS (CATEGORÍAS)",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    else:
        top_cat = cat_rev.index[0]
        top_val = float(cat_rev.iloc[0])
        share = (top_val / float(cat_rev.sum()) * 100.0) if float(cat_rev.sum()) > 0 else np.nan
        render_kpi_card(
            "MIX DE INGRESOS (CATEGORÍAS)",
            _fmt_currency(float(cat_rev.sum()), 0),
            f"Líder: <b>{top_cat}</b>",
            f"Top categoría aporta {share:.1f}% del ingreso ({_fmt_currency(top_val, 0)}).",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    n_cats = int(df["product_category"].nunique())
    max_topn = max(3, min(12, n_cats))
    topn_cat = st.slider(
        "Top N categorías (donut)",
        min_value=3,
        max_value=max_topn,
        value=min(7, max_topn),
        step=1,
        key="overview_donut_topn",
    )
    st.plotly_chart(chart_category_donut_topn(df, top_n=topn_cat), use_container_width=True, key="overview_category_donut")


def render_q1(df: pd.DataFrame):
    st.subheader("Q1 — Ventas en el Tiempo y Momentos del Día")
    c1, c2, c3 = st.columns(3)
    metric_labels = {"revenue": "Ingresos", "transaction_id": "Transacciones", "transaction_qty": "Unidades"}
    metric = c1.selectbox("Métrica", ["revenue", "transaction_id", "transaction_qty"], format_func=lambda x: metric_labels[x])
    stores = sorted(df["store_location"].unique())
    sel_store = c2.multiselect("Tienda (local)", stores, default=stores, key="q1_store_local")
    times = [x for x in TIME_ORDER if x in df["time_of_day"].unique()]
    sel_time = c3.multiselect("Franja horaria (local)", times, default=times, key="q1_time_local")

    d = df[df["store_location"].isin(sel_store) & df["time_of_day"].isin(sel_time)]
    # If no local filters selected or the resulting dataset is empty, show a default message and stop
    if d.empty:
        render_kpi_card(
            "Q1 | TENDENCIA MENSUAL",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
        )
        st.stop()
    aggfunc = "sum" if metric != "transaction_id" else "nunique"

    monthly = d.groupby(["month", "month_name"], observed=False)[metric].agg(aggfunc).reset_index().sort_values("month")
    # Filter out months with zero values to avoid connecting unrelated months in the line chart
    monthly = monthly[monthly[metric] > 0]
    monthly["mom_growth_pct"] = monthly[metric].pct_change() * 100

    if monthly.empty:
        render_kpi_card(
            "Q1 | TENDENCIA MENSUAL",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
        )
    else:
        peak = monthly.loc[monthly[metric].idxmax()]
        total_val = monthly[metric].sum() if aggfunc == "sum" else d["transaction_id"].nunique()
        val_text = _fmt_currency(float(total_val), 0) if metric == "revenue" else _fmt_number(float(total_val), 0)
        render_kpi_card(
            "Q1 | TENDENCIA MENSUAL",
            val_text,
            f"Líder: <b>{peak['month_name']}</b>",
            f"Pico mensual: {_fmt_currency(float(peak[metric]), 0) if metric == 'revenue' else _fmt_number(float(peak[metric]), 0)}.",
        )
    subtitle = "" if monthly.empty else (
        f"{metric_labels[metric]}: pico en {peak['month_name']} | MoM último: {monthly['mom_growth_pct'].iloc[-1]:.1f}%" if len(monthly) >= 2 and pd.notna(monthly["mom_growth_pct"].iloc[-1]) else f"{metric_labels[metric]}: pico en {peak['month_name']}"
    )
    st.plotly_chart(
        style_fig(
            px.line(
                monthly,
                x="month_name",
                y=metric,
                markers=True,
                color_discrete_sequence=[COLORS["coffee_brown"]],
                labels={"month_name": "Mes", metric: _axis_label_for_metric(metric)},
            ),
            f"{metric_labels[metric]} mensuales",
            subtitle,
        ),
        use_container_width=True,
        key="q1_monthly",
    )

    daily = d.groupby("transaction_date", observed=False)[metric].agg(aggfunc).reset_index()
    if daily.empty:
        render_kpi_card(
            "Q1 | TENDENCIA DIARIA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
            color_a=COLORS["roast_brown"],
            color_b=COLORS["coffee_brown"],
        )
    else:
        peak_day = daily.loc[daily[metric].idxmax()]
        last_day = daily.iloc[-1]
        val_text = _fmt_currency(float(last_day[metric]), 0) if metric == "revenue" else _fmt_number(float(last_day[metric]), 0)
        render_kpi_card(
            "Q1 | TENDENCIA DIARIA",
            val_text,
            f"Líder: <b>{pd.to_datetime(peak_day['transaction_date']).date()}</b>",
            f"Día pico: {_fmt_currency(float(peak_day[metric]), 0) if metric == 'revenue' else _fmt_number(float(peak_day[metric]), 0)}.",
            color_a=COLORS["roast_brown"],
            color_b=COLORS["coffee_brown"],
        )
    daily_sub = "" if daily.empty else (
        f"Día pico: {pd.to_datetime(peak_day['transaction_date']).date()} | {metric_labels[metric]}: {_fmt_currency(float(peak_day[metric]), 0) if metric == 'revenue' else _fmt_number(float(peak_day[metric]), 0)}"
    )
    st.plotly_chart(
        style_fig(
            px.line(
                daily,
                x="transaction_date",
                y=metric,
                color_discrete_sequence=[COLORS["roast_brown"]],
                labels={"transaction_date": "Fecha", metric: _axis_label_for_metric(metric)},
            ),
            f"Tendencia diaria ({metric_labels[metric]})",
            daily_sub,
        ),
        use_container_width=True,
        key="q1_daily",
    )

    dow = d.groupby("day_name", observed=False)[metric].agg(aggfunc).reindex(DAY_ORDER).reset_index()
    # Format text based on metric type
    if metric == "revenue":
        text_values = dow[metric].apply(lambda x: f"${x:,.0f}")
    else:
        text_values = dow[metric].apply(lambda x: f"{x:,.0f}")
    
    if dow.empty:
        render_kpi_card(
            "Q1 | DÍA DE SEMANA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
    else:
        peak = dow.loc[dow[metric].idxmax()]
        peak_val = float(peak[metric])
        render_kpi_card(
            "Q1 | DÍA DE SEMANA",
            _fmt_currency(peak_val, 0) if metric == "revenue" else _fmt_number(peak_val, 0),
            f"Líder: <b>{peak['day_name']}</b>",
            "El mejor día concentra el mayor nivel de actividad para el periodo filtrado.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
    st.plotly_chart(
        style_fig(
            px.bar(
                dow,
                x="day_name",
                y=metric,
                color_discrete_sequence=[COLORS["accent_blue"]],
                text=text_values,
                labels={"day_name": "Día", metric: _axis_label_for_metric(metric)},
            ),
            f"{metric_labels[metric]} por día de semana",
            "",
        ),
        use_container_width=True,
        key="q1_dow",
    )

    hourly = d.groupby("hour", observed=False)[metric].agg(aggfunc).reset_index()
    if hourly.empty:
        render_kpi_card(
            "Q1 | HORA DEL DÍA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
        )
    else:
        peak = hourly.loc[hourly[metric].idxmax()]
        peak_hour = int(peak["hour"]) if pd.notna(peak["hour"]) else None
        peak_val = float(peak[metric])
        render_kpi_card(
            "Q1 | HORA DEL DÍA",
            _fmt_currency(peak_val, 0) if metric == "revenue" else _fmt_number(peak_val, 0),
            f"Líder: <b>{peak_hour}:00</b>" if peak_hour is not None else "Líder: <b>--</b>",
            "La hora pico define el punto de mayor demanda.",
        )
    hour_sub = "" if hourly.empty else (
        f"Hora pico: {peak_hour}:00 | {metric_labels[metric]}: {_fmt_currency(peak_val, 0) if metric == 'revenue' else _fmt_number(peak_val, 0)}" if peak_hour is not None else ""
    )
    st.plotly_chart(
        style_fig(
            px.line(
                hourly,
                x="hour",
                y=metric,
                markers=True,
                color_discrete_sequence=[COLORS["coffee_brown"]],
                labels={"hour": "Hora", metric: _axis_label_for_metric(metric)},
            ),
            f"{metric_labels[metric]} por hora",
            hour_sub,
        ),
        use_container_width=True,
        key="q1_hourly",
    )

    # Heatmap KPI (max cell)
    try:
        pivot = d.pivot_table(index="day_name", columns="hour", values=(metric if metric != "transaction_id" else "transaction_qty"), aggfunc="sum")
        if pivot.empty or pivot.isna().all().all():
            raise ValueError("empty")
        max_idx = pivot.stack().idxmax()
        max_val = float(pivot.loc[max_idx[0], max_idx[1]])
        render_kpi_card(
            "Q1 | PUNTO CALIENTE (DÍA × HORA)",
            _fmt_currency(max_val, 0) if metric == "revenue" else _fmt_number(max_val, 0),
            f"Líder: <b>{max_idx[0]} @ {int(max_idx[1])}:00</b>",
            "Este punto concentra el máximo volumen del periodo filtrado.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    except Exception:
        render_kpi_card(
            "Q1 | PUNTO CALIENTE (DÍA × HORA)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    st.plotly_chart(create_heatmap_day_hour(d, metric if metric != "transaction_id" else "transaction_qty"), use_container_width=True, key="q1_heatmap")

    st.info("Insights esperados: junio como pico mensual, 10 AM como hora pico, lunes con mayor ingreso y dominancia de mañana (48.27% de transacciones).")


def render_q2(df: pd.DataFrame):
    st.subheader("Q2 — Desempeño de Productos y Categorías")
    c1, c2, c3 = st.columns(3)
    categories = sorted(df["product_category"].unique())
    sel_categories = c1.multiselect("Categorías", categories, default=categories, key="q2_categories")
    top_n = c2.slider("Top N", 5, 20, 10)
    metric_labels = {"revenue": "Ingresos", "transaction_qty": "Unidades", "transaction_id": "Transacciones"}
    metric = c3.selectbox("Métrica", ["revenue", "transaction_qty", "transaction_id"], format_func=lambda x: metric_labels[x])

    d = df[df["product_category"].isin(sel_categories)]
    if d.empty:
        render_kpi_card(
            "Q2 | CATEGORÍAS",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para las categorías seleccionadas.",
        )
        st.stop()
    aggfunc = "sum" if metric != "transaction_id" else "nunique"

    cat = d.groupby("product_category", observed=False)[metric].agg(aggfunc).reset_index().sort_values(metric, ascending=False)
    top = d.groupby("product_type", observed=False)[metric].agg(aggfunc).reset_index().sort_values(metric, ascending=False).head(top_n)
    bottom = d.groupby("product_type", observed=False)[metric].agg(aggfunc).reset_index().sort_values(metric, ascending=True).head(top_n)

    # Format text based on metric type
    if metric == "revenue":
        text_values = cat[metric].apply(lambda x: f"${x:,.0f}")
    else:
        text_values = cat[metric].apply(lambda x: f"{x:,.0f}")
    
    if cat.empty:
        render_kpi_card(
            "Q2 | CATEGORÍAS",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para las categorías seleccionadas.",
        )
    else:
        peak = cat.iloc[0]
        peak_val = float(peak[metric])
        render_kpi_card(
            "Q2 | CATEGORÍAS",
            _fmt_currency(peak_val, 0) if metric == "revenue" else _fmt_number(peak_val, 0),
            f"Líder: <b>{peak['product_category']}</b>",
            "La categoría líder concentra la mayor parte del desempeño en el periodo filtrado.",
        )
    cat_sub = "" if cat.empty else (
        f"Líder: {peak['product_category']} | {metric_labels[metric]}: {_fmt_currency(peak_val, 0) if metric == 'revenue' else _fmt_number(peak_val, 0)}"
    )
    st.plotly_chart(
        style_fig(
            px.bar(
                cat,
                x="product_category",
                y=metric,
                color="product_category",
                color_discrete_map=CATEGORY_COLORS,
                text=text_values,
                labels={"product_category": "Categoría", metric: _axis_label_for_metric(metric)},
            ),
            f"{metric_labels[metric]} por categoría",
            cat_sub,
        ),
        use_container_width=True,
        key="q2_cat",
    )
    # Format text based on metric type
    if metric == "revenue":
        text_values = top[metric].apply(lambda x: f"${x:,.0f}")
    else:
        text_values = top[metric].apply(lambda x: f"{x:,.0f}")

    # Horizontal bar labels: Product name + bold+italic sales inside bar.
    top_labels = [f"{p} <b><i>{v}</i></b>" for p, v in zip(top["product_type"].tolist(), text_values.tolist())]
    
    if top.empty:
        render_kpi_card(
            "Q2 | TOP PRODUCTOS",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
    else:
        leader = top.iloc[0]
        leader_val = float(leader[metric])
        render_kpi_card(
            "Q2 | TOP PRODUCTOS",
            _fmt_currency(leader_val, 0) if metric == "revenue" else _fmt_number(leader_val, 0),
            f"Líder: <b>{leader['product_type']}</b>",
            f"Top {top_n}: enfoque para inventario y promoción.",
        )
    top_sub = "" if top.empty else (
        f"Líder: {leader['product_type']} | {metric_labels[metric]}: {_fmt_currency(leader_val, 0) if metric == 'revenue' else _fmt_number(leader_val, 0)}"
    )
    fig_top = px.bar(
        top,
        x=metric,
        y="product_type",
        orientation="h",
        color=metric,
        color_continuous_scale="YlOrBr",
        text=top_labels,
        labels={metric: _axis_label_for_metric(metric), "product_type": "Producto"},
    )
    fig_top.update_layout(coloraxis_showscale=False)
    fig_top.update_traces(textfont=dict(color=_contrast_text_colors(top[metric])), selector=dict(type="bar"))
    st.plotly_chart(
        style_fig(
            fig_top,
            f"Top {top_n} productos ({metric_labels[metric]})",
            top_sub,
            x_title=_axis_label_for_metric(metric),
            y_title="Producto",
        ),
        use_container_width=True,
        key=f"q2_top_{top_n}",
    )
    # Format text based on metric type
    if metric == "revenue":
        text_values = bottom[metric].apply(lambda x: f"${x:,.0f}")
    else:
        text_values = bottom[metric].apply(lambda x: f"{x:,.0f}")

    bottom_labels = [f"{p} <b><i>{v}</i></b>" for p, v in zip(bottom["product_type"].tolist(), text_values.tolist())]
    
    if bottom.empty:
        render_kpi_card(
            "Q2 | BOTTOM PRODUCTOS",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["warning_orange"],
            color_b=COLORS["roast_brown"],
        )
    else:
        lag = bottom.iloc[0]
        lag_val = float(lag[metric])
        render_kpi_card(
            "Q2 | BOTTOM PRODUCTOS",
            _fmt_currency(lag_val, 0) if metric == "revenue" else _fmt_number(lag_val, 0),
            f"Líder: <b>{lag['product_type']}</b>",
            f"Prioridad para revisión: productos con menor desempeño.",
            color_a=COLORS["warning_orange"],
            color_b=COLORS["roast_brown"],
        )
    bottom_sub = "" if bottom.empty else (
        f"Menor: {lag['product_type']} | {metric_labels[metric]}: {_fmt_currency(lag_val, 0) if metric == 'revenue' else _fmt_number(lag_val, 0)}"
    )
    fig_bottom = px.bar(
        bottom,
        x=metric,
        y="product_type",
        orientation="h",
        color=metric,
        color_continuous_scale="YlOrBr",
        text=bottom_labels,
        labels={metric: _axis_label_for_metric(metric), "product_type": "Producto"},
    )
    fig_bottom.update_layout(coloraxis_showscale=False)
    fig_bottom.update_traces(textfont=dict(color=_contrast_text_colors(bottom[metric])), selector=dict(type="bar"))
    st.plotly_chart(
        style_fig(
            fig_bottom,
            f"Bottom {top_n} productos ({metric_labels[metric]})",
            bottom_sub,
            x_title=_axis_label_for_metric(metric),
            y_title="Producto",
        ),
        use_container_width=True,
        key=f"q2_bottom_{top_n}",
    )

    # Donut KPI
    cat_rev = d.groupby("product_category", observed=False)["revenue"].sum().sort_values(ascending=False)
    if cat_rev.empty:
        render_kpi_card(
            "Q2 | MIX DE CATEGORÍAS",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    else:
        top_cat = cat_rev.index[0]
        top_val = float(cat_rev.iloc[0])
        share = (top_val / float(cat_rev.sum()) * 100.0) if float(cat_rev.sum()) > 0 else np.nan
        render_kpi_card(
            "Q2 | MIX DE CATEGORÍAS",
            _fmt_currency(float(cat_rev.sum()), 0),
            f"Líder: <b>{top_cat}</b>",
            f"Participación: {share:.1f}% ({_fmt_currency(top_val, 0)}).",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
    n_cats = int(d["product_category"].nunique())
    max_topn = max(3, min(12, n_cats))
    topn_cat = st.slider(
        "Top N categorías (donut)",
        min_value=3,
        max_value=max_topn,
        value=min(7, max_topn),
        step=1,
        key="q2_donut_topn",
    )
    st.plotly_chart(chart_category_donut_topn(d, top_n=topn_cat), use_container_width=True, key="q2_category_donut")

    # Pareto KPI
    prod_rev = d.groupby("product_type", observed=False)["revenue"].sum().sort_values(ascending=False)
    if prod_rev.empty:
        render_kpi_card(
            "Q2 | PARETO",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
    else:
        top10 = float(prod_rev.head(10).sum())
        total = float(prod_rev.sum())
        pct = (top10 / total * 100.0) if total > 0 else np.nan
        render_kpi_card(
            "Q2 | PARETO",
            f"{pct:.1f}%",
            f"Líder: <b>{prod_rev.index[0]}</b>",
            f"Top 10 productos concentran {pct:.1f}% del ingreso ({_fmt_currency(top10, 0)}).",
        )
    st.plotly_chart(create_pareto(d, top_n=25), use_container_width=True, key="q2_pareto")

    st.info("Insights clave: Coffee (38.6%) y Tea (28.1%) dominan; Barista Espresso lidera; ~10 productos explican ~80% del ingreso.")


def render_q3(df: pd.DataFrame):
    st.subheader("Q3 — Comparación entre Tiendas")
    c1, c2 = st.columns(2)
    stores = sorted(df["store_location"].unique())
    sel_store = c1.multiselect("Tiendas a comparar", stores, default=stores, key="q3_compare_stores")
    categories = sorted(df["product_category"].unique())
    sel_cat = c2.multiselect("Categorías", categories, default=categories, key="q3_categories")

    d = df[df["store_location"].isin(sel_store) & df["product_category"].isin(sel_cat)]
    if d.empty:
        render_kpi_card(
            "Q3 | COMPARACIÓN ENTRE TIENDAS",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
        )
        st.stop()

    # NOTE: the Top N slider for the Mix chart will be displayed directly above the Mix chart
    # to keep the filter local to that visualization.

    rev_store = d.groupby("store_location", observed=False)["revenue"].sum().reset_index()
    txn_store = d.groupby("store_location", observed=False)["transaction_id"].nunique().reset_index(name="transacciones")
    # Older pandas versions may not support reset_index(name=...). Use to_frame then reset_index for compatibility.
    ticket = d.groupby("store_location", observed=False).apply(lambda x: x["revenue"].sum() / x["transaction_id"].nunique())
    ticket = ticket.to_frame("ticket_promedio").reset_index()

    # Format revenue values for display
    text_values = rev_store["revenue"].apply(lambda x: f"${x:,.0f}")
    if rev_store.empty:
        render_kpi_card(
            "Q3 | INGRESOS POR TIENDA",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
    else:
        leader = rev_store.loc[rev_store["revenue"].idxmax()]
        spread = float(rev_store["revenue"].max() - rev_store["revenue"].min()) if len(rev_store) > 1 else 0.0
        render_kpi_card(
            "Q3 | INGRESOS POR TIENDA",
            _fmt_currency(float(leader["revenue"]), 0),
            f"Líder: <b>{leader['store_location']}</b>",
            f"Diferencia entre tiendas (max-min): {_fmt_currency(spread, 0)}.",
        )
    st.plotly_chart(
        style_fig(
            px.bar(
                rev_store,
                x="store_location",
                y="revenue",
                color="store_location",
                color_discrete_sequence=[COLORS["coffee_brown"], COLORS["roast_brown"], COLORS["accent_blue"]],
                text=text_values,
                labels={"store_location": "Tienda", "revenue": _axis_label_for_metric("revenue")},
            ),
            "Ingresos por tienda",
            "",
            x_title="Tienda",
            y_title=_axis_label_for_metric("revenue"),
        ),
        use_container_width=True,
        key="q3_rev_store",
    )
    # Format transaction values for display
    text_values = txn_store["transacciones"].apply(lambda x: f"{x:,.0f}")
    if txn_store.empty:
        render_kpi_card(
            "Q3 | TRANSACCIONES POR TIENDA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
    else:
        leader = txn_store.loc[txn_store["transacciones"].idxmax()]
        render_kpi_card(
            "Q3 | TRANSACCIONES POR TIENDA",
            _fmt_number(float(leader["transacciones"]), 0),
            f"Líder: <b>{leader['store_location']}</b>",
            "Volumen transaccional para dimensionar operación.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
    st.plotly_chart(
        style_fig(
            px.bar(
                txn_store,
                x="store_location",
                y="transacciones",
                color="store_location",
                text=text_values,
                labels={"store_location": "Tienda", "transacciones": _axis_label_for_metric("transacciones")},
            ),
            "Transacciones por tienda",
            "",
            x_title="Tienda",
            y_title=_axis_label_for_metric("transacciones"),
        ),
        use_container_width=True,
        key="q3_txn_store",
    )
    # Format ticket values for display (currency format)
    text_values = ticket["ticket_promedio"].apply(lambda x: f"${x:,.2f}")
    if ticket.empty:
        render_kpi_card(
            "Q3 | TICKET PROMEDIO",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["roast_brown"],
        )
    else:
        leader = ticket.loc[ticket["ticket_promedio"].idxmax()]
        render_kpi_card(
            "Q3 | TICKET PROMEDIO",
            _fmt_currency(float(leader["ticket_promedio"]), 2),
            f"Líder: <b>{leader['store_location']}</b>",
            "Eficiencia comercial: ingreso por transacción.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["roast_brown"],
        )
    st.plotly_chart(
        style_fig(
            px.bar(
                ticket,
                x="store_location",
                y="ticket_promedio",
                color="store_location",
                text=text_values,
                labels={"store_location": "Tienda", "ticket_promedio": _axis_label_for_metric("ticket_promedio")},
            ),
            "Ticket promedio por tienda",
            "",
            x_title="Tienda",
            y_title=_axis_label_for_metric("ticket_promedio"),
        ),
        use_container_width=True,
        key="q3_ticket",
    )

    rev_month_store = d.groupby(["month", "month_name", "store_location"], observed=False)["revenue"].sum().reset_index().sort_values("month")
    # Filter out months with zero revenue to avoid connecting unrelated months in the line chart
    rev_month_store = rev_month_store[rev_month_store["revenue"] > 0]

    # KPI for monthly evolution by store
    if rev_month_store.empty:
        render_kpi_card(
            "Q3 | EVOLUCIÓN MENSUAL (TIENDAS)",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["roast_brown"],
            color_b=COLORS["coffee_brown"],
        )
        rev_month_sub = ""
    else:
        last_month = rev_month_store["month"].max()
        last_slice = rev_month_store[rev_month_store["month"] == last_month].sort_values("revenue", ascending=False)
        leader = last_slice.iloc[0]
        # Growth from first available month for the leader store
        store_series = rev_month_store[rev_month_store["store_location"] == leader["store_location"]].sort_values("month")
        first_val = float(store_series.iloc[0]["revenue"]) if not store_series.empty else 0.0
        last_val = float(store_series.iloc[-1]["revenue"]) if not store_series.empty else float(leader["revenue"])
        growth_pct = ((last_val / first_val - 1) * 100.0) if first_val > 0 else np.nan
        render_kpi_card(
            "Q3 | EVOLUCIÓN MENSUAL (TIENDAS)",
            _fmt_currency(float(leader["revenue"]), 0),
            f"Líder (último mes): <b>{leader['store_location']}</b>",
            f"Último mes: {leader['month_name']} | Crecimiento del líder (inicio→fin): {growth_pct:.1f}%" if pd.notna(growth_pct) else f"Último mes: {leader['month_name']}.",
            color_a=COLORS["roast_brown"],
            color_b=COLORS["coffee_brown"],
        )
        rev_month_sub = f"Último mes ({leader['month_name']}): líder {leader['store_location']} con {_fmt_currency(float(leader['revenue']), 0)}"

    st.plotly_chart(
        style_fig(
            px.line(
                rev_month_store,
                x="month_name",
                y="revenue",
                color="store_location",
                markers=True,
                labels={"month_name": "Mes", "revenue": _axis_label_for_metric("revenue"), "store_location": "Tienda"},
            ),
            "Evolución mensual por tienda",
            rev_month_sub,
            x_title="Mes",
            y_title=_axis_label_for_metric("revenue"),
        ),
        use_container_width=True,
        key="q3_rev_month_store",
    )

    # Place the Top N slider directly above the mix chart (locality requirement)
    n_cats = int(d["product_category"].nunique())
    if n_cats <= 1:
        topn_cat = n_cats
    else:
        max_topn = min(12, n_cats)
        prev = st.session_state.get("q3_mix_topn", min(7, max_topn))
        default = int(min(prev, max_topn)) if prev is not None else min(7, max_topn)
        # Ensure session state value is within current bounds to avoid Streamlit errors
        st.session_state["q3_mix_topn"] = default
        topn_cat = st.slider(
            "Top N categorías (mix por tienda)",
            min_value=1,
            max_value=max_topn,
            value=st.session_state["q3_mix_topn"],
            step=1,
            key="q3_mix_topn",
        )
    mix_df = _group_top_n_categories(d, category_col="product_category", value_col="revenue", top_n=topn_cat)

    mix = mix_df.groupby(["store_location", "product_category"], observed=False)["revenue"].sum().reset_index()
    # For stacked bar chart, we'll show text on each segment
    # Create text values for each bar segment
    mix["text"] = mix["revenue"].apply(lambda x: f"${x:,.0f}" if x > 0 else "")
    if mix.empty:
        render_kpi_card(
            "Q3 | MIX (TIENDA × CATEGORÍA)",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
    else:
        mix_best = mix.sort_values("revenue", ascending=False).iloc[0]
        render_kpi_card(
            "Q3 | MIX (TIENDA × CATEGORÍA)",
            _fmt_currency(float(mix_best["revenue"]), 0),
            f"Líder: <b>{mix_best['store_location']} · {mix_best['product_category']}</b>",
            "El segmento líder identifica dónde se concentra el ingreso.",
        )
    st.plotly_chart(
        style_fig(
            px.bar(
                mix,
                x="store_location",
                y="revenue",
                color="product_category",
                barmode="stack",
                color_discrete_map=CATEGORY_COLORS,
                text="text",
                labels={"store_location": "Tienda", "revenue": _axis_label_for_metric("revenue"), "product_category": "Categoría"},
            ),
            "Mix de categoría por tienda",
            "",
            x_title="Tienda",
            y_title=_axis_label_for_metric("revenue"),
        ),
        use_container_width=True,
        key="q3_mix_by_store",
    )

    peak = d.groupby(["store_location", "hour"], observed=False)["revenue"].sum().reset_index()

    if peak.empty:
        render_kpi_card(
            "Q3 | HORA PICO (POR TIENDA)",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
        peak_sub = ""
    else:
        best_point = peak.loc[peak["revenue"].idxmax()]
        peak_sub = f"Pico global: {best_point['store_location']} @ {int(best_point['hour'])}:00 ({_fmt_currency(float(best_point['revenue']), 0)})"
        render_kpi_card(
            "Q3 | HORA PICO (POR TIENDA)",
            _fmt_currency(float(best_point["revenue"]), 0),
            f"Líder: <b>{best_point['store_location']} @ {int(best_point['hour'])}:00</b>",
            "Comparación de picos horarios para planificar staffing y producción.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )

    st.plotly_chart(
        style_fig(
            px.line(
                peak,
                x="hour",
                y="revenue",
                color="store_location",
                markers=True,
                labels={"hour": "Hora", "revenue": _axis_label_for_metric("revenue"), "store_location": "Tienda"},
            ),
            "Comparación de hora pico por tienda",
            peak_sub,
            x_title="Hora",
            y_title=_axis_label_for_metric("revenue"),
        ),
        use_container_width=True,
        key="q3_peak_by_hour",
    )

    st.info("Insights: Hell's Kitchen lidera en volumen; Lower Manhattan sobresale en eficiencia (ticket e items/tx); Astoria tiene espacio para crecer en ticket.")


def render_q4(df: pd.DataFrame):
    st.subheader("Q4 — Tamaño de Canasta y Comportamiento Transaccional")

    # Build a true transaction-level table.
    # IMPORTANT: do not group with observed=False over categorical columns (it creates unobserved cartesian
    # combinations and injects many zero rows, which breaks averages like items/order).
    txn = (
        df.groupby("transaction_id", observed=True)
        .agg(
            store_location=("store_location", "first"),
            time_of_day=("time_of_day", "first"),
            hour=("hour", "first"),
            items=("transaction_qty", "sum"),
            revenue_txn=("revenue", "sum"),
        )
        .reset_index()
    )

    c1, c2, c3 = st.columns(3)
    qmin, qmax = int(txn["items"].min()), int(txn["items"].max())
    qty_range = c1.slider("Rango de cantidad por transacción", qmin, qmax, (qmin, qmax))
    stores = sorted(txn["store_location"].unique())
    sel_store = c2.multiselect("Tiendas", stores, default=stores, key="q4_stores")
    times = [x for x in TIME_ORDER if x in txn["time_of_day"].unique()]
    sel_times = c3.multiselect("Franja horaria", times, default=times, key="q4_times")

    d = txn[(txn["items"] >= qty_range[0]) & (txn["items"] <= qty_range[1]) & txn["store_location"].isin(sel_store) & txn["time_of_day"].isin(sel_times)]

    # If no local filters selected or the resulting transaction-level dataset is empty,
    # render a default KPI card and stop to avoid downstream errors.
    if d.empty:
        render_kpi_card(
            "Q4 | TAMAÑO DE CANASTA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
        st.stop()

    else:
        avg_items_val = float(d["items"].mean())
        p90 = float(d["items"].quantile(0.9))
        render_kpi_card(
            "Q4 | TAMAÑO DE CANASTA",
            f"{_fmt_number(avg_items_val, 1)} ítems",
            f"Líder: <b>P90 = {int(p90)}</b>",
            "Distribución del tamaño de compra por transacción.",
        )
    st.plotly_chart(
        style_fig(
            px.histogram(
                d,
                x="items",
                nbins=20,
                color_discrete_sequence=[COLORS["coffee_brown"]],
                labels={"items": _axis_label_for_metric("items")},
                text_auto=True
            ),
            "Distribución de cantidad por transacción",
            "",
            x_title=_axis_label_for_metric("items"),
            y_title="Frecuencia",
        ),
        use_container_width=True,
        key="q4_items_dist",
    )

    items_by_hour = d.groupby("hour", observed=False)["items"].mean().reset_index()
    if items_by_hour.empty:
        render_kpi_card(
            "Q4 | ÍTEMS POR HORA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["roast_brown"],
        )
        items_hour_sub = ""
    else:
        best = items_by_hour.loc[items_by_hour["items"].idxmax()]
        worst = items_by_hour.loc[items_by_hour["items"].idxmin()]
        render_kpi_card(
            "Q4 | ÍTEMS POR HORA",
            f"{_fmt_number(float(best['items']), 1)} ítems",
            f"Líder: <b>{int(best['hour'])}:00</b>",
            f"Rango: {int(worst['hour'])}:00 ({_fmt_number(float(worst['items']), 1)}) a {int(best['hour'])}:00 ({_fmt_number(float(best['items']), 1)}).",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["roast_brown"],
        )
        items_hour_sub = f"Pico de ítems: {int(best['hour'])}:00 ({_fmt_number(float(best['items']), 1)})"

    st.plotly_chart(
        style_fig(
            px.line(
                items_by_hour,
                x="hour",
                y="items",
                markers=True,
                color_discrete_sequence=[COLORS["accent_blue"]],
                labels={"hour": "Hora", "items": _axis_label_for_metric("items")},
            ),
            "Items promedio por transacción según hora",
            items_hour_sub,
            x_title="Hora",
            y_title=_axis_label_for_metric("items"),
        ),
        use_container_width=True,
        key="q4_items_by_hour",
    )
    # Calculate average items per store
    avg_items = d.groupby("store_location", observed=False)["items"].mean().reset_index()
    # Format values for display
    text_values = avg_items["items"].apply(lambda x: f"{x:.1f} items")
    if avg_items.empty:
        render_kpi_card(
            "Q4 | ÍTEMS PROMEDIO POR TIENDA",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
        )
    else:
        leader = avg_items.loc[avg_items["items"].idxmax()]
        value_text = f"{_fmt_number(float(leader['items']), 1)}"
        leader_text = f"Líder: <b>{leader['store_location']}</b>"
        render_kpi_card(
            "Q4 | ÍTEMS PROMEDIO POR TIENDA",
            value_text,
            leader_text,
            "Mayor canasta promedio sugiere mayor oportunidad de cross-sell.",
        )

        bar_graph_items_by_store = px.bar(avg_items, x="store_location", y="items", color="store_location", text=text_values,
                     labels={"store_location": "Tienda", "items": _axis_label_for_metric("items")}, )
        style_fig(
            bar_graph_items_by_store,
            "Items promedio por tienda",
            leader_text + ' con ' + value_text + ' items por transaccion',
            x_title="Tienda",
            y_title=_axis_label_for_metric("items"),
        )
        st.plotly_chart(
            bar_graph_items_by_store.update_layout(showlegend=False, xaxis={
                "tickangle": 0
            }),
            use_container_width=True,
            key="q4_items_by_store",
        )

    # KPI for revenue per transaction distribution
    med = d.groupby("store_location", observed=False)["revenue_txn"].median().sort_values(ascending=False)
    if med.empty:
        render_kpi_card(
            "Q4 | INGRESO POR TRANSACCIÓN",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )
        box_sub = ""
    else:
        leader_store = med.index[0]
        leader_med = float(med.iloc[0])
        box_sub = f"Mediana más alta: {leader_store} ({_fmt_currency(leader_med, 2)})"
        value_text  =_fmt_currency(leader_med, 2)
        leader_text = f"Líder: <b>{leader_store}</b>"
        render_kpi_card(
            "Q4 | INGRESO POR TRANSACCIÓN",
            value_text,
            leader_text,
            "La mediana resume el ticket típico y el boxplot revela dispersión/outliers.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["coffee_brown"],
        )

    boxplot_revenue_per_store = px.box(d, x="store_location", y="revenue_txn", color="store_location",
                 labels={"store_location": "Tienda", "revenue_txn": _axis_label_for_metric("revenue_txn")}, )

    style_fig(
            boxplot_revenue_per_store,
            "Distribución de ingreso por transacción (boxplot)",
            box_sub,
            x_title="Tienda",
            y_title=_axis_label_for_metric("revenue_txn"),
        )
    st.plotly_chart(
        boxplot_revenue_per_store.update_layout(
            showlegend=False, xaxis={"tickangle": 0}),
        use_container_width=True,
        key="q4_revenue_box",
    )

    threshold = d["revenue_txn"].quantile(0.95)
    hv = d[d["revenue_txn"] >= threshold].groupby("store_location", observed=False).size().reset_index(name="transacciones_high_value")
    # Format high value transaction counts for display
    text_values = hv["transacciones_high_value"].apply(lambda x: f"{x:,.0f}")
    if hv.empty:
        render_kpi_card(
            "Q4 | TRANSACCIONES ALTO VALOR",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["danger_red"],
        )
    else:
        leader = hv.loc[hv["transacciones_high_value"].idxmax()]
        value_text = f"{_fmt_number(float(leader['transacciones_high_value']), 0)} transacciones"
        leader_text = f"Líder: <b>{leader['store_location']}</b>"
        render_kpi_card(
            "Q4 | TRANSACCIONES ALTO VALOR",
            value_text,
            leader_text,
            "Concentra el segmento top 5% de tickets.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["danger_red"],
        )
        high_value_transaction_chart = px.bar(hv, x="store_location", y="transacciones_high_value", color="store_location", text=text_values,
                     labels={"store_location": "Tienda",
                             "transacciones_high_value": _axis_label_for_metric("transacciones_high_value")})

        style_fig(
                high_value_transaction_chart,
                "Segmento de transacciones alto valor (top 5%)",
                leader_text + ' con ' + value_text,
                x_title="Tienda",
                y_title=_axis_label_for_metric("transacciones_high_value")
            )
        st.plotly_chart(
            high_value_transaction_chart.update_layout(showlegend=False, xaxis={"tickangle": 0}),
            use_container_width=True,
            key="q4_high_value",
        )




def render_q5(df: pd.DataFrame):
    st.subheader("Q5 — Trayectorias de Crecimiento")
    c1, c2, c3 = st.columns(3)
    categories = sorted(df["product_category"].unique())
    stores = sorted(df["store_location"].unique())
    sel_cat = c1.multiselect("Categorías", categories, default=categories, key="q5_categories")
    sel_store = c2.multiselect("Tiendas", stores, default=stores, key="q5_stores")
    growth_metric = c3.selectbox("Métrica de crecimiento", ["percentage", "absolute", "indexed"], format_func=lambda x: {"percentage": "% crecimiento", "absolute": "Crecimiento absoluto", "indexed": "Índice (enero=100)"}[x])

    d = df[df["product_category"].isin(sel_cat) & df["store_location"].isin(sel_store)]
    if d.empty:
        render_kpi_card(
            "Q5 | CRECIMIENTO (GENERAL)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
            color_a=COLORS["success_green"],
            color_b=COLORS["coffee_brown"],
        )
        st.stop()

    # Top N slider for groupings used in heatmaps / mix
    n_cats = int(d["product_category"].nunique())
    # If there's 0 or 1 category, there's no point rendering a slider. If there are
    # 2+ categories, allow the user to choose Top N between 1 and min(12, n_cats).
    if n_cats <= 1:
        topn_cat = n_cats
    else:
        max_topn = min(12, n_cats)
        # Ensure we pick a sensible default that respects prior session state but
        # does not exceed the current max_topn.
        prev = st.session_state.get("q5_topn", min(7, max_topn))
        default = int(min(prev, max_topn)) if prev is not None else min(7, max_topn)
        # Ensure session state value is within current bounds to avoid Streamlit errors
        st.session_state["q5_topn"] = default if default >= 1 else 1
        # Slider requires min_value < max_value when rendered; use 1..max_topn
        topn_cat = st.slider(
            "Top N categorías (Q5)",
            min_value=1,
            max_value=max_topn,
            value=st.session_state["q5_topn"],
            step=1,
            key="q5_topn",
        )

    d_grouped = _group_top_n_categories(d, category_col="product_category", value_col="revenue", top_n=topn_cat)
    if d.empty:
        render_kpi_card(
            "Q5 | CRECIMIENTO (GENERAL)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros locales seleccionados.",
            color_a=COLORS["success_green"],
            color_b=COLORS["coffee_brown"],
        )
        st.stop()
    # Use grouped dataset (Top N + Other) for category time series
    cat_month = d_grouped.groupby(["month", "month_name", "product_category"], observed=False)["revenue"].sum().reset_index().sort_values("month")
    # Filter out months with zero revenue to avoid connecting unrelated months in the line chart
    cat_month = cat_month[cat_month["revenue"] > 0]

    cat_month["pct_growth"] = cat_month.groupby("product_category", observed=False)["revenue"].pct_change() * 100
    cat_month["abs_growth"] = cat_month.groupby("product_category", observed=False)["revenue"].diff()
    first_vals = cat_month.groupby("product_category", observed=False)["revenue"].transform("first")
    cat_month["indexed"] = np.where(first_vals > 0, cat_month["revenue"] / first_vals * 100, np.nan)

    y_col = {"percentage": "pct_growth", "absolute": "abs_growth", "indexed": "indexed"}[growth_metric]

    # KPI for category growth (selected metric)
    last_month = cat_month["month"].max() if not cat_month.empty else None
    last_slice = cat_month[cat_month["month"] == last_month] if last_month is not None else cat_month
    if cat_month.empty or last_slice.empty:
        render_kpi_card(
            "Q5 | CRECIMIENTO (CATEGORÍAS)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["success_green"],
            color_b=COLORS["coffee_brown"],
        )
        cat_sub = ""
    else:
        # Use last month value for the selected growth series
        series = last_slice[["product_category", y_col]].dropna().sort_values(y_col, ascending=False)
        if series.empty:
            leader_cat = last_slice.iloc[0]["product_category"]
            leader_val = np.nan
        else:
            leader_cat = series.iloc[0]["product_category"]
            leader_val = float(series.iloc[0][y_col])

        if growth_metric == "percentage":
            value_text = f"{leader_val:.1f}%" if pd.notna(leader_val) else "--"
        elif growth_metric == "absolute":
            value_text = _fmt_currency(leader_val, 0) if pd.notna(leader_val) else "--"
        else:
            value_text = f"{leader_val:.0f}" if pd.notna(leader_val) else "--"

        render_kpi_card(
            "Q5 | CRECIMIENTO (CATEGORÍAS)",
            value_text,
            f"Líder (último mes): <b>{leader_cat}</b>",
            f"Último mes: {last_slice.iloc[0]['month_name']} | Métrica: { {'percentage':'% crecimiento','absolute':'crecimiento absoluto','indexed':'índice'}[growth_metric] }.",
            color_a=COLORS["success_green"],
            color_b=COLORS["coffee_brown"],
        )
        cat_sub = f"Último mes: líder {leader_cat} ({value_text})"

    st.plotly_chart(
        style_fig(
            px.line(cat_month, x="month_name", y=y_col, color="product_category", markers=True, color_discrete_map=CATEGORY_COLORS),
            "Crecimiento mensual por categoría",
            cat_sub,
            x_title="Mes",
            y_title=_axis_label_for_metric(y_col),
        ),
        use_container_width=True,
        key="q5_cat_growth",
    )

    # KPI for indexed growth
    if cat_month.empty:
        render_kpi_card(
            "Q5 | ÍNDICE (ENERO=100)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["dark_espresso"],
            color_b=COLORS["roast_brown"],
        )
        idx_sub = ""
    else:
        idx_last = cat_month[cat_month["month"] == cat_month["month"].max()][["product_category", "indexed"]].dropna()
        if idx_last.empty:
            idx_sub = ""
        else:
            idx_leader = idx_last.sort_values("indexed", ascending=False).iloc[0]
            render_kpi_card(
                "Q5 | ÍNDICE (ENERO=100)",
                f"{float(idx_leader['indexed']):.0f}",
                f"Líder (último mes): <b>{idx_leader['product_category']}</b>",
                "Comparación relativa vs. enero para priorizar categorías con mayor tracción.",
                color_a=COLORS["dark_espresso"],
                color_b=COLORS["roast_brown"],
            )
            idx_sub = f"Líder último mes: {idx_leader['product_category']} ({float(idx_leader['indexed']):.0f})"

    st.plotly_chart(
        style_fig(
            px.line(cat_month, x="month_name", y="indexed", color="product_category", markers=True, color_discrete_map=CATEGORY_COLORS),
            "Crecimiento indexado (enero = 100)",
            idx_sub,
            x_title="Mes",
            y_title=_axis_label_for_metric("indexed"),
        ),
        use_container_width=True,
        key="q5_indexed",
    )

    store_growth = d.groupby(["month", "month_name", "store_location"], observed=False)["revenue"].sum().reset_index().sort_values("month")
    # Filter out months with zero revenue to avoid connecting unrelated months in the line chart
    store_growth = store_growth[store_growth["revenue"] > 0]

    if store_growth.empty:
        render_kpi_card(
            "Q5 | CRECIMIENTO (TIENDAS)",
            "$--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
        store_sub = ""
    else:
        # Use growth from first to last available month per store
        by_store = store_growth.sort_values("month").groupby("store_location", observed=False)["revenue"].agg(["first", "last"]).reset_index()
        by_store["growth_pct"] = np.where(by_store["first"] > 0, (by_store["last"] / by_store["first"] - 1) * 100.0, np.nan)
        leader = by_store.sort_values("growth_pct", ascending=False).iloc[0]
        render_kpi_card(
            "Q5 | CRECIMIENTO (TIENDAS)",
            f"{float(leader['growth_pct']):.1f}%" if pd.notna(leader["growth_pct"]) else "--",
            f"Líder (inicio→fin): <b>{leader['store_location']}</b>",
            f"Ingreso último mes: {_fmt_currency(float(leader['last']), 0)}.",
            color_a=COLORS["accent_blue"],
            color_b=COLORS["coffee_brown"],
        )
        store_sub = f"Mayor crecimiento: {leader['store_location']} ({float(leader['growth_pct']):.1f}%)" if pd.notna(leader["growth_pct"]) else ""

    st.plotly_chart(
        style_fig(
            px.line(store_growth, x="month_name", y="revenue", color="store_location", markers=True),
            "Crecimiento de ingresos por tienda",
            store_sub,
            x_title="Mes",
            y_title=_axis_label_for_metric("revenue"),
        ),
        use_container_width=True,
        key="q5_store_growth",
    )



    # Build heatmap pivot but only include month columns that actually have data.
    pivot = cat_month.pivot_table(index="product_category", columns="month_name", values="pct_growth", aggfunc="mean")
    # Determine which of the first-six months are present in the filtered data, preserving order.
    months_present = [m for m in MONTH_ORDER[:6] if m in cat_month["month_name"].dropna().unique()]
    if months_present:
        heat = pivot.reindex(columns=months_present)
    else:
        # No months present -> keep empty frame (handled downstream)
        heat = pivot
    # If Enero exists, fill its NaNs with 0 because pct_change for Enero is undefined
    # (there is no prior month in the dataset). This prevents Enero from appearing
    # as empty cells in the heatmap while preserving other months' NaNs.
    if "Enero" in heat.columns:
        try:
            heat["Enero"] = heat["Enero"].fillna(0.0)
        except Exception:
            # If anything unexpected happens, fall back to no-op.
            pass

    # Drop columns that are entirely empty (all NaN). Keep columns with at least one observed value.
    heat = heat.dropna(axis=1, how="all")

    try:
        max_cell = heat.stack().idxmax()
        max_val = float(heat.loc[max_cell[0], max_cell[1]])
        render_kpi_card(
            "Q5 | PUNTO CALIENTE (CRECIMIENTO %)",
            f"{max_val:.1f}%",
            f"Líder: <b>{max_cell[0]} · {max_cell[1]}</b>",
            "Identifica el salto de crecimiento mas marcado (promedio) para priorizar acciones.",
            color_a=COLORS["warning_orange"],
            color_b=COLORS["danger_red"],
        )
        heat_sub = f"Mayor crecimiento: {max_cell[0]} en {max_cell[1]} ({max_val:.1f}%)"
    except Exception:
        render_kpi_card(
            "Q5 | PUNTO CALIENTE (CRECIMIENTO %)",
            "--",
            "Líder: <b>--</b>",
            "No hay datos disponibles para los filtros seleccionados.",
            color_a=COLORS["warning_orange"],
            color_b=COLORS["danger_red"],
        )
        heat_sub = ""

    colorscale = [[0, COLORS["cream"]], [1, COLORS["coffee_brown"]]]
    fig_heat = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale=colorscale,
        labels={"x": "Mes", "y": "Categoría", "color": _axis_label_for_metric("pct_growth")},
    )
    # Force the x-axis to only show the months we explicitly included (and in that order).
    try:
        if months_present:
            fig_heat.update_xaxes(categoryorder="array", categoryarray=months_present)
    except Exception:
        pass
    z = np.asarray(heat.to_numpy(dtype=float))
    flat_colors = _heatmap_text_colors(z, colorscale)
    flat_text = [_fmt_cell_value("pct_growth", v) for v in z.flatten(order="C")]
    xs = list(heat.columns)
    ys = list(heat.index)
    xi, yi = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    fig_heat.add_trace(
        go.Scatter(
            x=[xs[i] for i in xi.flatten(order="C")],
            y=[ys[j] for j in yi.flatten(order="C")],
            mode="text",
            text=flat_text,
            textfont=dict(color=flat_colors, size=11),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    style_fig(
            fig_heat,
            "Heatmap: Crecimiento % Por Categoría y Mes",
            heat_sub,
            x_title="Mes",
            y_title="Categoría",
        )
    st.plotly_chart(fig_heat.update_layout(yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True)), use_container_width=True)

    st.info("Insights: crecimiento total ene-jun de 103.83%; identificar categorías y tiendas con mayor tracción para priorizar promoción.")


def render_recommendations():
    st.subheader("Recomendaciones Estratégicas")
    recs = [
        {
            "title": "1) Optimización de Staffing para Horas Pico",
            "evidence": "48.27% de transacciones en la mañana y hora pico a las 10 AM.",
            "action": "Reforzar personal en pico de 10 AM y en lunes.",
            "kpi": "Tiempo de servicio y transacciones por hora.",
            "impact": "Impacto esperado: 5-10% de incremento en conversión.",
        },
        {
            "title": "2) Programa de Upselling en Astoria",
            "evidence": "Astoria presenta el ticket promedio más bajo entre tiendas.",
            "action": "Implementar combos y suggestive selling en punto de caja.",
            "kpi": "Ticket promedio Astoria y % de transacciones multi-item.",
            "impact": "Impacto esperado: +$15K a +$25K semestrales.",
        },
        {
            "title": "3) Gestión de Inventario Crítico (Top 10)",
            "evidence": "Un conjunto reducido de productos concentra la mayoría del ingreso (Pareto).",
            "action": "Asegurar disponibilidad y reposición prioritaria para productos tractores.",
            "kpi": "Tasa de stock-out en top productos.",
            "impact": "Impacto esperado: prevenir 2-5% de pérdida de ingresos.",
        },
        {
            "title": "4) Estrategia de Cross-Selling",
            "evidence": "41.5% de transacciones ya incluyen múltiples ítems.",
            "action": "Entrenar baristas en add-on selling y bundles de alto margen.",
            "kpi": "% de transacciones multi-item.",
            "impact": "Impacto esperado: +$30K a +$50K semestrales.",
        },
    ]

    for r in recs:
        st.markdown(
            f"""
            <div class='rec-card'>
                <h4>{r['title']}</h4>
                <p><b>Evidencia:</b> {r['evidence']}</p>
                <p><b>Acción:</b> {r['action']}</p>
                <p><b>KPI:</b> {r['kpi']}</p>
                <p><b>Impacto:</b> {r['impact']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================
# 5. Main app
# =========================
def main():
    st.set_page_config(page_title="Maven Roasters Dashboard", page_icon="☕", layout="wide")
    inject_css()

    st.title("☕ Dashboard Ejecutivo — Maven Roasters")
    st.caption("Análisis comercial de enero a junio 2023")

    full_df = load_data(DATA_PATH)
    filtered_df = apply_global_filters(full_df)

    if filtered_df.empty:
        st.warning("⚠️ No hay datos con los filtros seleccionados. Ajusta el rango o categorías en la barra lateral.")
        st.stop()

    tabs = st.tabs([
        "Overview",
        "Q1 — Ventas en el Tiempo y Momentos del Día",
        "Q2 — Desempeño de Productos y Categorías",
        "Q3 — Comparación entre Tiendas",
        "Q4 — Tamaño de Canasta y Comportamiento Transaccional",
        "Q5 — Trayectorias de Crecimiento",
        "Raw Data",
        "Recomendaciones Estratégicas",
    ])

    with tabs[0]:
        render_overview(filtered_df, full_df)
    with tabs[1]:
        render_q1(filtered_df)
    with tabs[2]:
        render_q2(filtered_df)
    with tabs[3]:
        render_q3(filtered_df)
    with tabs[4]:
        render_q4(filtered_df)
    with tabs[5]:
        render_q5(filtered_df)
    with tabs[6]:
        # Raw Data tab - show transformed columns
        st.subheader("Raw Data — muestra de filas con columnas transformadas")
        display_cols = [
            "transaction_id",
            "transaction_date",
            "transaction_time",
            "transaction_qty",
            "store_location",
            "product_category",
            "product_type",
            "unit_price",
            "revenue",
            "month_name",
            "day_name",
            "hour",
            "time_of_day",
            "week",
        ]
        existing = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[existing].head(1000).reset_index(drop=True), use_container_width=True)

        #Datos de crecimiento por mes

        jan = filtered_df[filtered_df["month"] == 1].groupby("product_category", observed=False)["revenue"].sum()
        jun = filtered_df[filtered_df["month"] == 6].groupby("product_category", observed=False)["revenue"].sum()
        jan = jan.to_frame("enero").reset_index()
        jun = jun.to_frame("junio").reset_index()
        comp = jan.merge(jun, on="product_category", how="outer").fillna(0)
        comp["crecimiento_%"] = np.where(comp["enero"] > 0, (comp["junio"] / comp["enero"] - 1) * 100, np.nan)
        st.dataframe(comp.style.format({"enero": "${:,.2f}", "junio": "${:,.2f}", "crecimiento_%": "{:.2f}%"}),
                     use_container_width=True)
    with tabs[7]:
        render_recommendations()


if __name__ == "__main__":
    main()
