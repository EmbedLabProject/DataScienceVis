import datetime
from datetime import datetime, timedelta

import folium
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost as xgb
from branca.colormap import linear
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from streamlit_folium import st_folium

import json
import plotly.graph_objects as go
import pandas as pd
import json
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import json
import pandas as pd
import streamlit as st


from Constant import *

@st.cache_data
def get_prob(model, input) :
    return model.predict_proba(input)

def get_shap_values(model, input) :
    pass


def plot_probabilities_bar(probs, labels,height,width, highlight_color='darkorange', default_color='#271E5C'):
    assert len(probs) == len(labels), "Length of probabilities and labels must match."

    probs = np.array(probs)
    max_index = np.argmax(probs)

    colors = [highlight_color if i == max_index else default_color for i in range(len(probs))]

    text = [f"{p*100:.1f}%" for p in probs]
    text_font_size = [18 if i == max_index else 10 for i in range(len(probs))]
    text_font_color = ['white' if i == max_index else 'gray' for i in range(len(probs))]

    fig = go.Figure()

    for i in range(len(probs)):
        fig.add_trace(go.Bar(
            x=[labels[i]],
            y=[probs[i]],
            name=labels[i],
            marker_color=colors[i],
            text=text[i],
            textposition='outside',
            textfont=dict(size=text_font_size[i], color=text_font_color[i]),
            hoverinfo='x+y'
        ))

    fig.update_layout(
        xaxis_title='Labels',
        yaxis_title='Probability',
        yaxis_range=[0, max(probs) + 0.1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white')),
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        width=width
    )

    st.plotly_chart(fig)

def plot_probabilities_pie(probs, labels, height, width, highlight_color='darkorange', default_color='#271E5C'):
    assert len(probs) == len(labels), "Length of probabilities and labels must match."

    probs = np.array(probs)
    max_index = np.argmax(probs)

    colors = [highlight_color if i == max_index else default_color for i in range(len(probs))]
    pulls = [0.10 if i == max_index else 0.01 for i in range(len(probs))]

    text_colors = ['white'] * len(probs)
    text_colors[max_index] = 'lightgray'

    text_sizes = [20] * len(probs)
    text_sizes[max_index] = 28

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=labels,
        values=probs,
        pull=pulls,
        marker=dict(colors=colors),
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+value+percent',
        textfont=dict(color='white', size=18),
    ))

    fig.update_traces(
        textfont=dict(color='white'),
        selector=dict(type='pie')
    )

    fig.update_layout(
        height=height,
        width=width,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=18),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig)

def plot_time_series_mean_sd(
    data_dict: dict,
    title="Time Series with Confidence Bounds",
    xlabel: str = "Time",
    ylabel: str = "Value",
    max_legend_items: int = 10,
    fallback_time_index: pd.DatetimeIndex = None
):
    """
    Plot multiple labeled time series with mean and confidence bounds using Plotly.
    If data_dict is empty, show an empty plot using fallback_time_index for x-axis.
    Legend is hidden unless there are 2 or more series.
    """
    fig = go.Figure()
    show_legend = len(data_dict) > 1

    if not data_dict:
        if fallback_time_index is not None:
            fig.add_trace(go.Scatter(
                x=fallback_time_index,
                y=[None] * len(fallback_time_index),
                mode='lines',
                name=None,
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ))
            start = fallback_time_index.min()
            end = fallback_time_index.max()
        else:
            st.warning("No data to show and no fallback time index provided.")
            return
    else:
        with open('ColorPallete.json', 'r', encoding='utf-8') as f:
            color_dict = json.load(f)

        all_dates = []

        for i, (label, df) in enumerate(data_dict.items()):
            color = color_dict.get(label, f"rgba(0,0,255,{1.0 - i*0.1})")
            fill_color = color_dict.get(label + "light", 'rgba(0,0,255,0.2)')
            all_dates.append(df.index)

            fig.add_traces([
                go.Scatter(
                    x=pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]),
                    y=pd.concat([df['upper'], df['lower'][::-1]]),
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                go.Scatter(
                    x=df.index,
                    y=df["mean"],
                    mode="lines",
                    name=label if show_legend and i < max_legend_items else None,
                    line=dict(color=color, width=2),
                    showlegend=show_legend and i < max_legend_items
                )
            ])

        all_dates_flat = pd.concat([pd.Series(d) for d in all_dates])
        start, end = all_dates_flat.min(), all_dates_flat.max()
        span_days = (end - start).days

        if span_days <= 60:
            freq = 'D'
        elif span_days <= 730:
            freq = 'MS'
        else:
            freq = 'YS'

        for tick in pd.date_range(start=start, end=end, freq=freq):
            fig.add_vline(x=tick, line=dict(color='gray', dash='dash', width=0.5), opacity=0.4)

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=show_legend,
        margin=dict(r=50),  # reduce right margin since legend is gone
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_distance_series_mean_sd(
    data_dict: dict,
    title="Time Series with Confidence Bounds",
    xlabel: str = "Time",
    ylabel: str = "Value",
    max_legend_items: int = 10,
    fallback_index: pd.Index = None
):
    """
    Plot multiple labeled time series with mean and confidence bounds using Plotly.
    Supports both datetime and numeric x-axes.
    Displays an empty graph if data_dict is empty.
    """
    fig = go.Figure()

    if not data_dict:
        if fallback_index is not None:
            fig.add_trace(go.Scatter(
                x=fallback_index,
                y=[None] * len(fallback_index),
                mode='lines',
                name="No data selected",
                line=dict(color='gray', dash='dot')
            ))
            x_min, x_max = fallback_index.min(), fallback_index.max()
        else:
            st.warning("No data to display and no fallback index provided.")
            return
    else:
        with open('ColorPallete.json', 'r', encoding='utf-8') as f:
            color_dict = json.load(f)

        is_datetime = pd.api.types.is_datetime64_any_dtype(next(iter(data_dict.values())).index)

        all_x = []

        for i, (label, df) in enumerate(data_dict.items()):
            x = df.index
            all_x.append(pd.Series(x))  # Ensures safe concatenation

            color = color_dict.get(label, f"rgba(0,0,255,{1.0 - i * 0.1})")
            fill_color = color_dict.get(label + "light", f"rgba(0,0,255,0.2)")

            show_legend = i < max_legend_items

            # Confidence bounds
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(x), pd.Series(x[::-1])]),
                y=pd.concat([df["upper"], df["lower"][::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=x,
                y=df["mean"],
                mode="lines",
                name=label if show_legend else None,
                line=dict(color=color, width=2),
                showlegend=show_legend
            ))

        # Vertical guide lines
        all_x_ = [pd.Series(df.index) for df in data_dict.values()]
        flat_x = pd.concat(all_x_)
        x_min, x_max = flat_x.min(), flat_x.max()

        if is_datetime:
            span_days = (x_max - x_min).days
            freq = 'D' if span_days <= 60 else 'MS' if span_days <= 730 else 'YS'
            ticks = pd.date_range(start=x_min, end=x_max, freq=freq)
        else:
            ticks = np.linspace(x_min, x_max, num=10)

        for tick in ticks:
            fig.add_vline(
                x=tick,
                line=dict(color='gray', dash='dash', width=0.5),
                opacity=0.4
            )

        if len(data_dict) > max_legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name=f"...and {len(data_dict) - max_legend_items} more",
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=True
            ))

    # Final layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=1.02,
            y=0.5,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)'
        ),
        margin=dict(r=200),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_custom_quartile_chart(
    quartiles: dict,
    title="Custom Quartile Chart",
    xlabel="Category",
    ylabel="Value",
):
    """
    Plot custom quartile chart with Plotly:
    - Vertical line from Q1 to Q4
    - Bar (box) from Q2 to Q3

    Parameters:
    - quartiles: dict[label] = {"Q1": ..., "Q2": ..., "Q3": ..., "Q4": ...}
    - title: str
    - xlabel: str
    - ylabel: str
    """
    with open('ColorPallete.json', 'r', encoding='utf-8') as f:
        color_dict = json.load(f)

    fig = go.Figure()

    labels = list(quartiles.keys())

    for i, label in enumerate(labels):
        q = quartiles[label]
        q1, q2, q3, q4 = q["Q1"], q["Q2"], q["Q3"], q["Q4"]
        color = color_dict.get(label, "blue")

        # Vertical line from Q1 to Q4
        fig.add_trace(go.Scatter(
            x=[label, label],
            y=[q1, q4],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False
        ))

        # Bar from Q2 to Q3
        fig.add_trace(go.Bar(
            x=[label],
            y=[q3 - q2],
            base=q2,
            marker=dict(color=color, line=dict(color="black", width=1)),
            opacity=0.6,
            name=label,
            showlegend=False
        ))

        # Add text label above Q3
        fig.add_trace(go.Scatter(
            x=[label],
            y=[q3 + 0.5],
            mode="text",
            text=[label],
            textposition="top center",
            textfont=dict(color="white", size=10),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_colored_districts_pydeck(color_map):
    """
    Plots Bangkok districts with specified colors using pydeck.

    Parameters:
    - geojson_path: path to the GeoJSON file
    - color_map: dict[district_name] = hex_color (e.g., "#FF0000")
    """
    # Load GeoJSON
    with open("./CoordPolygon.json", 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    # Map colors to each feature
    for feature in geojson['features']:
        district_name = feature['properties']['dname']
        hex_color = color_map.get(district_name, '#888888')  # fallback grey
        # Convert hex to RGB array
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        feature['properties']['fill_color'] = list(rgb) + [180]  # RGBA

    # Create pydeck layer
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        opacity=0.7,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        pickable=True,
    )

    # Create pydeck view
    view_state = pdk.ViewState(
        latitude=13.7563,
        longitude=100.5018,
        zoom=10.5,
        pitch=0
    )

    # Deck
    r = pdk.Deck(
        layers=[geojson_layer],
        initial_view_state=view_state,
        tooltip={"text": "{dname}"}
    )

    st.pydeck_chart(r)
