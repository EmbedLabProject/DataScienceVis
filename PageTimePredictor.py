import json

import folium
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

from Plot import *


@st.cache_data
def load_map_polygon() :
    with open("./CoordPolygon.json", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return geojson_data
@st.cache_data
def basic_card(title, detail) :
    with st.container():
        st.markdown(
            """
            <div style="border: 2px solid #A9A9A9; border-radius: 10px; padding: 20px; background-color: transparent; margin-bottom: 10px;">
                <h5 style="text-align: left; color: white;  font-size: 15px;">""" + title + """</h5>
                <p style="color: white; font-size: 40px; margin-top: 0; margin-bottom: 0px;">""" + detail + """</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
@st.cache_data
def contribution_card(title, confidence, contributions):
    field_titles = ["Problem Type", "Date", "Time", "Location"]
    
    contributions = [min(max(int(c), 0), 100) for c in contributions]

    with st.container():
        bar_html = ""
        bar_html += f"""
                    <div style="border: 2px solid #A9A9A9; border-radius: 10px; padding: 20px; background-color: transparent; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="color: white; margin: 0;">{title}</h3>
                            <h3 style="color: white; margin: 0;">{confidence:.0f}%</h3>
                        </div>
                    """

        max_index = contributions.index(max(contributions))
        for i, field in enumerate(field_titles):
            percent = min(max(int(contributions[i]), 0), 100)
            bar_color = 'darkorange' if i == max_index else '#271E5C'
            bar_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="flex: 1; color: white; font-size: 14px;">{field}</div>
                <div style="flex: 4; background-color: #444; border-radius: 5px; height: 10px;">
                    <div style="width: {percent}%; background-color: {bar_color}; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """
        bar_html+="""</div>"""

        st.markdown(bar_html, unsafe_allow_html=True)


def map_input(unknownCoord) :
    lat, lon = None, None
    if not unknownCoord:
        geojson_data = load_map_polygon()
        all_shapes = [shape(feature["geometry"]) for feature in geojson_data["features"]]
        from shapely.ops import unary_union
        bangkok_area = unary_union(all_shapes)

        m = folium.Map(location=[13.7563, 100.5018], zoom_start=10, tiles="CartoDB dark_matter")

        folium.GeoJson(
            geojson_data,
            name="Bangkok Districts",
            style_function=lambda feature: {
                "fillColor": "#FFA50040",
                "color": "#FFA500",
                "weight": 1,
                "fillOpacity": 0.3,
            },
            tooltip=folium.GeoJsonTooltip(fields=["dname", "dname_e"], aliases=["ชื่อเขต", "District"])
        ).add_to(m)

        m.add_child(folium.LatLngPopup())

        output = st_folium(m, height=400, use_container_width=True)

        if output and "last_clicked" in output and output["last_clicked"]:
            lat = output["last_clicked"]["lat"]
            lon = output["last_clicked"]["lng"]
            point = Point(lon, lat)

            if not bangkok_area.contains(point):
                st.error("Selected location is outside of Bangkok.")
                validInput = False
    else :
        st.markdown(
            """
            <div style="
                height: 400px;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px dashed #FFA500;
                background-color: #111;
                color: #FFA500;
                font-size: 22px;
                font-weight: bold;
                border-radius: 8px;
            ">
                🗺️ Location selection disabled (Unknown Location checked)
            </div>
            """,
            unsafe_allow_html=True
        )
    return lat, lon

def problem_type_input() :
    selected_type = []
    selected_type = st.multiselect("Problem Type :", problem_type, default=[])
    return selected_type

def date_input() :
    selected_date = None
    left_col, right_col = st.columns([3, 1])
    with right_col:
        st.markdown("<div style='padding-top: 1.8em; text-align: center;'>", unsafe_allow_html=True)
        unknownDate = st.checkbox("Unknown Date")
        st.markdown("</div>", unsafe_allow_html=True)
    with left_col:
        selected_date = st.date_input("Date", disabled=unknownDate)
    return selected_date

def time_input() :
    selected_time = None
    left_col, right_col = st.columns([3, 1])
    with right_col:
        st.markdown("<div style='padding-top: 1.8em; text-align: center;'>", unsafe_allow_html=True)
        unknownTime = st.checkbox("Unknown Time")
        st.markdown("</div>", unsafe_allow_html=True)
    with left_col:
        selected_time = st.time_input("Time", disabled=unknownTime)
    return selected_time



def show() :
    st.title("Time Predictor")
    lat, lon = None, None
    selected_type = []
    selected_date = None
    selected_time = None

    unknownCoord = st.checkbox("Unknown Location", value=False)
    col1, col2 = st.columns(2)
    with col1 :  
        lat, lon = map_input(unknownCoord)
    with col2 :
        selected_type = problem_type_input()
        selected_date = date_input()
        select_time = time_input()

    st.title("Result")
    col1, col2 = st.columns(2)
    with col1 :
        basic_card("Estimate Time : ","15 min")
        basic_card("Confidence : ","85 %")
    with col2 : 
        plot_probabilities_pie([0.1,0.3,0.4,0.1,0.1],["1 day","2-3 days","4-5 days","6-10 days","10+ days"],300,750)
    
    st.markdown("---")

    contribution_card("Pothole Prediction", 87.3, [80, 60, 90, 70])
    col1, col2 = st.columns(2)
    with col1 :   
        contribution_card("Pothole Prediction", 87.3, [80, 60, 90, 70])
        contribution_card("Pothole Prediction", 87.3, [80, 60, 90, 70])
    with col2 :
        contribution_card("Pothole Prediction", 87.3, [80, 60, 90, 70])
        contribution_card("Pothole Prediction", 87.3, [80, 60, 90, 70])