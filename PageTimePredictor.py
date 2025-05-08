import json

import folium
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

from Plot import *
from TimePredictor import predict_time
from GetLocation import get_address_from_latlong

# import model
import datetime
import xgboost as xgb

model = xgb.Booster()
model.load_model("model.json")
api_key = "dad93076eb9a905a9122a00806c70616"  # แทนที่ด้วย API key ของคุณ

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
                <p style="color: white; font-size: 55px; margin-top: 0; margin-bottom: 0px;">""" + detail + """</p>
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
    lat, lon = 13.7360, 100.5338
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
    selected_date = st.date_input("Date")
    return selected_date

def time_input() :
    selected_time = None
    selected_time = st.time_input("Time")
    return selected_time



def show() :
    st.title("Time Predictor")
    lat, lon = None, None # laatitude and longtitude, float
    selected_type = [] # problem types in thai, list of string
    selected_date = None # Date, streamlit Date
    selected_time = None # Time, streamlit Time

    col1, col2 = st.columns(2)
    with col1 :  
        lat, lon = map_input(False)
    with col2 :
        selected_type = problem_type_input()
        selected_date = date_input()
        selected_time = time_input()
    
    timestamp = datetime.datetime.combine(selected_date, selected_time).isoformat()
    coords = None if False else f"{lat},{lon}"
    # district, subdistrict = "บางซื่อ", "บางซื่อ"  # ต้องไปแก้
    district, subdistrict = get_address_from_latlong(lat, lon, api_key)
    preds, margins, est, conf, contrib_data = predict_time(
        model=model,
        problem_types=selected_type,
        timestamp=timestamp,
        coords=coords,
        district=district,
        subdistrict=subdistrict
    )
    
    labels = ["1 day", "1-3 days", "3-7 days", "7-21 days", "21-90 days", "90+ days"]
    estimated_label = labels[est]

    st.title("Result") # put result here
    col1, col2 = st.columns(2)
    with col1 :
        basic_card("Estimate Time :", f"{estimated_label}")
        basic_card("Confidence :", f"{conf:.0f} %")
    with col2 : 
        labels = ["1 day", "1-3 days", "3-7 days", "7-21 days", "21-90 days", "90+ days"]
        plot_probabilities_pie(preds, labels, 450, 750)
    
    st.markdown("---")

    # Put the most confidence card on top(the first one) and the other ordered non-ascending.
    # Put the (second parameter) the confidence rate of each class
    # Put the (third parameter) the contribution of each input to that class with range 1-100
    #   The feature are ordered as follows ["Problem Type", "Date", "Time", "Location"]

    col1, col2 = st.columns(2)
    for idx, item in enumerate(contrib_data):
        with (col1 if idx % 2 == 0 else col2):
            contribution_card(item["label"], item["confidence"], item["contributions"])

        