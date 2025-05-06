import json

import folium
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium
import pandas as pd
import plotly.colors as pc
import numpy as np
from Constant import *
from Plot import *
from ModelEngine import *

df = pd.DataFrame()
is_df_load = False
def get_df() :
    global is_df_load
    global df
    if(is_df_load) :
        return df
    df = pd.read_csv('DataVis.csv')

    df.rename(columns=lambda col: col[5:] if col.startswith("type_") else col, inplace=True)

    district_columns = [
        'คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
        'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
        'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ', 'บางขุนเทียน',
        'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด', 'บางรัก',
        'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ', 'ป้อมปราบศัตรูพ่าย',
        'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ', 'มีนบุรี', 'ยานนาวา',
        'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว', 'วังทองหลาง',
        'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร', 'สายไหม',
        'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง'
    ]

    weekday_columns = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

    # Combine both lists
    binary_columns = district_columns + weekday_columns

    # Convert True/False to 1/0
    df[binary_columns] = df[binary_columns].astype(int)

    df["distance"] = pd.DataFrame({
        'distance': np.random.randint(1, 10001, size=len(df))
    })
    is_df_load = True
    return df

@st.cache_data
def load_map_polygon() :
    with open("./CoordPolygon.json", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return geojson_data
def select_multiple_districts_map(unknownCoord):
    if "selected_districts" not in st.session_state:
        st.session_state["selected_districts"] = []

    selected_districts = st.session_state["selected_districts"]

    if unknownCoord:
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
                🏙️ District selection disabled (Unknown Location checked)
            </div>
            """,
            unsafe_allow_html=True
        )
        return selected_districts

    geojson_data = load_map_polygon()
    last_clicked = st.session_state.get("last_clicked")

    # Handle click from last session
    if last_clicked:
        point = Point(last_clicked["lng"], last_clicked["lat"])
        for feature in geojson_data["features"]:
            polygon = shape(feature["geometry"])
            district_name = feature["properties"]["dname"]
            if polygon.contains(point):
                if district_name in selected_districts:
                    selected_districts.remove(district_name)
                else:
                    selected_districts.append(district_name)
                break
        st.session_state["selected_districts"] = selected_districts
        st.session_state["last_clicked"] = None  # Clear click to avoid double toggle
        st.rerun()  # Now safe to use after upgrade

    # Draw map
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=10, tiles="CartoDB dark_matter")

    for feature in geojson_data["features"]:
        district_name = feature["properties"]["dname"]
        is_selected = district_name in selected_districts

        folium.GeoJson(
            data=feature,
            name=district_name,
            style_function=lambda f, selected=is_selected: {
                "fillColor": "#00FF0040" if selected else "#FFA50040",
                "color": "#00FF00" if selected else "#FFA500",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["dname", "dname_e"],
                aliases=["ชื่อเขต", "District"]
            )
        ).add_to(m)

    output = st_folium(m, height=500, use_container_width=True)

    # Save click for processing on next run
    if output and "last_clicked" in output and output["last_clicked"]:
        st.session_state["last_clicked"] = output["last_clicked"]
        st.rerun()

    if selected_districts:
        st.success("Selected Districts: " + ", ".join(f"**{d}**" for d in selected_districts))

    def remove_khet_prefix(text):
        if text.startswith("เขต"):
            return text[3:]
        return text

    return [remove_khet_prefix(d) for d in selected_districts]



def show_type_date(data_dict) :

    result_dict = {}

    for label, df in data_dict.items():
        result = pd.DataFrame({
            "mean": df["Q2"],
            "lower": df["Q1"],
            "upper": df["Q3"]
        }, index=df.index)
    
        result_dict[label] = result

    plot_time_series_mean_sd(
        data_dict=result_dict,
        title="Estimate solve time of each problem type throughout time",
        xlabel="Time",
        ylabel="Expected time to solve (hr)",
    )

def show_type_day(df) :
    quartiles = df.to_dict(orient='index')

    plot_custom_quartile_chart(
        quartiles,
        title="Estimate solve time on each day of a week",
        xlabel="Day of Week",
        ylabel="Estimated solve time (h)")

def show_type_distance(data_dict) :
    result_dict = {}

    for label, df in data_dict.items():
        result = pd.DataFrame({
            "mean": df["Q2"],
            "lower": df["Q1"],
            "upper": df["Q3"]
        }, index=df.index)

        result.index = result.index.map(lambda interval: (interval.left + interval.right) / 2)
        result.index.name = 'distance_avg'
    
        result_dict[label] = result
    
    plot_distance_series_mean_sd(
        data_dict=result_dict,
        title="Estimated solve time of each problem type by distance to office",
        xlabel="Distance (km)",
        ylabel="Expected solve time (hours)",
    )

def show_type_map(df) :
    min_val = df['Average Duration'].min()
    max_val = df['Average Duration'].max()
    df['normalized'] = (df['Average Duration'] - min_val) / (max_val - min_val)

    lower_color = np.array([23, 20, 47])
    upper_color = np.array([255, 95, 2])

    def interpolate_color(value, lower_color, upper_color):
        color = lower_color + value * (upper_color - lower_color)
        return tuple(int(x) for x in color)

    def rgb_to_hex(rgb_tuple):
        return '#%02x%02x%02x' % rgb_tuple

    color_map = {
        f"เขต{district}": rgb_to_hex(interpolate_color(norm_val, lower_color, upper_color))
        for district, norm_val in df['normalized'].items()
    }

    st.markdown("**Estimate solve of each the problem by district**")
    plot_colored_districts_pydeck(color_map)



def show_time_type(df) :
    quartiles = df.to_dict(orient='index')
    
    plot_custom_quartile_chart(quartiles,
        title="Estimate solve time of each problem type",
        xlabel="Problem type",
        ylabel="Expected time to solve (hr)")

def show_time_day(df) :
    quartiles = df.to_dict(orient='index')

    plot_custom_quartile_chart(
        quartiles,
        title="Estimate solve time on each day of a week",
        xlabel="Day of Week",
        ylabel="Estimate solve time (hr)")

def show_time_distance(df) :
    result = pd.DataFrame({
        "mean": df["Q2"],
        "lower": df["Q1"],
        "upper": df["Q3"]
    }, index=df.index)

    result.index = result.index.map(lambda interval: (interval.left + interval.right) / 2)
    result.index.name = 'distance_avg'

    plot_distance_series_mean_sd({"total":result},
        title="Estimate solve time of each given distance",
        xlabel="Distance from office",
        ylabel="Expected time to solve (hr)")

def show_time_map(df) :
    min_val = df['Mean Duration'].min()
    max_val = df['Mean Duration'].max()
    df['normalized'] = (df['Mean Duration'] - min_val) / (max_val - min_val)

    # Custom color scale from #17142F (lower) to #FF5F02 (upper)
    lower_color = np.array([23, 20, 47])  # #17142F (RGB)
    upper_color = np.array([255, 95, 2])  # #FF5F02 (RGB)

    def interpolate_color(value, lower_color, upper_color):
        # Interpolate between the lower and upper color based on the normalized value
        color = lower_color + value * (upper_color - lower_color)
        return tuple(int(x) for x in color)

    def rgb_to_hex(rgb_tuple):
        return '#%02x%02x%02x' % rgb_tuple

    # Generate color map with the new custom scale
    color_map = {
        f"เขต{district}": rgb_to_hex(interpolate_color(norm_val, lower_color, upper_color))
        for district, norm_val in df['normalized'].items()
    }

    st.markdown("**Estimate solve of each the problem by district**")
    plot_colored_districts_pydeck(color_map)



def show_map_type(df) :
    df = df.set_index('ProblemType')
    quartiles = df.to_dict(orient='index')
    
    plot_custom_quartile_chart(quartiles,
        title="Estimate solve time of each problem type throughout time",
        xlabel="Time",
        ylabel="Expected time to solve (hr)")

def show_map_day(df) :
    df = df.set_index('Weekday')
    quartiles = df.to_dict(orient='index')

    plot_custom_quartile_chart(
        quartiles,
        title="Estimate solve time on each day of a week",
        xlabel="",
        ylabel="expected time to solved")
    
def show_map_distance(df) :
    df = df.set_index('distance')
    result = pd.DataFrame({
        "mean": df["Q2"],
        "lower": df["Q1"],
        "upper": df["Q3"]
    }, index=df.index)

    plot_distance_series_mean_sd({"total":result})

def show_map_date(df) :
    df = df.set_index('timestamp')
    result = pd.DataFrame({
        "mean": df["Q2"],
        "lower": df["Q1"],
        "upper": df["Q3"]
    }, index=df.index)

    plot_time_series_mean_sd(
        data_dict={"total":result},
        title="Estimate solve time of each problem type throughout time",
        xlabel="Year",
        ylabel="Expected time to solve",
    )


def show_type() :
    selected_types = st.multiselect("Select types of problem you want to compare", problem_type)

    q_day, avg_district, q_time, q_distance = analyze_problem_types(get_df() ,selected_types)

    col1, col2 = st.columns(2)
    with col1 :
        show_type_date(q_time)
    with col2 :
        show_type_day(q_day)
    show_type_distance(q_distance)
    show_type_map(avg_district)

def show_time() :
    min_date = datetime(2022, 1, 1).date()
    max_date = datetime(2025, 12, 31).date()

    col1, col2 = st.columns(2)
    with col1 :
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2 :
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")

    q_day, q_type, mean_district, q_distance = analyze_duration_in_period(get_df(), pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    col1, col2 = st.columns(2)
    with col1 :
        show_time_type(q_type)
    with col2 :
        show_time_day(q_day)
    show_time_distance(q_distance)
    show_time_map(mean_district)
    
def show_coordinate() :
    unknownCoord = st.checkbox("Unknown Location", value=False)
    districts = select_multiple_districts_map(unknownCoord)
    if(unknownCoord or len(districts) == 0) :
        districts = district_all
    q_type, q_day, q_time, q_distance = analyze_by_districts(get_df(), districts)

    col1, col2 = st.columns(2)
    with col1 :
        show_map_type(q_type)
    with col2 :
        show_map_day(q_day)
    show_map_date(q_time)
    show_map_distance(q_distance)


def show() :
    st.title("Feature Importance")

    mode = st.selectbox("Choose an feature", ["Please Select", "Type", "Time", "Coordinate"])
    if(mode == "Please Select") :
        pass
    elif(mode == "Type") :
        show_type()
    elif(mode == "Time") :
        show_time()
    else :
        show_coordinate()

