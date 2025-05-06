import json

import folium
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium
import pandas as pd
from streamlit_extras.st_autorefresh import st_autorefresh

from Constant import *
from Plot import *

# Refresh the app every 5 minutes (300000 milliseconds)
st_autorefresh(interval=300000, key="data_reload")

# Always read the file fresh on each run
df = pd.read_csv("./realtime_scraping/out/traffy_reports_latest.csv")


def get_df() :
    global is_df_load
    global df
    if(is_df_load) :
        return df
    df = pd.read_csv('./realtime_scraping/out/traffy_reports_latest.csv')
    is_df_load = True
    return df






def render_case_card(image_url, address_string, status_string, time, description, tags, estimate_time, confidence_percent):
    tags_html = "".join([
        f"""<span style="background-color:#17142F; color:#DDDDDD; padding:3px 8px; border-radius:5px; margin-right:5px; font-size:12px;">{tag}</span>"""
        for tag in tags
    ])

    card_html = f"""
    <div style="border: 2px solid #A9A9A9; border-radius: 10px; padding: 16px; background-color: transparent; margin-bottom: 12px;">
        <div style="display: flex; flex-direction: row; gap: 20px;">
            <div style="flex: 1;">
                <img src="{image_url}" alt="case image" style="width: 100%; border-radius: 8px; object-fit: cover; max-height: 140px;">
            </div>
            <div style="flex: 2; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="margin-bottom: 6px;">{tags_html}</div>
                    <div style="color: #ccc; font-size: 14px; margin-bottom: 4px;"><b>Status:</b> {status_string}</div>
                    <div style="color: #ccc; font-size: 14px; margin-bottom: 4px;"><b>Time:</b> {time}</div>
                    <div style="color: #ccc; font-size: 14px;"><b>Address:</b> {address_string}</div>
                </div>
            </div>
        </div>
        <div style="margin-top: 12px; color: #DDDDDD; font-size: 14px; line-height: 1.4;">
            {description}
        </div>
        <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center;">
            <div style="color: #FFA500; font-size: 14px;"><b>Estimated Time:</b> {estimate_time}</div>
            <div style="color: #FFA500; font-size: 14px;"><b>Confidence:</b> {confidence_percent:.0f}%</div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def show() :
    st.title("Real Time Dashboard")

    col1, col2 = st.columns(2)
    with col1 :
        render_case_card(
            image_url=df.iloc[0]["image_url"],
            address_string=df.iloc[0]["address"],
            status_string=df.iloc[0]["status"],
            time=df.iloc[0]["report_time"],
            description=df.iloc[0]["description"],
            tags=df.iloc[0]["tags"],
            estimate_time=df.iloc[0]["est"],
            confidence_percent=df.iloc[0]["conf"]
        )
        render_case_card(
            image_url=df.iloc[1]["image_url"],
            address_string=df.iloc[1]["address"],
            status_string=df.iloc[1]["status"],
            time=df.iloc[1]["report_time"],
            description=df.iloc[1]["description"],
            tags=df.iloc[1]["tags"],
            estimate_time=df.iloc[1]["est"],
            confidence_percent=df.iloc[1]["conf"]
        )
        render_case_card(
            image_url=df.iloc[2]["image_url"],
            address_string=df.iloc[2]["address"],
            status_string=df.iloc[2]["status"],
            time=df.iloc[2]["report_time"],
            description=df.iloc[2]["description"],
            tags=df.iloc[2]["tags"],
            estimate_time=df.iloc[2]["est"],
            confidence_percent=df.iloc[2]["conf"]
        )
    with col2 :
        render_case_card(
            image_url=df.iloc[3]["image_url"],
            address_string=df.iloc[3]["address"],
            status_string=df.iloc[3]["status"],
            time=df.iloc[3]["report_time"],
            description=df.iloc[3]["description"],
            tags=df.iloc[3]["tags"],
            estimate_time=df.iloc[3]["est"],
            confidence_percent=df.iloc[3]["conf"]
        )
        render_case_card(
            image_url=df.iloc[4]["image_url"],
            address_string=df.iloc[4]["address"],
            status_string=df.iloc[4]["status"],
            time=df.iloc[4]["report_time"],
            description=df.iloc[4]["description"],
            tags=df.iloc[4]["tags"],
            estimate_time=df.iloc[4]["est"],
            confidence_percent=df.iloc[4]["conf"]
        )
        render_case_card(
            image_url=df.iloc[5]["image_url"],
            address_string=df.iloc[5]["address"],
            status_string=df.iloc[5]["status"],
            time=df.iloc[5]["report_time"],
            description=df.iloc[5]["description"],
            tags=df.iloc[5]["tags"],
            estimate_time=df.iloc[5]["est"],
            confidence_percent=df.iloc[5]["conf"]
        )