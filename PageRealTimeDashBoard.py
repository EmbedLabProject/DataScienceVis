import json

import folium
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

from Constant import *
from Plot import *


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
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )
        render_case_card(
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )
        render_case_card(
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )
    with col2 :
        render_case_card(
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )
        render_case_card(
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )
        render_case_card(
            image_url="https://storage.googleapis.com/traffy_public_bucket/attachment/2025-05/29eb35ff69e6029d599b9b4f2effead6.jpg",
            address_string="123 Sukhumvit Road, Bangkok",
            status_string="Resolved",
            time="2025-05-05 14:30",
            description="ถนนซอยเจริญกรุง 107 ยาวไปจนทะลุออกถนนเจริญราษฎร์ มีการเทพื้นถนนใหม่ แต่ยังไม่มีการตีเส้นพื้นถนน ประม...เพิ่มเติม",
            tags=["Water", "Urgent", "Public Utility"],
            estimate_time="1 hr 45 min",
            confidence_percent=93.2
        )