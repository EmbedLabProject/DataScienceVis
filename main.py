import streamlit as st

from PageRealTimeDashBoard import show as show_real_time_dashboard
from PageTimePredictor import show as show_time_predictor
from PageFeatureImportance import show as show_feature_importance

from Plot import *

# Init
# model = xgb.XGBClassifier()
# model.load_model('xgb_model.json')

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Time Predictor'
st.set_page_config(layout="wide")

# Sidebar with navigation buttons
with st.sidebar:
    st.title('Navigation Bar')
    if st.button('Time Predictor',  use_container_width=True, type="primary"):
        st.session_state.page = 'TimePredictor'
    if st.button('Feature Importance',  use_container_width=True, type="secondary"):
        st.session_state.page = 'FeatureImportance'
    if st.button('Real-time Dashboard',  use_container_width=True, type="secondary"):
        st.session_state.page = 'RealTimeDashBoard'

if st.session_state.page == 'TimePredictor':
    show_time_predictor()
elif st.session_state.page == 'FeatureImportance':
    show_feature_importance()
elif st.session_state.page == 'RealTimeDashBoard':
    show_real_time_dashboard()