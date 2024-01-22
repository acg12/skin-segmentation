import streamlit as st
# import extra_streamlit_components as stxs
from views import home, run_model
import utils as utl

st.set_page_config(layout="wide", page_title='SkinCam: Skin Lesion Segmentation')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

def navigation():
    route = utl.get_current_route("nav")
    if route == "home":
        home.load_view()
    elif route == "try-now":
        run_model.load_view()
    else:
        home.load_view()
        
navigation()