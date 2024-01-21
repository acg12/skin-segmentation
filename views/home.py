import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
from streamlit.components.v1 import html

def load_view():
    with open('assets/home.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # row1 = row(3, vertical_align="center")
    # row1.button("Try Now")
    # row1.button("Try Now2")
    # row1.button("Try Now3")

    st.markdown("""
        <div class="d-flex flex-column justify-content-around my-container mb-5">
            <div class="text title">
                Perform Segmentation with SkinCam
            </div>
            <div class="text">
                <div class="caption">
                    Witness the power of Deep Learning through our app, which takes a dermoscopic image of a skin lesion and performs segmentation to produce a binary mask of where the lesion is in the image.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.button("Try for free")
    