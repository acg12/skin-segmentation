import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.components.v1 import html

def load_view():
    with open('assets/home.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # row1 = row(3, vertical_align="center")
    # row1.button("Try Now")
    # row1.button("Try Now2")
    # row1.button("Try Now3")

    add_vertical_space(15)
    st.markdown("""
        <div class="d-flex flex-column align-items-center justify-content-between my-container">
            <div class="text title">
                Perform Segmentation with SkinCam
            </div>
            <div class="text">
                <div class="caption">
                    Witness the power of Deep Learning through our app, which takes a dermoscopic image of a skin lesion and performs segmentation to produce a binary mask of the lesion in the image.
                </div>
            </div>
            <div class="try-btn-title">
                <a href="/?nav=try-now" class="try-btn-title-a">Try for free</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    html('''
        <script>
            var navigationTabs = window.parent.document.getElementsByClassName("try-btn-title-a");
            var cleanNavbar = function(navigation_element) {
                navigation_element.removeAttribute('target')
                console.log(navigation_element)
            }
            
            for (var i = 0; i < navigationTabs.length; i++) {
                cleanNavbar(navigationTabs[i]);
            }
        </script>
    ''')

    