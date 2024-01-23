import streamlit as st
from streamlit.components.v1 import html

from paths import NAVBAR_PATHS


def inject_custom_css():
    st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">', unsafe_allow_html=True)

    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_current_route(key):
    try:
        return st.query_params[key]
    except:
        return None


def navbar_component():
    # with open("assets/images/settings.png", "rb") as image_file:
    #     image_as_base64 = base64.b64encode(image_file.read())

    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        if key != 'Try Now':
            navbar_items += (f'<div><a class="navitem navitem-padding" href="/?nav={value}">{key}</a></div>')
            

    # settings_items = ''
    # for key, value in SETTINGS.items():
    #     settings_items += (
    #         f'<a href="/?nav={value}" class="settingsNav">{key}</a>')

    component = f'''
            <div class="d-flex flex-row my-navbar justify-content-between align-items-center px-5 py-4">
                <div class="logo">
                    SkinCam
                </div>
                <div class="d-flex flex-row align-items-center">
                    {navbar_items}
                    <div>
                        <a href="/?nav=try-now" class="navitem try-btn" style="color: white;">Try Now</a>
                    </div>
                </div>
            </div>
            '''
    st.markdown(component, unsafe_allow_html=True)
    js = '''
    <script>
        // navbar elements
        var navigationTabs = window.parent.document.getElementsByClassName("navitem");
        var cleanNavbar = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }
        
        for (var i = 0; i < navigationTabs.length; i++) {
            cleanNavbar(navigationTabs[i]);
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
    '''
    html(js)