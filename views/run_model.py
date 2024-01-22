import streamlit as st
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval
from streamlit_image_select import image_select
from streamlit_extras.add_vertical_space import add_vertical_space
import time
import base64
import urllib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from model import residual_unet
from preprocessing import preprocess
from evaluation import metrics
import utils as utl

AXIS = 3
SIZE = 128
PREPROCESS_SIZE = (512, 486)
THRESHOLD = 0.5

SELECTED_IMAGE = None
UPLOADED_FILE = None
PREPROCESSED_IMAGE = None
PREDICTION = None
UPLOADED_FILE_BYTES = None
MODEL = None
PHASE = 1

tf.keras.backend.clear_session()

@st.cache_resource
def load_model():
  model = residual_unet.Residual_UNet()
  model.load_weights('resunet_preproc_best_weight').expect_partial()
  model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), 
                loss=metrics.dice_coef_loss, metrics=['accuracy', metrics.dice_coef])
  
  return model

def progress_bar():
  with open('assets/run_model.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

  st.markdown(f'''
    <div class="d-flex flex-row justify-content-center mb-4">
      <div class="w-25 d-flex flex-row justify-content-between progress-bar">
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE <= 3 else ""}">
          1
        </div>
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE == 2 or PHASE == 3 else ""}">
          2
        </div>
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE == 3 else ""}">
          3
        </div>
      </div>
    </div>
  ''', unsafe_allow_html=True)

def upload_view():
  global UPLOADED_FILE
  global SELECTED_IMAGE
   
  st.markdown(f'''
    <div>
      <div class="h1">
        Upload Image
      </div>
      <div class="second-row my-4 d-flex flex-row justify-content-between align-items-center">
        <div class="caption">
          To get started, simply pick one of the dermoscopic images we have prepared for you to play around with or upload a dermoscopic image of your own!
        </div>
        <div class="next-btn p-3 me-5">
          <a href="/?nav=try-now&step=loading" class="link-a">
            Next
            <img src="https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/assets/images/chevron.png" width="30vw">
          </a>
        </div>
      </div>
    </div>
  ''', unsafe_allow_html=True)

  SELECTED_IMAGE = image_select(
    label="",
    images=[
        "https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/testing_images/ISIC_0000000.jpg",
        "https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/testing_images/ISIC_0000001.jpg",
    ],
    use_container_width=False,
  )

  st.markdown(f'''
    <div class="or-txt">
      OR
    </div>
  ''', unsafe_allow_html=True)

  UPLOADED_FILE = st.file_uploader("Upload an image", accept_multiple_files=False, type=['png'], label_visibility="collapsed")

  _, col2, _ = st.columns([2, 1, 2])

  if UPLOADED_FILE is not None:
    bytes_data = UPLOADED_FILE.getvalue()
    buffer = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = Image.fromarray(image_rgb)

    resized = image_rgb.resize((SIZE, SIZE))
    fig1, ax1 = plt.subplots(figsize=(2,2))
    ax1.imshow(resized)
    ax1.axis('off')
    with col2:
      st.pyplot(fig1)

    UPLOADED_FILE = image
    print(f'Uploaded file size: {UPLOADED_FILE.shape}')

def loading_view():
  global UPLOADED_FILE
  global SELECTED_IMAGE
  global PREPROCESSED_IMAGE
  global PREDICTION
  global UPLOADED_FILE_BYTES
  global MODEL

  st.markdown(f'''
    <div class="h1">
      Performing segmentation...
    </div>
  ''', unsafe_allow_html=True)

  add_vertical_space(5)

  # Read the file
  if UPLOADED_FILE is None:
    req = urllib.request.urlopen(str(SELECTED_IMAGE))
    read = req.read()
    
    bytes = bytearray(read)

    arr = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    base64_data = base64.b64encode(read).decode('utf8')

    UPLOADED_FILE = img
    UPLOADED_FILE_BYTES = base64_data
    print(UPLOADED_FILE.shape)

  if UPLOADED_FILE is None:
    streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now', name='_self')")
    return
  
  st.markdown(f'''
    <div class="d-flex flex-row justify-content-center" style="height: max-content">
      <div>
        <img id="preview-img" class="rounded-4" src="data:image/png;base64,{UPLOADED_FILE_BYTES}">
        <div class="loader-container">
          <div class="loader"></div>
        </div>
      </div>
      <div class="cover rounded-4"><div>
    </div>
  ''', unsafe_allow_html=True)

  # Save Image
  image_rgb = cv2.cvtColor(UPLOADED_FILE, cv2.COLOR_BGR2RGB)
  image_rgb = Image.fromarray(image_rgb)
  image_rgb = image_rgb.resize((SIZE, SIZE))

  # Preprocess the image
  image = cv2.resize(UPLOADED_FILE, PREPROCESS_SIZE)
  dst = preprocess.remove_hair(image)
  gray = preprocess.shade_of_gray_cc(dst)
  gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
  gray_img = Image.fromarray(gray_img)
  gray_img = gray_img.resize((SIZE, SIZE))
  PREPROCESSED_IMAGE = gray_img

  # Predict Image
  test_img = Image.fromarray(gray)
  test_img = test_img.resize((SIZE, SIZE))
  test_img_input = np.expand_dims(test_img, 0) / 255
  prediction_array = MODEL.predict(test_img_input)[0,:,:,0]
  prediction_array = np.round(prediction_array, 1)
  PREDICTION = np.array((prediction_array > THRESHOLD).astype(np.uint8))

  UPLOADED_FILE = image_rgb

  # time.sleep(1)
  # streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now&step=finished', name='_self')")

def results_view():
  global UPLOADED_FILE
  global PREPROCESSED_IMAGE
  global PREDICTION

  if UPLOADED_FILE is None:
    streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now', name='_self')")
    return

  col1, col2, col3 = st.columns([1, 1, 1])

  fig1, ax1 = plt.subplots(figsize=(2,2))
  ax1.imshow(UPLOADED_FILE)
  ax1.axis('off') 

  fig2, ax2 = plt.subplots(figsize=(2,2))
  ax2.imshow(PREPROCESSED_IMAGE)
  ax2.axis('off')

  fig3, ax3 = plt.subplots(figsize=(2,2))
  ax3.imshow(PREDICTION, cmap='gray')
  ax3.axis('off')

  with col1:
    st.subheader('Original Image')
    st.pyplot(fig1)
  
  with col2:
    st.subheader('DullRazor + Shades of Gray')
    st.pyplot(fig2)

  with col3:
    st.subheader('Prediction Mask')
    st.pyplot(fig3)

def load_view():
  global PHASE
  global MODEL
  route = utl.get_current_route("step")
  MODEL = load_model()

  if route == None:
     PHASE = 1
     progress_bar()
     upload_view()
  elif route == "loading":
     PHASE = 2
     progress_bar()
     loading_view()
  else:
     PHASE = 3
     progress_bar()
     results_view()

  html('''
        <script>
            var navigationTabs = window.parent.document.getElementsByClassName("link-a");
            var cleanNavbar = function(navigation_element) {
                navigation_element.removeAttribute('target')
                console.log(navigation_element)
            }
            
            for (var i = 0; i < navigationTabs.length; i++) {
                cleanNavbar(navigationTabs[i]);
            }
        </script>
    ''')
      
  # original_img, preprocessed_img, pred_img = st.columns([1,1,1])

  # if uploaded_file is not None:
  #   # Read Image
  #   bytes_data = uploaded_file.getvalue()
  #   image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
  #   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #   image_rgb = Image.fromarray(image_rgb)
  #   image_rgb = image_rgb.resize((SIZE, SIZE))

  #   # Preprocess Image
  #   image = cv2.resize(image, PREPROCESS_SIZE)
  #   dst = preprocess.remove_hair(image)
  #   gray = preprocess.shade_of_gray_cc(dst)
  #   gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
  #   gray_img = Image.fromarray(gray_img)
  #   gray_img = gray_img.resize((SIZE, SIZE))

  #   # Predict Image
  #   test_img = Image.fromarray(gray)
  #   test_img = test_img.resize((SIZE, SIZE))
  #   test_img_input = np.expand_dims(test_img, 0) / 255
  #   prediction_array = model.predict(test_img_input)[0,:,:,0]
  #   prediction_array = np.round(prediction_array, 1)
  #   prediction = np.array((prediction_array > THRESHOLD).astype(np.uint8))

  #   # Ground truth
  #   # gt = cv2.imread("[isi nama file nya]", 0)
  #   # gt = Image.fromarray(gt)
  #   # gt = gt.resize((SIZE, SIZE))
  #   # gt = np.array(gt) / 255

  #   # print(metrics.dice_coef_eval(gt, prediction))

  #   fig1, ax1 = plt.subplots(figsize=(2,2))
  #   ax1.imshow(image_rgb)
  #   ax1.axis('off') 

  #   fig2, ax2 = plt.subplots(figsize=(2,2))
  #   ax2.imshow(gray_img)
  #   ax2.axis('off')

  #   fig3, ax3 = plt.subplots(figsize=(2,2))
  #   ax3.imshow(prediction, cmap='gray')
  #   ax3.axis('off')

  #   with original_img:
  #     st.subheader('Original Image')
  #     st.pyplot(fig1)
    
  #   with preprocessed_img:
  #     st.subheader('DullRazor + Shades of Gray')
  #     st.pyplot(fig2)

  #   with pred_img:
  #     st.subheader('Prediction Mask')
  #     st.pyplot(fig3)