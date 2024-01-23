import streamlit as st
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval
from streamlit_image_select import image_select
from streamlit_extras.add_vertical_space import add_vertical_space
import time
import base64
import io
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
  global UPLOADED_FILE_BYTES
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
        <div class="next-btn py-2 px-4 me-5">
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
        "https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/testing_images/ISIC_0000043.jpg",
        "https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/testing_images/ISIC_0000069.jpg",
        "https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/testing_images/ISIC_0000165.jpg",
    ],
    use_container_width=False,
  )

  st.markdown(f'''
    <div class="or-txt">
      OR
    </div>
  ''', unsafe_allow_html=True)

  UPLOADED_FILE = st.file_uploader("Upload an image", accept_multiple_files=False, type=['jpg'], label_visibility="collapsed")

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
    UPLOADED_FILE_BYTES = base64.b64encode(buffer.tobytes()).decode('utf-8')
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

  if SELECTED_IMAGE is None:
    streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now', name='_self')")
    return

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
    # print(UPLOADED_FILE.shape)

  if UPLOADED_FILE is None:
    streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now', name='_self')")
    return
  
  st.markdown(f'''
    <div class="d-flex flex-row justify-content-center" style="height: max-content">
      <div>
        <img id="preview-img" class="rounded-4" src="data:image/jpeg;base64,{UPLOADED_FILE_BYTES}">
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
  image_rgb = image_rgb.convert("RGB")

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
  prediction = np.array((prediction_array > THRESHOLD).astype(np.uint8))
  prediction *= 255
  prediction = Image.fromarray(prediction).convert('L')
  prediction = Image.merge('RGB', [prediction]*3) 
  # prediction = prediction.convert("P")
  PREDICTION = prediction
  print(f'prediction {PREDICTION.mode}')

  UPLOADED_FILE = image_rgb
  print(f'uploaded {UPLOADED_FILE.mode}')

  time.sleep(1)
  streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now&step=finished', name='_self')")

def results_view():
  global UPLOADED_FILE
  # global PREPROCESSED_IMAGE
  global PREDICTION

  if UPLOADED_FILE is None:
    streamlit_js_eval(js_expressions="parent.window.open('/?nav=try-now', name='_self')")
    return

  b = io.BytesIO()
  UPLOADED_FILE.save(b, 'png')
  uploaded_bytes = base64.b64encode(b.getvalue()).decode('utf-8')

  c = io.BytesIO()
  PREDICTION.save(c, 'png')
  prediction_bytes = base64.b64encode(c.getvalue()).decode('utf-8')

  st.markdown(f'''
    <div>
      <div class="h1">
        And here it is!
      </div>
      <div class="second-row my-4 d-flex flex-row justify-content-between align-items-center">
        <div class="caption">
          On the left we have your dermoscopy image, along with the binary mask produced by our trained deep learning model is on the right! Hover your mouse on the right image to see some cool effects.
        </div>
        <div class="next-btn reset-btn py-2 px-4 me-5">
          <a href="/?nav=try-now" class="link-a">
            Reset
            <img src="https://cdn-icons-png.flaticon.com/512/7613/7613923.png" width="30vw">
          </a>
        </div>
      </div>
      <div class="third-row d-flex flex-row justify-content-around">
        <div class="img-container w-75 d-flex flex-row justify-content-center mt-5">
          <div class="my-container">
            <img id="original-img" class="shadow-lg rounded-4" src="data:image/png;base64,{uploaded_bytes}">
          </div>
          <div class="my-container">
            <img id="top-left" src="https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/assets/images/top left.png">
            <img id="bottom-right" src="https://raw.githubusercontent.com/DnYAlv/segmentation_app/angela/frontend/assets/images/bottom right.png">
            <img id="predicted-img" class="rounded-4" src="data:image/png;base64,{prediction_bytes}">
            <img id="under-img" class="shadow-lg rounded-4" src="data:image/png;base64,{uploaded_bytes}">
          </div>
        </div>
      </div>
    </div>
  ''', unsafe_allow_html=True)

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