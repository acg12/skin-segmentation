import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from model import residual_unet
from preprocessing import preprocess
from evaluation import metrics

AXIS = 3
SIZE = 128
PREPROCESS_SIZE = (512, 486)
THRESHOLD = 0.5

PHASE = 1
UPLOADED_FILE = None

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
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE == 1 else ""}">
          1
        </div>
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE == 2 else ""}">
          2
        </div>
        <div class="d-flex flex-column justify-content-center circle {"active" if PHASE == 3 else ""}">
          3
        </div>
      </div>
    </div>
  ''', unsafe_allow_html=True)

def upload_view():
  st.markdown('''
    <div class="h1">
      Upload Image
    </div>
    <div class="second-row my-4">
      <div class="caption">
        To get started, simply pick one of the dermoscopic images we have prepared for you to play around with or upload a dermoscopic image of your own!
      </div>
      <div class="next-btn">
        <button>
            <img src="../assets/images/chevron.png" width="4rem">
        </button>
      </div>
    </div>
  ''', unsafe_allow_html=True)

def loading_view():
  pass

def results_view():
  pass

def load_view():
  model = load_model()

  progress_bar()

  if PHASE == 1:
     upload_view()
  elif PHASE == 2:
     loading_view()
  else:
     results_view()

  # st.title('Skin Lesion Segmentation')

  uploaded_file = st.file_uploader("Upload an image")
      
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