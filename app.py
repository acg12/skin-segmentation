import streamlit as st
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

# Set Page
st.set_page_config(layout='wide')

model = residual_unet.Residual_UNet()
model.load_weights('resunet_preproc_best_weight')
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), 
                  loss=metrics.dice_coef_loss, metrics=['accuracy', metrics.dice_coef])
st.title('Skin Lesion Segmentation')

uploaded_file = st.file_uploader("Upload an image")
    
original_img, preprocessed_img, pred_img = st.columns([1,1,1])

if uploaded_file is not None:
  # Read Image
  bytes_data = uploaded_file.getvalue()
  image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_rgb = Image.fromarray(image_rgb)
  image_rgb = image_rgb.resize((SIZE, SIZE))

  # Preprocess Image
  image = cv2.resize(image, PREPROCESS_SIZE)
  dst = preprocess.remove_hair(image)
  gray = preprocess.shade_of_gray_cc(dst)
  gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
  gray_img = Image.fromarray(gray_img)
  gray_img = gray_img.resize((SIZE, SIZE))

  # Predict Image
  test_img = Image.fromarray(gray)
  test_img = test_img.resize((SIZE, SIZE))
  test_img_input = np.expand_dims(test_img, 0)
  prediction_array = model.predict(test_img_input)[0,:,:,0]
  prediction_array = np.round(prediction_array, 1)
  prediction = np.array((prediction_array > THRESHOLD).astype(np.uint8))

  fig1, ax1 = plt.subplots(figsize=(2,2))
  ax1.imshow(image_rgb)
  ax1.axis('off') 

  fig2, ax2 = plt.subplots(figsize=(2,2))
  ax2.imshow(gray_img)
  ax2.axis('off')

  fig3, ax3 = plt.subplots(figsize=(2,2))
  ax3.imshow(prediction, cmap='gray')
  ax3.axis('off')

  with original_img:
    st.subheader('Original Image')
    st.pyplot(fig1)
  
  with preprocessed_img:
    st.subheader('DullRazor + Shades of Gray')
    st.pyplot(fig2)

  with pred_img:
    st.subheader('Prediction Mask')
    st.pyplot(fig3)
