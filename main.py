from turtle import color
import pandas as pd 
import streamlit as st 
from  PIL import Image, ImageEnhance
import threshold as threshold
import numpy as np
import cv2
import extra_streamlit_components as stx

st.set_page_config(
    page_title="Filtering and Edge detection",
    page_icon="✅",
    layout="wide",
)
st.title("Filtering and Ege detection")
with open(r"style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
chosen_id = stx.tab_bar(data=[
stx.TabBarItemData(id="tab1", title="Filters",description=''),
stx.TabBarItemData(id="tab2", title="Histograms",description=''),stx.TabBarItemData(id='tab3',title='Hybrid',description='')])
sidebar = st.sidebar.container()

#images 

if chosen_id == "tab1":

    
       sidebar.selectbox('Add Noise',('Uniform Noise','Gaussian Noise','Salt & Pepper Noise'))
       col1 , col2 = sidebar.columns(2)
       snr_value = col1.slider('SNR ratio', 0, step=1, max_value=100, value=50, label_visibility='visible')
       sigma_value = col2.slider('Sigma', 0, step=1, max_value=255, value=128, label_visibility='visible')
       sidebar.selectbox('Apply Filter',('Average Filter','Gaussian Filter','Median Filter'))
       col3 , col4 = sidebar.columns(2)
       mask_slider =col3.select_slider('Mask Size',options=['3x3','5x5','7x7','9x9'],label_visibility='visible')
       sigma_slider = col4.slider('Sigma', 0, step=1, max_value=100, value=50, label_visibility='visible')
       sidebar.selectbox('Detect Edges',('Sobel','Roberts','Prewitt','Canny Edge'))
     #images
       if my_upload is not None:
        image = Image.open(my_upload)
        i_image, f_image = st.columns( [1, 1])
        with i_image:
            st.markdown('<p style="text-align: center;">Input Image</p>',unsafe_allow_html=True)
            st.image(image,width=350)  
        with f_image:
            st.markdown('<p style="text-align: center;">Filtered Image</p>',unsafe_allow_html=True)
        n_image, e_image = st.columns( [1, 1])
        with n_image:
            st.markdown('<p style="text-align: center;">Noisy Image</p>',unsafe_allow_html=True) 
        with e_image:
            st.markdown('<p style="text-align: center;">Edge Detection Image</p>',unsafe_allow_html=True)
    

elif chosen_id == "tab2":
    histogram= sidebar.selectbox('Histogram',('normalized image','equalized image'))
    thresholding= sidebar.selectbox('thresholding',('local thresholding','global thresholding'))
    sidebar.button('Transform to gray')
    if my_upload is not None:
        image = Image.open(my_upload)
        i_image, f_image = st.columns( [1, 1])
        with i_image:
            st.markdown('<p style="text-align: center;">Input Image</p>',unsafe_allow_html=True)
            st.image(image,width=350)  
        with f_image:
            st.markdown('<p style="text-align: center;">Output Image</p>',unsafe_allow_html=True)        
        chart1, chart2 = st.columns( [1, 1])
        with chart1:
            st.markdown('<p style="text-align: center;">Original Histogram</p>',unsafe_allow_html=True) 
            st.bar_chart()
        with chart2:
            st.markdown('<p style="text-align: center;">Global Histogram</p>',unsafe_allow_html=True) 
            st.bar_chart()
elif chosen_id=='tab3': 
       #images

    sidebar.button('Make Hybrid')
    if my_upload is not None:
      image = Image.open(my_upload)
      i_image, f_image = st.columns( [1, 1])
      with i_image:
          st.markdown('<p style="text-align: center;">Input1 Image</p>',unsafe_allow_html=True)
          st.image(image,width=350)  
      with f_image:
          st.markdown('<p style="text-align: center;">Input2 Image</p>',unsafe_allow_html=True)
      n_image, e_image = st.columns( [1, 1])
      with n_image:
          st.markdown('<p style="text-align: center;">Hybrid Image</p>',unsafe_allow_html=True)

else:
    sidebar.empty()