from turtle import color
import pandas as pd 
import streamlit as st 
from  PIL import Image, ImageEnhance
import functions as fn
import numpy as np
import cv2
import extra_streamlit_components as stx
from io import StringIO
import os
# files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)
st.set_page_config(
    page_title="Filtering and Edge detection",
    page_icon="✅",
    layout="wide",
)
st.title("Filtering and Ege detection")
css = """
.uploadedFiles {
    display: none;
}
"""
# or `visibility: hidden;`


with open(r"style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
chosen_id = stx.tab_bar(data=[
stx.TabBarItemData(id="tab1", title="Filters",description=''),
stx.TabBarItemData(id="tab2", title="Histograms",description=''),
stx.TabBarItemData(id='tab3',title='Hybrid',description='')])
sidebar = st.sidebar.container()




def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


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
  
            col1, col2 = st.columns(2)
            second = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
            High_pass_first = st.checkbox('high pass for the first image')

            if my_upload is not None:
                if second is not None:
                    path_1='img/'+ my_upload.name
                    path_2='img/'+second.name
                    sidebar.button('Make Hybrid')
                    flag_1 = 0
                    flag_2=1
                    if High_pass_first:
                        flag_1 = 1
                        flag_2=0


                
                    i_image, i_image_2, f_image,f_image_2 = st.columns([1,1,1,1])
                    with i_image_2:
                        st.image(fn.prepare(path_1)) 
                    with i_image:
                        st.markdown('<p style="text-align: center;">Input1 Image</p>',unsafe_allow_html=True)
                        updated_path_1 = fn.getfilter(path_1,flag_1)
                        st.image(updated_path_1 ) 
                    with f_image_2:
                        st.image(fn.prepare(path_2))         
                    with f_image:
                        st.markdown('<p style="text-align: center;">Input2 Image</p>',unsafe_allow_html=True)
                        updated_path_2 = fn.getfilter(path_2,flag_2)

                        st.image(updated_path_2)
                
                    n_image, e_image = st.columns( [1, 1])
                    with n_image:
                        st.markdown('<p style="text-align: center;">Hybrid Image</p>',unsafe_allow_html=True)
                        st.image(fn.hybrid_images(updated_path_1,updated_path_2))

          
else:
    sidebar.empty()