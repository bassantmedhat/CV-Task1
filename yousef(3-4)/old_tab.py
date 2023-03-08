
# if chosen_id == "tab1":

    
#        selected_noise = sidebar.selectbox('Add Noise',('Uniform Noise','Gaussian Noise','Salt & Pepper Noise'))
       
#        col1 , col2 = sidebar.columns(2)
#        snr_value = col1.slider('SNR ratio', 0, step=1, max_value=100, value=50, label_visibility='visible')
#        sigma_noise = col2.slider('Sigma', 0.0, step=0.01, max_value=1.0, value=0.128, label_visibility='visible')
#        selected_filter = sidebar.selectbox('Apply Filter',('Average Filter','Gaussian Filter','Median Filter'))
#        col3 , col4 = sidebar.columns(2)
#        mask_slider =col3.select_slider('Mask Size',options=['3x3','5x5','7x7','9x9'],label_visibility='visible')
#        sigma_slider = col4.slider('Sigma', 0, step=1, max_value=100, value=50, label_visibility='visible')
#        edge = sidebar.selectbox('Detect Edges',('Sobel','Roberts','Prewitt','Canny Edge'))
#        noisy_image = 0
#      #images
#        if my_upload is not None:
#         image = Image.open(my_upload)
#         pic = image.convert("L")
#         img = np.array(pic) 
#         # print(type(img))
#         i_image, n_image = st.columns( [1, 1])
#         with i_image:
#             st.markdown('<p style="text-align: center;">Input Image</p>',unsafe_allow_html=True)
#             st.image(image,width=350)  
        
#         f_image, e_image = st.columns( [1, 1])
#         with n_image:
            
#             st.markdown('<p style="text-align: center;">Noisy Image</p>',unsafe_allow_html=True) 
#             if selected_noise == "Uniform Noise":
             
#                 noisy_image = fs.add_uniform_noise(img, a=0, b=sigma_noise)

#                 st.image(noisy_image, caption='Uniform Noise', width=350)
#             elif selected_noise == "Gaussian Noise":
#                     # SNR to variance conversion
#                     var = np.var(image) / (10**(snr_value / 10))
#                     noisy_image = fs.add_gaussian_noise(img, mean=0, var=var)


#                     st.image(noisy_image, caption='Gaussian Noise', width=350)
#             else:
#                     noisy_image = fs.add_salt_pepper_noise(img, pepper_amount=sigma_noise)

#                     st.image(noisy_image, caption='Salt & Pepper Noise', width=350)

#         with f_image:
#             st.markdown('<p style="text-align: center;">Filtered Image</p>',unsafe_allow_html=True)
#             if selected_filter == "Gaussian Filter":
#                 g_filter = fs.gaussian_filter(noisy_image*255)
#                 g_filter_norm = g_filter / 255.0  # Normalize to [0.0, 1.0]
#                 st.image(g_filter_norm, caption='Gaussian Filter', width=350)
#             elif selected_filter == "Average Filter":
#                     avg_filter = fs.average_filter(noisy_image*255)
#                     # avg_filter_norm = avg_filter / 255.0  # Normalize to [0.0, 1.0]
#                     st.image(avg_filter, caption='Average Filter', width=350)
#             else:
#                     removed_noise = fs.median_filter(noisy_image*255, 3)
#                     removed_noise_norm = removed_noise / 255.0  # Normalize to [0.0, 1.0]
#                     st.image(removed_noise_norm, caption='Median Filter', width=350)
#         with e_image:
#             # ('Sobel','Roberts','Prewitt','Canny Edge'))
#             st.markdown('<p style="text-align: center;">Edge Detection Image</p>',unsafe_allow_html=True)
#             if edge == "Sobel":
            
#                 edge_img = fs.edge_detection(img, 'sobel')
#                 # g_filter_norm = g_filter / 255.0  # Normalize to [0.0, 1.0]
#                 st.image(edge_img, caption='Sobel', width=350)
#             elif edge == "Roberts":
#                     edge_img = fs.edge_detection(img, "roberts")
#                     # avg_filter_norm = avg_filter / 255.0  # Normalize to [0.0, 1.0]
#                     st.image(edge_img, caption='roberts', width=350)
#             elif edge == "Prewitt":
#                     edge_img = fs.edge_detection(img, "prewitt")
#                     # avg_filter_norm = avg_filter / 255.0  # Normalize to [0.0, 1.0]
#                     st.image(edge_img, caption='prewitt', width=350)
#             else:
#                     edge_img = fs.edge_detection(img)
#                     # removed_noise_norm = removed_noise / 255.0  # Normalize to [0.0, 1.0]
#                     st.image(edge_img, caption='Canny', width=350)
        