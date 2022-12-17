import gradio as gr
import numpy as np
from PIL import Image

def greet(input_text, input_image):
    #input_text = input_request[0]
    #input_image = input_request[1]

    if len(input_text) == 0:
      # нет текста - сразу ошибка
      output_img1 = Image.open('/content/drive/MyDrive/error-rubber-stamp-word-error-inside-illustration-109026446.jpg')
    elif type(input_image) == type(None):
      # нет фото - генерируем только по тексту
      output_img1 = Image.open('/content/drive/MyDrive/depositphotos_1045357-stock-photo-ok-button.jpg')
    else:
      # всё есть - генерируем
      output_img1 = input_image

    return (output_img1, output_img1, output_img1, output_img1, output_img1)

height = 256
width = 256
img1_s = gr.Image().style(height=height, width=width)
img2_s = gr.Image().style(height=height, width=width)
img3_s = gr.Image().style(height=height, width=width)
img4_s = gr.Image().style(height=height, width=width)
img5_s = gr.Image().style(height=height, width=width)
demo = gr.Interface(greet, inputs=[gr.Textbox(placeholder="Request Here..."), gr.Image()], outputs=[img1_s, img2_s, img3_s, img4_s, img5_s], allow_flagging="never")

demo.launch(debug=True, share=True)
