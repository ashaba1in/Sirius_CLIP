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
img1_s = gr.Image(label="image1", tool="editor").style(height=height, width=width)
img2_s = gr.Image(label="image2", tool="editor").style(height=height, width=width)
img3_s = gr.Image(label="image3", tool="editor").style(height=height, width=width)
img4_s = gr.Image(label="image4", tool="editor").style(height=height, width=width)
img5_s = gr.Image(label="image5", tool="editor").style(height=height, width=width)
demo = gr.Interface(fn=greet, inputs=[gr.Textbox(placeholder="Request Here..."), gr.Image()], outputs=[img1_s, img2_s, img3_s, img4_s, img5_s], allow_flagging="never", description='Генерация изображений при помощи нейросети CLIP<br>Введите текстовый запрос, по которому будет сгенерировано изображение. К тексту можно добавить своё изображение, на основе которого будет получено новое')

demo.launch(debug=True)