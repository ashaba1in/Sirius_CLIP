import gradio as gr
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from generate_from_clip import generate

def greet(input_text, input_image):
    if len(input_text) == 0:
        # нет текста
        if type(input_image) != type(None):
            # есть картинка
            return (input_image, input_image, input_image, input_image, input_image)
        else:
            # ничего нет
            return
    elif type(input_image) == type(None):
        # нет фото - генерируем только по тексту
        img = generate(input_text, use_img=False)
        for i in range(5):
            img[i] = transform(img[i].permute(2, 0, 1))
        return (img[0], img[1], img[2], img[3], img[4])

    else:
        # всё есть - генерируем
        img = generate(input_text, input_image, use_img=True)
        for i in range(5):
            img[i] = transform(img[i].permute(2, 0, 1))
        return (img[0], img[1], img[2], img[3], img[4])

height = 256
width = 256
img1_s = gr.Image(label="image1", tool="editor").style(height=height, width=width)
img2_s = gr.Image(label="image2", tool="editor").style(height=height, width=width)
img3_s = gr.Image(label="image3", tool="editor").style(height=height, width=width)
img4_s = gr.Image(label="image4", tool="editor").style(height=height, width=width)
img5_s = gr.Image(label="image5", tool="editor").style(height=height, width=width)
demo = gr.Interface(fn=greet, inputs=[gr.Textbox(placeholder="Request Here..."), gr.Image()], outputs=[img1_s, img2_s, img3_s, img4_s, img5_s], allow_flagging="never", description='Генерация изображений при помощи нейросети CLIP<br>Введите текстовый запрос, по которому будет сгенерировано изображение. К тексту можно добавить своё изображение, на основе которого будет получено новое')

demo.launch(debug=True, share=True)