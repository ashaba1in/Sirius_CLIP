import gradio as gr
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from generate_from_clip import generate

def greet(input_text):
    transform = ToPILImage()
    img = generate(input_text)
    for i in range(3):
        img[i] = transform(img[i].permute(2, 0, 1))
    print('READY')
    print(type(img[0]))
    return (img[0], img[1], img[2])

def get_demo():
    height = 256
    width = 256
    img1_s = gr.Image(label="image1", tool="editor", type="pil").style(height=height, width=width)
    img2_s = gr.Image(label="image2", tool="editor", type="pil").style(height=height, width=width)
    img3_s = gr.Image(label="image3", tool="editor", type="pil").style(height=height, width=width)
    demo = gr.Interface(fn=greet, inputs=[gr.Textbox(placeholder="Request Here...")], outputs=[img1_s, img2_s, img3_s], allow_flagging="never", description='Генерация изображений при помощи нейросети CLIP<br>Введите текстовый запрос, по которому будет сгенерировано изображение.')

    return demo

demo = get_demo()
demo.launch(share=True, debug=True, enable_queue=True)
