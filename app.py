from fawkes.protection import Fawkes
import gradio as gr
import os

def predict(level, img):
  # print(img)
  fwks = Fawkes("extractor_2", '0', 1, mode=level)
  fwks.run_protection([img], format='jpeg')
  splt = img.split(".")
  print(os.listdir('/tmp'))
  return splt[0] + "_cloaked.jpeg"

gr.Interface(fn=predict, inputs=[gr.components.Dropdown(["low", "mid", "high"], label="Protection Level"),
                                 gr.components.Image(type='filepath')],
                                 outputs=gr.components.Image(type="pil")).launch(show_error=True, server_name="0.0.0.0")
