from fawkes.protection import Fawkes
import gradio as gr
import os

def predict(img, level):
  # print(img)
  fwks = Fawkes("extractor_2", '0', 1, mode=level)
  fwks.run_protection([img], format='jpeg')
  splt = img.split(".")
  print(os.listdir('/tmp'))
  return splt[0] + "_cloaked.jpeg"

gr.Interface(fn=predict, inputs=[gr.components.Image(type='filepath'),
                                 gr.components.Radio(["low", "mid", "high"], label="Protection Level")],
                                 outputs=gr.components.Image(type="pil"), allow_flagging="never").launch(show_error=True, quiet=False)
