from fawkes.protection import Fawkes
from fawkes.utils import Faces, reverse_process_cloaked, load_extractor
from fawkes.differentiator import FawkesMaskGeneration
from keras.preprocessing import image
import numpy as np
import gradio as gr
from PIL import ExifTags

IMG_SIZE = 112
PREPROCESS = 'raw'

# To pre-emptively download the files at boot
fwks_l = Fawkes("extractor_2", '0', 1, mode='low')
fwks_m = Fawkes("extractor_2", '0', 1, mode='mid')
fwks_h = Fawkes("extractor_2", '0', 1, mode='high')

def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X

def predict(img, level, th=0.04, sd=1e7, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False, exp="", maximize=True,
                       save_last_on_failed=True):
  
  img = img.convert('RGB')
  img = image.img_to_array(img)

  if level == 'low':
    fwks = fwks_l
  elif level == 'mid':
    fwks = fwks_m
  elif level == 'high':
    fwks = fwks_h

#   fwks = Fawkes("extractor_2", '0', 1, mode=level)

  current_param = "-".join([str(x) for x in [fwks.th, sd, fwks.lr, fwks.max_step, batch_size, format,
                                              separate_target, debug]])
  faces = Faces(['./Current Face'], [img], fwks.aligner, verbose=1, no_align=False)
  original_images = faces.cropped_faces

  if len(original_images) == 0:
      raise Exception("No face detected. ")
  original_images = np.array(original_images)

  if current_param != fwks.protector_param:
      fwks.protector_param = current_param
      if fwks.protector is not None:
          del fwks.protector
      if batch_size == -1:
          batch_size = len(original_images)
      fwks.protector = FawkesMaskGeneration(fwks.feature_extractors_ls,
                                            batch_size=batch_size,
                                            mimic_img=True,
                                            intensity_range=PREPROCESS,
                                            initial_const=sd,
                                            learning_rate=fwks.lr,
                                            max_iterations=fwks.max_step,
                                            l_threshold=fwks.th,
                                            verbose=debug,
                                            maximize=maximize,
                                            keep_final=False,
                                            image_shape=(IMG_SIZE, IMG_SIZE, 3),
                                            loss_method='features',
                                            tanh_process=True,
                                            save_last_on_failed=save_last_on_failed,
                                            )
  protected_images = generate_cloak_images(fwks.protector, original_images)
  faces.cloaked_cropped_faces = protected_images

  final_images, images_without_face = faces.merge_faces(
      reverse_process_cloaked(protected_images, preprocess=PREPROCESS),
      reverse_process_cloaked(original_images, preprocess=PREPROCESS))

  # print(final_images)

  return final_images[-1].astype(np.uint8)
  print("Done!")


  fwks.run_protection([img], format='jpeg')
  splt = img.split(".")
  # print(os.listdir('/tmp'))
  return splt[0] + "_cloaked.jpeg"

gr.Interface(fn=predict, inputs=[gr.components.Image(type='pil'),
                                 gr.components.Radio(["low", "mid", "high"], label="Protection Level")],
                                 outputs=gr.components.Image(type="numpy"), allow_flagging="never").launch(show_error=True, quiet=False)
