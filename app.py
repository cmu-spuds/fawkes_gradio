from fawkes.protection import Fawkes
from fawkes.utils import Faces, reverse_process_cloaked
from fawkes.differentiator import FawkesMaskGeneration
from keras import utils
import numpy as np
import gradio as gr
import spaces

IMG_SIZE = 112
PREPROCESS = "raw"


def get_extractors():
    hash_map = {
        "extractor_2": "ce703d481db2b83513bbdafa27434703",
        "extractor_0": "94854151fd9077997d69ceda107f9c6b",
    }
    for key, value in hash_map.items():
        utils.get_file(
            fname="{}.h5".format(key),
            origin="http://mirror.cs.uchicago.edu/fawkes/files/{}.h5".format(key),
            md5_hash=value,
            cache_subdir="model",
        )


def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X


def preproc(img):
    img = img.convert("RGB")
    img = utils.img_to_array(img)
    return img


@spaces.GPU
def predict(
    img,
    level,
    sd=1e7,
    format="png",
    separate_target=True,
    debug=False,
    maximize=True,
    save_last_on_failed=True,
    progress=gr.Progress(track_tqdm=True),
):
    img = preproc(img)

    if level == "low":
        fwks = Fawkes("extractor_2", 1, mode="low")
    elif level == "mid":
        fwks = Fawkes("extractor_2", 1, mode="mid")
    elif level == "high":
        fwks = Fawkes("extractor_2", 1, mode="high")

    current_param = "-".join(
        [
            str(x)
            for x in [
                fwks.th,
                sd,
                fwks.lr,
                fwks.max_step,
                -1,
                format,
                separate_target,
                debug,
            ]
        ]
    )
    faces = Faces(["./Current Face"], [img], fwks.aligner, verbose=0, no_align=False)
    original_images = faces.cropped_faces

    if len(original_images) == 0:
        raise Exception("No face detected. ")
    original_images = np.array(original_images)

    if current_param != fwks.protector_param:
        fwks.protector_param = current_param
        if fwks.protector is not None:
            del fwks.protector
        batch_size = len(original_images)
        fwks.protector = FawkesMaskGeneration(
            fwks.feature_extractors_ls,
            batch_size=batch_size,
            mimic_img=True,
            intensity_range=PREPROCESS,
            initial_const=sd,
            learning_rate=fwks.lr,
            max_iterations=fwks.max_step,
            l_threshold=fwks.th,
            verbose=0,
            maximize=maximize,
            keep_final=False,
            image_shape=(IMG_SIZE, IMG_SIZE, 3),
            loss_method="features",
            tanh_process=True,
            save_last_on_failed=save_last_on_failed,
        )
    protected_images = generate_cloak_images(fwks.protector, original_images)
    faces.cloaked_cropped_faces = protected_images

    final_images, _ = faces.merge_faces(
        reverse_process_cloaked(protected_images, preprocess=PREPROCESS),
        reverse_process_cloaked(original_images, preprocess=PREPROCESS),
    )

    return final_images[-1].astype(np.uint8)


# Download extractors pre-emptively
get_extractors()

gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Image(type="pil"),
        gr.components.Radio(["low", "mid", "high"], label="Protection Level"),
    ],
    outputs=gr.components.Image(type="pil"),
    allow_flagging="never",
).launch(show_error=True, quiet=False)
