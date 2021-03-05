# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
from typing import List, Tuple

import imageio
from io import BytesIO
from PIL import Image
import hydra
from omegaconf import open_dict, DictConfig
from typing import Mapping

import numpy as np
import os
import streamlit as st
from dataloader.ufs_data import UnknownTestData

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


@st.cache(
    hash_funcs={torch.jit._script.RecursiveScriptModule: id}, allow_output_mutation=True
)
def load_model(
    model_path: str = "saved_models/traced_pspnet_model.pt",
) -> torch.ScriptModule:
    """Loads and returns the torchscript model from path.

    Parameters
    ----------
    model_path : str, optional
        The path to the torchscript model, by default "models/face_blur.pt"

    Returns
    -------
    torch.ScriptModule
        The loaded torchscript model object.
    """
    return torch.jit.load(model_path, map_location="cpu")


@st.cache(
    hash_funcs={torch.jit._script.RecursiveScriptModule: id}, allow_output_mutation=True
)
def get_inference_on_bytes_image(
    model: torch.ScriptModule, image_file: BytesIO
) -> Tuple[Image.Image, Image.Image]:
    """Runs model inference on a BytesIO image stream.

    Parameters
    ----------
    model : torch.ScriptModule
        The TorchScript loaded module.
    image_file : BytesIO
        The image file uploaded returned as a BytesIO object.

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        A tuple of input image to the model and the output from the model.
    """
    if isinstance(image_file, st.uploaded_file_manager.UploadedFile):
        raw_image = Image.open(image_file).convert("L")
    else:
        raw_image = Image.fromarray(image_file).convert("L")
    # print(f"model type {type(model)}")
    image = torch.tensor(np.array(raw_image)).unsqueeze(dim=0).unsqueeze(dim=0)
    # print(image.dtype)
    img = transforms.Resize(size=(512, 512), interpolation=Image.BILINEAR)(image)
    img = img.float()
    output = model(img).squeeze()
    print(f"output shape is {output.dtype}")
    # return Image.fromarray(image[0].detach().numpy()), Image.fromarray(
    #     output[0].detach().numpy()
    # )
    return img.detach().numpy().squeeze().astype(np.uint8), output.detach().numpy()


def get_config_from_sidebar() -> dict:
    """Defines and returns the configuration in the sidebar.

    Returns
    -------
    dict
        A dict of the different inputs chosen by the user.
    """

    st.sidebar.markdown("## Upload image/video data and annotation")

    upload_video = st.sidebar.checkbox("Upload video?", value=False)
    with st.sidebar.beta_expander("See explanation"):
        st.write(
            """
            If True, you can upload video.

            If False, you can upload images.
        """
        )

    if upload_video:

        video_file = st.sidebar.file_uploader(
            "Upload your video here (mp4, avi).", type=["mp4", "avi"]
        )

        if video_file is None:
            st.stop()

        video = imageio.get_reader(video_file, "ffmpeg")
        image_files = []
        for image in video.iter_data():
            image_files.append(image)

    else:

        image_files = st.sidebar.file_uploader(
            "Upload your image/s here.",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    if len(image_files) == 0:
        st.stop()

    elif len(image_files) == 1:
        image_idx = 0

    else:
        num_images = len(image_files)
        image_idx = st.sidebar.slider("Choose an image index", 0, num_images - 1, 0)

    fit_to_column = st.sidebar.checkbox("Fit images to column width")

    return {
        "image": image_files[image_idx],
        "fit_to_column": fit_to_column,
    }

def sidebar(cfg: Mapping):

    # add a sidebar heading
    st.sidebar.markdown("# Choose images")

    # possible model options
    model_options = ["pspnet", "unet", "deeplabv3"]
    # default dataset index
    default_model_idx = 0  # corresponds to pspnet model

    # dataset to choose
    dataset = st.sidebar.selectbox(
        "Which Model to choose?", model_options, default_model_idx
    )
    transform = transforms.Compose(
        [
            transforms.Resize(size=(cfg.data.width, cfg.data.height), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    dataset_obj = UnknownTestData(
        cfg,
        data_root=cfg.st_data.data_dir,
        bbox_dir=None, # cfg.st_data.bbox_dir
        transform=transform,
    )

    # get number of images in the dataset
    num_images = len(dataset_obj)

    # get image index
    image_idx = st.sidebar.slider("Choose an image index", 0, num_images - 1, 0)
    print(f"image idx is {image_idx}")

    # get image path
    image_path = dataset_obj.img_files[image_idx]
    # get processed image and target
    image = dataset_obj[image_idx]

    return image, image_idx, image_path


@hydra.main(config_name="../config")
def main(cfg: DictConfig):
    """Main function for the script.

    Defines the user input configuration, loads and runs the model and visualizes
    the output.
    """
    model_path = cfg.st_data.model_path
    save_path = cfg.st_data.save_path
    image, image_idx, image_path = sidebar(cfg)
    model = load_model(model_path=model_path)

    # construct UI layout
    st.title("Semantic Segmentation of Aquatic Organisms in Underwater Videos")

    # get the raw image to display on the web page
    raw_image = Image.open(image_path).convert("L")

    # get the model prediction
    prediction = model(image.unsqueeze(dim=0)).detach().squeeze().numpy()
    prediction = np.where(prediction > 0.6, 255, 0).astype(np.uint8)

    resized_pred = Image.fromarray(prediction).resize(size=(raw_image.width, raw_image.height))

    st.image(
        [raw_image, resized_pred],
        caption=["original image", "predicted mask"],
        width=224
        # use_column_width=True,
    )

    # st.image(
    #     prediction,
    #     caption="predicted mask",
    #     width=224
    #     # use_column_width=True,
    # )
    is_ok = st.sidebar.button(label="Segmentation OK?")
    if is_ok:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, os.path.basename(image_path[:-3]) + "png")
        resized_pred.save(out_path)
        st.sidebar.write("Mask saved as ground-truth")
    else:
        # st.sidebar.write("Mask is not perfect and is not saved.")
        print("Mask is not perfect and is not saved.")



if __name__ == "__main__":
    main()
