# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


def export_to_torchscript(model_path: str):
    """
    Loads PyTorch model file and returns corresponding TorchScript model.

    Args:
        model_path (str): path to the PyTorch model.

    Returns: TorchScript model

    """
    model = torch.load(model_path, map_location="cpu")
    # Switch the model to eval model
    model.eval()
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 1, 512, 512)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save the TorchScript model
    traced_script_module.save("../saved_models/traced_pspnet_model.pt")
    print("model saved")


def load_torchscript_model(model_path: str) -> torch.jit._script.RecursiveScriptModule:
    """
    Load TorchScript model and run inference

    Args:
        model_path (str): TorchScript model path

    Returns: TorchScript model

    """
    model = torch.jit.load(model_path, map_location="cpu")
    # Random input
    # rand = torch.rand(size=(1, 1, 512, 512))
    # Image input
    img = Image.open(
        "/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/data/segmentation/simple_cod_subset/images/19_07_2016_05h02m31s573ms.jpg"
    ).convert("L")
    img = transforms.Resize(size=(512, 512))(img)  # Applying input transforms
    img = transforms.ToTensor()(img)  # Input to tensor type

    with torch.no_grad():
        pred = model(img.unsqueeze(dim=0)).numpy().squeeze()
    return model


if __name__ == "__main__":
    path = f"../saved_models/pspnet.pt"
    # export_to_torchscript(path)
    model = load_torchscript_model(model_path="../saved_models/traced_pspnet_model.pt")
