import torch

import numpy as np
import pickle
import os
from PIL import Image

def getImagePaths(raw, mask, background,  image1, image2, image3):
    I_1_path = os.path.join(raw, image1)
    M_1_path = os.path.join(mask, image1.split('.')[0] + '.png')
    I_2_path = os.path.join(raw, image2)
    M_2_path = os.path.join(mask, image2.split('.')[0] + '.png')
    I_3_path = os.path.join(raw, image3)
    M_3_path = os.path.join(mask, image3.split('.')[0] + '.png')

    return {
        'I_1_path': I_1_path,
        'I_2_path': I_2_path,
        'I_3_path': I_3_path,
        'M_1_path': M_1_path,
        'M_2_path': M_2_path,
        'M_3_path': M_3_path,
    }


def addBatchDim(listOfVariables):
    modifiedVariables = []

    for var in listOfVariables:
        var = torch.unsqueeze(var, axis=0)
        modifiedVariables.append(var)

    return modifiedVariables


def writeMaskToDisk(li, names, dest):
    for idx, var in enumerate(li):
        var = makeMask(var)
        var = var[0, :, :, 0]
        var = Image.fromarray(var)
        var.save(os.path.join(dest, names[idx]))


def makeMask(tensor):
    return (
        tensor.detach()
        .clamp_(min=0, max=1)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def writeImageToDisk(li, names, dest):
    for idx, var in enumerate(li):
        var = makeImage(var)
        var = var[0]
        var = Image.fromarray(var)
        var.save(os.path.join(dest, names[idx]))


def makeImage(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def writePickleToDisk(li, names, dest):
    for idx, var in enumerate(li):
        with open(os.path.join(dest, names[idx]), "wb") as handle:
            pickle.dump(
                li[idx].detach().cpu().numpy(),
                handle
            )


