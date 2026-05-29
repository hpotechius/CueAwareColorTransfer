"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import OpenImageIO as oiio
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class Reader:
    # ----------------------------------------------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_img(path, img_type):
        if img_type == "rgb_exr":
            return Reader.__read_rgb_exr(path)
        elif img_type == "sem_exr":
            return Reader.__read_sem_exr(path)
        elif img_type == "grayscale_exr":
            return Reader.__read_grayscale_exr(path)
        elif img_type == "rgb_png":
            return Reader.__read_rgb_png(path)
        elif img_type == "multilayer_exr":
            return Reader.__read_multilayer_exr(path)
        elif img_type == "rgba_png":
            return Reader.__read_rgba_png(path)
        elif img_type == "grayscale_png":
            return Reader.__read_grayscale_png(path)
        else:
            raise ValueError(f"Unsupported image type: {img_type}. Supported types are: rgb_exr, rgb_png, grayscale_png.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Reads an RGB-PNG image and returns a NumPy array with shape (H, W, 3).
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_rgb_png(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        img = Image.open(path).convert("RGB")
        rgb_array = np.array(img, dtype=np.float32) / 255.0
        return rgb_array
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Reads a  RGB-EXR image and returns a NumPy array with shape (H, W, 3) using OpenImageIO.
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_rgb_exr(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        input_img = oiio.ImageInput.open(path)
        if not input_img:
            raise ValueError(f"{path} is not a valid EXR file.")
        spec = input_img.spec()
        data = input_img.read_image("float")
        input_img.close()
        data = np.asarray(data).reshape(spec.height, spec.width, spec.nchannels)
        if data.shape[2] > 3:
            data = data[:, :, :3]
        return data
    
    # ----------------------------------------------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_grayscale_exr(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        input_img = oiio.ImageInput.open(path)
        if not input_img:
            raise ValueError(f"{path} is not a valid EXR file.")
        spec = input_img.spec()
        if spec.nchannels != 1:
            raise ValueError(f"{path} does not have exactly one channel.")
        data = input_img.read_image("float")
        input_img.close()
        arr = np.asarray(data).reshape(spec.height, spec.width, 1)
        return arr

    # ----------------------------------------------------------------------------------------------------------------------
    # Reads a multi-channel EXR image and returns a NumPy array with shape (H, W, C) using OpenImageIO.
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_sem_exr(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        input_img = oiio.ImageInput.open(path)
        if not input_img:
            raise ValueError(f"{path} is not a valid EXR file.")
        spec = input_img.spec()
        data = input_img.read_image("float")
        input_img.close()
        data = np.asarray(data).reshape(spec.height, spec.width, spec.nchannels)
        return data

    # ----------------------------------------------------------------------------------------------------------------------
    # Reads an RGBA-PNG image and returns a NumPy array with shape (H, W, 4).
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_rgba_png(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        img = Image.open(path).convert("RGBA")
        rgba_array = np.array(img, dtype=np.float32) / 255.0
        return rgba_array

    # ----------------------------------------------------------------------------------------------------------------------
    # Reads an Grayscale-PNG image and returns a NumPy array with shape (H, W, 1).
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_grayscale_png(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        img = Image.open(path).convert("I;16")
        greyscale_array = np.array(img, dtype=np.float32) / 65535.0
        return greyscale_array

    # ----------------------------------------------------------------------------------------------------------------------
    # Reads a multilayer EXR (beauty_exr) as (num_layers, height, width, 3) using OpenImageIO.
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __read_multilayer_exr(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        input_img = oiio.ImageInput.open(file_path)
        if not input_img:
            raise ValueError(f"{file_path} is not a valid EXR file.")
        spec = input_img.spec()
        channelnames = spec.channelnames
        nchannels = spec.nchannels
        data = input_img.read_image("float")
        input_img.close()
        data = np.asarray(data).reshape(spec.height, spec.width, nchannels)
        # Extract layer names (before the dot)
        layer_dict = {}
        from collections import defaultdict
        layer_indices = defaultdict(list)
        for idx, cname in enumerate(channelnames):
            lname, ctype = cname.split(".")
            layer_indices[lname].append((idx, ctype))
        for lname, idxs_types in layer_indices.items():
            # Check whether RGB or grayscale (Y)
            if len(idxs_types) == 1 and idxs_types[0][1] == "Y":
                idx = idxs_types[0][0]
                arr = data[:, :, idx][:, :, np.newaxis]  # (H,W,1)
                layer_dict[lname] = arr
            elif len(idxs_types) == 3:
                # Explicitly sort channels into R,G,B order
                order = ["R", "G", "B"]
                idxs_sorted = [idx for idx, ctype in sorted(idxs_types, key=lambda x: order.index(x[1]))]
                arr = np.stack([data[:, :, i] for i in idxs_sorted], axis=-1)  # (H,W,3)
                layer_dict[lname] = arr
            else:
                # Unknown or mixed channels, ignore
                continue
        return layer_dict

# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class Writer:
    # ----------------------------------------------------------------------------------------------------------------------
    # Writes an image based on the specified type.
    # ----------------------------------------------------------------------------------------------------------------------
    def write_img(path, data, img_type, apply_gamma=True):
        if img_type == "rgb_png":
            Writer.__write_rgb_png(path, data, apply_gamma)
        elif img_type == "rgba_png":
            Writer.__write_rgba_png(path, data, apply_gamma)
        elif img_type == "sem_exr":
            Writer.__write_sem_exr(path, data)
        elif img_type == "multilayer_exr":
            Writer.__write_multilayer_exr(path, data)
        elif img_type == "greyscale_png":
            Writer.__write_greyscale(path, data, apply_gamma, format="PNG")
        elif img_type == "greyscale_webp":
            Writer.__write_greyscale(path, data, apply_gamma, format="WEBP")
        elif img_type == "rgb_exr":
            Writer.__write_rgb_exr(path, data)
        else:
            raise ValueError(f"Unsupported image type: {img_type}. Supported types are: rgb_png, rgb_exr.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Writes an RGB image (float32 or uint8) in PNG format.
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_rgb_png(path, data, apply_gamma=True):
        if apply_gamma:
            gamma = 2.2
            data = np.clip(data, 0, 1) ** (1 / gamma)
        if data.dtype == np.float32:
            data = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(data, mode="RGB")
        img.save(path, format="PNG")
        print(f"PNG written: {path}")

    # ----------------------------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_rgba_png(path, data, apply_gamma=True):
        if apply_gamma:
            gamma = 2.2
            data = np.clip(data, 0, 1) ** (1 / gamma)
        if data.dtype == np.float32:
            data = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(data, mode="RGBA")
        img.save(path, format="PNG")
        print(f"PNG written: {path}")

    # ----------------------------------------------------------------------------------------------------------------------
    # Writes a multi-channel EXR image (sem_exr) using OpenImageIO.
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_sem_exr(path, data):
        height, width, channels = data.shape
        spec = oiio.ImageSpec(width, height, channels, oiio.BASETYPE.FLOAT)
        out = oiio.ImageOutput.create(path)
        if not out:
            raise RuntimeError(f"Could not create EXR file at {path}")
        out.open(path, spec)
        out.write_image(data.astype(np.float32).tobytes())
        out.close()
        print(f"EXR written: {path}")

    # ----------------------------------------------------------------------------------------------------------------------
    # Writes a greyscale image (float32 or uint8) in PNG format.
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_greyscale(path, data, apply_gamma, format="PNG"):
        if apply_gamma:
            gamma = 2.2
            data = np.clip(data, 0, 1) ** (1 / gamma)
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[..., 0]
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(data, mode="L")
        img.save(path, format=format)
        print(f"Greyscale {format} written: {path}")

    # ----------------------------------------------------------------------------------------------------------------------
    # Writes an RGB image (float32) in EXR format using OpenImageIO.
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_rgb_exr(path, data):
        height, width = data.shape[:2]
        # Ensure channels are in RGB order
        if data.shape[2] == 3:
            data_rgb = data[..., [0, 1, 2]]  # [R,G,B]
        else:
            raise ValueError("Input must have 3 channels (RGB)")
        spec = oiio.ImageSpec(width, height, 3, oiio.BASETYPE.FLOAT)
        spec.channelnames = ["R", "G", "B"]
        out = oiio.ImageOutput.create(path)
        if not out:
            raise RuntimeError(f"Could not create EXR file at {path}")
        out.open(path, spec)
        out.write_image(np.ascontiguousarray(data_rgb, dtype=np.float32))
        out.close()
        print(f"EXR written: {path}")

    # ----------------------------------------------------------------------------------------------------------------------
    # Writes the multilayer stack of size (X,W,H,3) in EXR format using OpenImageIO.
    # Each layer is written as consecutive channels: layer0_R, layer0_G, layer0_B, layer1_R, ...
    # ----------------------------------------------------------------------------------------------------------------------
    def __write_multilayer_exr(file_path: str, pass_dict: dict):
        # pass_dict: {layer_name: np.ndarray (H,W,3) or (H,W,1) or (H,W)}
        layer_names = list(pass_dict.keys())
        arrays = []
        channelnames = []
        for lname in layer_names:
            arr = np.asarray(pass_dict[lname], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.shape[2] == 1:
                arrays.append(arr)
                channelnames.append(f"{lname}.Y")
            elif arr.shape[2] == 3:
                arrays.append(arr)
                channelnames.extend([f"{lname}.{c}" for c in ["R", "G", "B"]])
            else:
                raise ValueError(f"Layer '{lname}' must have 1 or 3 channels, got shape {arr.shape}")
        # Check Shapes
        shapes = [arr.shape[:2] for arr in arrays]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("All layers must have the same height and width.")
        height, width = shapes[0]
        total_channels = len(channelnames)
        spec = oiio.ImageSpec(width, height, total_channels, oiio.BASETYPE.FLOAT)
        spec.channelnames = channelnames
        out = oiio.ImageOutput.create(file_path)
        if not out:
            raise RuntimeError(f"Could not create EXR file at {file_path}")
        out.open(file_path, spec)
        # (height, width, total_channels)
        data = np.concatenate(arrays, axis=2)
        data = np.ascontiguousarray(data, dtype=np.float32)
        out.write_image(data)
        out.close()

# ----------------------------------------------------------------------------------------------------------------------
# ...
# ----------------------------------------------------------------------------------------------------------------------
def find_single_stack(folder: Path, suffix: str) -> Path:
    matches = list(folder.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found in {folder}")
    if len(matches) > 1:
        raise RuntimeError(f"More than one *{suffix} file found in {folder}: {matches}")
    return matches[0]

# ----------------------------------------------------------------------------------------------------------------------
# ...
# ----------------------------------------------------------------------------------------------------------------------
def reconstruct_beauty_pass(beauty_dict: dict) -> np.ndarray:
    refractions_composite = beauty_dict["RefractionsFilter"] * beauty_dict["RefractionsRaw"]
    reflections_composite = beauty_dict["ReflectionsFilter"] * beauty_dict["ReflectionsRaw"]
    translucency_composite = beauty_dict["TransTint"] * (beauty_dict["TransGIRaw"] + beauty_dict["TransLightingRaw"])
    diffuse_composite = beauty_dict["DiffuseFilter"] * (beauty_dict["DiffuseLightingRaw"] + beauty_dict["GIRaw"] + beauty_dict["SSSRaw"] + beauty_dict["CausticsRaw"])
    beauty = beauty_dict["Background"] + refractions_composite + beauty_dict["SpecularLighting"] + reflections_composite + translucency_composite + diffuse_composite

    return beauty