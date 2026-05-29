"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import os
import numpy as np
import time
import argparse
from pathlib import Path
from utils.image_tools import Reader, Writer, find_single_stack, reconstruct_beauty_pass
from utils.color_transfer import reinhard_color_transfer, color_transfer_rbf_ot, semantic_color_transfer
from utils.plotting import plot_color_distribution_3d

# ----------------------------------------------------------------------------------------------------------------------
# Usage:
# python main.py 
#        --src_folder "testdata/input/S0_V0_C0_A0_I0"
#        --ref_folder "testdata/input/S0_V1_C3_A0_I0"
#        --out_folder "testdata/output/S0_V0_C0_A0_I0-S0_V1_C3_A0_I0"
#        --colortransfer reinhard 
#        --config A C 
#        --plotsEnabled
#
# src_folder: Path to the source folder containing the EXR stacks
# ref_folder: Path to the reference folder containing the EXR stacks
# out_folder: Path to the output folder where results will be saved
# colortransfer: "reinhard" or "optimal"
# config: combination of cues A (illumination), B (geometry) and C (semantic)
# plotsEnabled: whether to plot the color distributions (3D scatter and histogram)
#
# python main.py --src_folder "testdata/input/S1_V4_C3_A1_I4" --ref_folder "testdata/input/S0_V4_C1_A2_I5" --out_folder "testdata/output/S0_V0_C0_A0_I0-S0_V1_C3_A0_I0" --colortransfer reinhard --config A --plotsEnabled
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # [1] Process parameters
    print("Process parameters...")
    parser = argparse.ArgumentParser(description="Run color transfer")
    parser.add_argument("--src_folder", required=True, help="Path to the source folder")
    parser.add_argument("--ref_folder", required=True, help="Path to the reference folder")
    parser.add_argument("--out_folder", required=True, help="Path to the output folder")
    parser.add_argument("--colortransfer", choices=["reinhard", "optimal"], default="reinhard")
    parser.add_argument("--config", nargs="+", help="Cues e.g. A C (Multiple cues can be combined, e.g. A B C)")
    parser.add_argument("--plotsEnabled", action="store_true", help="Enable plotting of color distributions")
    args = parser.parse_args()

    ct_t = args.colortransfer
    kom = tuple(args.config) if args.config else tuple()
    out_folder_name = args.out_folder
    print(f"Read source folder: {args.src_folder}")
    print(f"Read reference folder: {args.ref_folder}")
    print(f"Selected color transfer type: {ct_t}")
    print(f"Selected cues: {kom}")
    print(f"Plots enabled: {args.plotsEnabled}")
    print(f"Save results to folder: {out_folder_name}")

    # [2] Read source and reference images
    print("Read source and reference images...")
    src_folder = Path(args.src_folder)
    ref_folder = Path(args.ref_folder)

    src_bea_stack = Reader.read_img(str(find_single_stack(src_folder, "_beauty_stack.exr")), "multilayer_exr")
    src_pro_stack = Reader.read_img(str(find_single_stack(src_folder, "_props_stack.exr")), "multilayer_exr")
    src_sem_stack = Reader.read_img(str(find_single_stack(src_folder, "_semantic_stack.exr")), "multilayer_exr")

    ref_bea_stack = Reader.read_img(str(find_single_stack(ref_folder, "_beauty_stack.exr")), "multilayer_exr")
    ref_pro_stack = Reader.read_img(str(find_single_stack(ref_folder, "_props_stack.exr")), "multilayer_exr")
    ref_sem_stack = Reader.read_img(str(find_single_stack(ref_folder, "_semantic_stack.exr")), "multilayer_exr")

    # [3] Reconstruct the beauty Pass
    print("Reconstruct the beauty Pass...")
    src_beauty_pass = reconstruct_beauty_pass(src_bea_stack)
    ref_beauty_pass = reconstruct_beauty_pass(ref_bea_stack)

    # [4] Apply gamma encoding
    print("Apply gamma encoding...")
    src_beauty_pass = np.clip(src_beauty_pass, 0, 1) ** (1 / 2.2)
    ref_beauty_pass = np.clip(ref_beauty_pass, 0, 1) ** (1 / 2.2)

    # Add itendifiers for the color transfer type to output folder name
    ctt = "-R" if ct_t == "reinhard" else "-O"
    out_name = out_folder_name + ctt

    # This boolean controls whether the output image should be gamma-encoded before saving. For the beauty pass reconstruction we apply gamma encoding.
    apply_gamma = False

    src_in = src_beauty_pass
    ref_in = ref_beauty_pass

    # Weights for the color transfer (initialized to 1, modified based on cues)
    src_weights = np.ones((src_in.shape[0], src_in.shape[1], 1), dtype=src_in.dtype)
    ref_weights = np.ones((ref_in.shape[0], ref_in.shape[1], 1), dtype=ref_in.dtype)

    # Illumination-aware
    if "A" in kom:
        out_name += "A"

        # Outside areas have to be masked out, because they have no color information
        print("Mask outside areas (only necessary for illumination-aware color transfer)...")
        src_outside_mask = src_sem_stack["Outside (M9)"]
        src_binary_mask = (src_outside_mask != 0)
        src_inverted_mask = 1 - src_binary_mask
        ref_outside_mask = ref_sem_stack["Outside (M9)"]
        ref_binary_mask = (ref_outside_mask != 0)
        ref_inverted_mask = 1 - ref_binary_mask

        # For the illumination-aware color transfer, we only use the diffuse filter as input.
        src_in = src_bea_stack["DiffuseFilter"]
        ref_in = ref_bea_stack["DiffuseFilter"]
        # The weights are set to 0 for the outside areas, so they do not contribute to the color transfer.
        src_weights *= src_inverted_mask
        ref_weights *= ref_inverted_mask

    # Geometry-aware
    if "B" in kom:
        out_name += "B"

        # The weights are multiplied by the "Size" channel from the props stack, so that larger objects have more influence on the color transfer.
        src_weights *= src_pro_stack["Size"]
        ref_weights *= ref_pro_stack["Size"]

    # Semantic-aware
    if "C" in kom:
        out_name += "C"

    if ct_t == "reinhard":
        if "C" in kom:
            output = semantic_color_transfer(src_in, ref_in, src_weights, ref_weights, src_sem_stack, ref_sem_stack, "reinhard")
        else:
            output = reinhard_color_transfer(src_in, ref_in, src_weights, ref_weights)
    elif ct_t == "optimal":
        if "C" in kom:
            output = semantic_color_transfer(src_in, ref_in, src_weights, ref_weights, src_sem_stack, ref_sem_stack, "optimal", parallel=False)
        else:
            output = color_transfer_rbf_ot(src_in, ref_in, src_weights, ref_weights)

    # Beauty Pass reconstruction
    if "A" in kom:
        src_bea_stack_A = src_bea_stack.copy()
        src_bea_stack_A["DiffuseFilter"] = output
        output_diff = output
        output = reconstruct_beauty_pass(src_bea_stack_A)
        apply_gamma = True

    out_path = out_name
    os.makedirs(out_path, exist_ok=True)
    # write source, reference and output images
    Writer.write_img(f"{out_path}/src.png", src_beauty_pass, "rgb_png", apply_gamma=False)
    Writer.write_img(f"{out_path}/ref.png", ref_beauty_pass, "rgb_png", apply_gamma=False)
    Writer.write_img(f"{out_path}/out.png", output, "rgb_png", apply_gamma=apply_gamma)


    if args.plotsEnabled:
        src_plot_img = plot_color_distribution_3d(src_beauty_pass, filename=out_path)
        Writer.write_img(f"{out_path}/src_plot.png", src_plot_img, "rgb_png", apply_gamma=False)

        ref_plot_img = plot_color_distribution_3d(ref_beauty_pass, filename=out_path)
        Writer.write_img(f"{out_path}/ref_plot.png", ref_plot_img, "rgb_png", apply_gamma=False)

        out_plot_img = plot_color_distribution_3d(output, filename=out_path)
        Writer.write_img(f"{out_path}/out_plot.png", out_plot_img, "rgb_png", apply_gamma=False)