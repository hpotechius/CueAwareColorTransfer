# CueAwareColorTransfer
![](https://img.shields.io/badge/build-24.04.3%20LTS-orange?logo=ubuntu&label=Ubuntu) ![python3.12.6](https://img.shields.io/badge/build-3.12.6-blue?logo=python&label=Python) ![](https://img.shields.io/badge/build-MIT-purple?label=License) ![](https://img.shields.io/badge/build-GeForce%20RTX%204060%20Ti-white?logo=nvidia&label=GPU) ![](https://img.shields.io/badge/build-intel%20Core%20i7--14700KF-white?logo=intel&label=CPU) ![](https://img.shields.io/badge/macOS-Tahoe-deepskyblue?logo=apple) ![](https://img.shields.io/badge/Apple%20Silicon-M2-white)

This repository contains the color transfer methods and experiments presented in the paper _The Impact of Intrinsic Scene Cues on Perceived Color Transfer Quality_.

## Dataset
This color transfer method is specifically designed for the [Intrinsic Decomposition Dataset (IDD)](https://huggingface.co/datasets/hpotechius/IntrinsicDecompositionDataset), which is described in detail in the associated publication and [supplementary material](https://potechius.com/ICIP2026ColorTransfer/). Visit our [viewer](https://potechius.com/IDD-Viewer/) to explore the dataset and its contents.

## Usage

Quick example to run the pipeline from the project root:

```bash
python main.py \
    --src_folder "testdata/input/S1_V4_C3_A1_I4" \
    --ref_folder "testdata/input/S0_V4_C1_A2_I5" \
    --out_folder "testdata/output/S1_V4_C3_A1_I4-S0_V4_C1_A2_I5" \
    --colortransfer reinhard \
    --config A C \
    --plotsEnabled
```

Arguments:

- `src_folder`: Path to the source folder containing the EXR stacks.
- `ref_folder`: Path to the reference folder containing the EXR stacks.
- `out_folder`: Path to the output folder where results will be saved.
- `colortransfer`: Choose the color transfer method, e.g. `reinhard` or `optimal`.
- `config`: Combination of cues using letters `A` (illumination), `B` (geometry) and `C` (semantic).
- `plotsEnabled`: Enable plotting of color distributions (3D scatter).

Notes:

- Multiple cue letters can be provided to `--config` (e.g. `--config A B C`).
- Output and plots will be written to the directory given by `--out_folder`.
- Ensure dependencies listed in `requirements.txt` are installed before running.

Required input files:

- Both the `src_folder` and the `ref_folder` must contain three EXR stacks with the following filename endings:
    - `_beauty_stack.exr`
    - `_props_stack.exr`
    - `_semantic_stack.exr`

    The pipeline will automatically discover these files in the provided folders; please ensure each exists exactly once per folder.

Output folder naming:

- The pipeline will append a canonical suffix to the `--out_folder` you provide. The suffix has the form `-<Algo><Cues>` where `<Algo>` is `R` (for `reinhard`) or `O` (for `optimal`) and `<Cues>` is the concatenation of any selected cues in alphabetical order (`A`, `B`, `C`).
- Example: if you pass `--out_folder "testdata/output/S1_V4_C3_A1_I4-S0_V4_C1_A2_I5" --colortransfer reinhard --config A B C`, the actual output directory created will be:

```
testdata/output/S1_V4_C3_A1_I4-S0_V4_C1_A2_I5-RABC
```

If no cues are provided the suffix will still include the algorithm letter, e.g. `-R` or `-O` depending on the method.

## Example

<p align="center">
  <img src="https://github.com/user-attachments/assets/e14071f6-0c49-4668-be1f-858301e241f3" width="800">
  <br>
  <em>Figure 1: Source and reference images with corresponding color distributions in RGB color space..</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/360f72f3-24e5-43bd-88ad-fbf0baf09817" width="800">
  <br>
  <em>Figure 2: Results from Reinhard's method with different scene cues enabled.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/712db8b0-21b8-4d62-b5aa-f20d6090c2a6" width="800">
  <br>
  <em>Figure 3: Results from the optimal transport method with different scene cues enabled.</em>
</p>

## Citation
If you utilize this code in your research, kindly provide a citation:
```
@inproceedings{potechius2026,
    author={Potechius, H. and Sikora, T. and Knorr, S.},
    booktitle={IEEE International Conference on Image Processing (ICIP)}, 
    title={The Impact of Intrinsic Scene Cues on Perceived Color Transfer Quality}, 
    year={2026},
    location = {Tampere, Finland}
}
```
