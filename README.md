<p align="center">
<a href="https://www.python.org/downloads/">
        <img alt="Build" src="https://img.shields.io/badge/3.10-Python-green">
</a>
<a href="https://pytorch.org">
        <img alt="Build" src="https://img.shields.io/badge/2.0-PyTorch-green">
</a>
<a href="https://huggingface.co/stabilityai/stable-diffusion-2">
        <img alt="Build" src="https://img.shields.io/badge/StabilityAI-Stable_Diffusion v2-green">
</a>
<a href="https://huggingface.co/">
        <img height=20px alt="Build" src="https://huggingface.co/datasets/huggingface/badges/raw/main/powered-by-huggingface-light.svg">
</a>
<br>

</p>

<h1 align="center">
    <p>Leveraging Optimization for Adaptive Attacks on Image Watermarks</p>
</h1>

This repository is the official implementation of two papers 
*"Leveraging Optimization for Adaptive Attacks on Image Watermarks"*.

* KeyGen: Generate watermarking keys 
* Embed: Embed a watermark into a generator
* Verify: Check whether the watermark is present in an image
* Adaptive-Attack: Instantiate an adaptive attack against a set of watermarked images.

We implement these functions for all five watermarking methods presented in the paper. 


If you find our code or paper useful, please cite
```
@article{lukas2023leveraging,
  title = {Leveraging Optimization for Adaptive Attacks on Image Watermarks},
  author = {Lukas, Nils and Diaa, Abdulrahman and Fenaux, Lucas and Kerschbaum, Florian},
  journal = {The Twelfth International Conference on Learning Representations (ICLR 2024)},
  year = {2024}
}
```
