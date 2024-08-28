# Unofficial Implementation of DiffuseIR

This is an unofficial implementation of the paper "DiffuseIR: Diffusion Models for Isotropic Reconstruction of 3D Microscopic Images"

## Requirements

Ubuntu 22.04 LTS, Python 3.9, PyTorch 2.4.0, CUDA 11.8

## Usage

Since DiffuseIR shares a similar idea with [Repaint](https://github.com/andreas128/RePaint), which only modified the reverse process in inference for conditional guidance, we basically include the dataloader and test code for electron microscope images.

For training, you may refer to the original [Improved Diffusion](https://github.com/openai/improved-diffusion) repository, once you have a trained unconditional diffusion model, you can use the test code `my_test.py` to generate images with conditional guidance.