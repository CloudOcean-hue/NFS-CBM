# NFS-CBM

Official PyTorch implementation of **Non-negative Factorized Concept Bottleneck Models for Automatic Concept Discovery with Generative Models**.

## Abstract

NFS-CBM is an end-to-end concept bottleneck model for automatic concept discovery and generative visual explanation. Its Factorized Semantic Bottleneck uses non-negative concept activations and a decorrelated global basis to organize discriminative visual information into compact concept units. A concept-conditioned diffusion generator maps these concepts to image-space interventions. After training, a vision-language model summarizes consistent intervention effects and assigns readable semantic names to the discovered concepts.

## Method Overview

<img src="assets/method_pipeline.svg" width="100%">

## Main Results

NFS-CBM is evaluated on CUB-200-2011, Stanford Cars, and ImageNet-100 with ResNet-50 and ViT-L/14 backbones. The main accuracy comparison is shown below.

<img src="assets/accuracy_table.svg" width="100%">

## Concept Naming

The [automatic concept-naming pipeline](concept_naming/) selects the **5 highest-activation samples per concept** by default and uses their before/after intervention pairs to assign a semantic name. Both GPT and local Qwen3-VL naming backends are supported.

## To-do

- [x] Release the concept-naming pipeline.
- [ ] Release the full training and evaluation code.
- [ ] Release pretrained models and reproduction instructions.

The full code and pretrained models will be released after publication.
