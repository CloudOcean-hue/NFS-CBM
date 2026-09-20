# Automatic concept naming pipeline

This pipeline is an auxiliary analysis tool for a trained NFS-CBM. It names all concept dimensions from the visual changes produced by concept perturbation.

## Pipeline

![NFS-CBM automatic concept naming pipeline](../assets/naming_pipeline.png)

For every test image, the exported model returns class logits and the non-negative concept vector `z`. For each concept dimension `k`, the pipeline selects the `top_m` test samples with the largest `z[k]`. It reconstructs each selected image twice: once with the original vector and once after changing only `z[k]` by the configured perturbation. GPT-5.4 receives the paired reconstructions, dataset context, concept identifier, and fixed naming prompt. One structured result is written for every dimension. The same saved image pairs can alternatively be named with the fully local Qwen3-VL backend described below.

## Required local inputs

The repository supplies the model interface and pipeline. The following files remain local:

```text
checkpoints/nfs_cbm_inference.pt
data/test/<class_name>/<image files>
```

The checkpoint path and dataset path are configured in `configs/naming_pipeline.json`. The example uses virtual paths and does not contain weights.

The checkpoint must be a TorchScript module with:

- `forward(images) -> (logits, concepts)`
- `reconstruct(images, concepts) -> reconstructed_images`

`model/nfs_cbm.py` provides the NFS-CBM inference wrapper, factorized semantic bottleneck, checkpoint export, and checkpoint validation. Pass the trained backbone and concept-conditioned diffusion generator into that wrapper before export.

## Run

```bash
python -m pip install -r requirements.txt
cp configs/naming_pipeline.example.json configs/naming_pipeline.json
export OPENAI_API_KEY="your_api_key"
python concept_naming/pipeline.py --config configs/naming_pipeline.json
```

Use `--dry-run` to load the model, scan the dataset, select samples, perturb every dimension, and generate all image pairs without sending requests to OpenAI:

```bash
python concept_naming/pipeline.py --config configs/naming_pipeline.json --dry-run
```

Use `--resume` to keep completed concept names when rerunning an interrupted job.

## Open-weight Qwen3-VL option

`name_concepts_qwen.py` provides a local naming option based on the Apache-2.0-licensed [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). It uses the model's own visual encoder and autoregressive language component to compare each intervention pair and generate the structured concept name. It does not require an OpenAI API key or send images to an external service.

The released setup fixes the following model revision:

```text
Model: Qwen/Qwen3-VL-8B-Instruct
Weights: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
Revision: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Commit: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

### Install the local backend

Use a separate environment so the Qwen dependencies do not alter the environment used to generate NFS-CBM intervention images:

```bash
python3.12 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install --upgrade pip
python -m pip install -r concept_naming/requirements_qwen.txt
```

Download the pinned weights into the Hugging Face cache:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

The pinned requirements reproduce the tested CUDA 12.4 environment. If the host uses a different CUDA runtime, install the matching PyTorch build first and then install the remaining packages from `requirements_qwen.txt`.

### Name all generated concepts locally

First generate the intervention pairs with the existing NFS-CBM environment. `--dry-run` performs model inference and saves all image pairs without calling OpenAI:

```bash
python concept_naming/pipeline.py \
  --config configs/naming_pipeline.json \
  --dry-run
```

Then activate the Qwen environment and name those same pairs locally:

```bash
source .venv-qwen/bin/activate
HF_HUB_OFFLINE=1 python concept_naming/name_concepts_qwen.py \
  --pairs-root outputs/concept_naming/pairs \
  --dataset cub \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --gpu 0 \
  --max-pairs 5 \
  --output outputs/concept_naming_qwen \
  --resume
```

Use `--dataset cars` for Stanford Cars or `--dataset imagenet100` for ImageNet-100. An explicit description can instead be supplied with `--dataset-context`. The Qwen backend also accepts the same manifest format as `name_concepts.py`:

```bash
HF_HUB_OFFLINE=1 python concept_naming/name_concepts_qwen.py \
  --manifest concept_naming/manifest.example.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --gpu 0 \
  --output outputs/concept_naming_qwen \
  --resume
```

For one image pair:

```bash
HF_HUB_OFFLINE=1 python concept_naming/name_concepts_qwen.py \
  --before path/to/image_A.png \
  --after path/to/image_B.png \
  --concept-id concept_000 \
  --dataset cub \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --gpu 0 \
  --output outputs/concept_naming_qwen
```

The tested inference configuration uses FP16 weights without quantization, eager attention, RGB inputs with a maximum side length of 768 pixels, deterministic greedy decoding (`do_sample=False`), no temperature, top-p, or top-k sampling, and `max_new_tokens=512`. The default configuration was validated on one NVIDIA Tesla V100-SXM2 32 GB GPU. Use `--max-image-side` or `--max-pairs` to reduce memory use when necessary.

The local output follows the existing naming format: `names.csv` and `names.md` summarize the concepts, while each checksum-named response directory stores `request_metadata.json`, `raw_output.txt`, and `result.json`. `--resume` reuses completed results, and `--dry-run` validates the image pairs without loading Qwen3-VL.

## Configuration

The example fixes GPT-5.4 snapshot `gpt-5.4-2026-03-05`, image detail `high`, reasoning effort `high`, five selected samples per dimension, and an additive perturbation of `1.0`. The manuscript defines `top_m` and the perturbation symbolically without reporting their numerical values. The example values are therefore configuration placeholders and must be replaced with the values used for the target checkpoint.

The perturbation value must match the setting used to produce the reported naming results. A negative value is accepted when the intended perturbation decreases the target dimension. The pipeline records the setting in `run_metadata.json`.

## Output

```text
outputs/concept_naming/
  concept_dictionary.csv
  concept_dictionary.json
  run_metadata.json
  pairs/
    concept_000/
      sample_01_original.png
      sample_01_perturbed.png
  responses/
    concept_000/
      result.json
      raw_response.json
      request_metadata.json
```

The dictionary contains every concept identifier, name, confidence value, definition, and supporting evidence. The image pairs remain the primary visual evidence.

## Full-run result

The following vector figure summarizes the completed naming run for all 128 concept dimensions on each benchmark.

![Semantic statistics for the complete all-dimension run](../results/semantic_statistics.svg)

## Scope

The pipeline performs inference, sample selection, concept perturbation, image generation, and post-hoc naming. It does not train NFS-CBM or download a checkpoint. The repository excludes weights by design.

The naming prompt follows Supplementary Material Section II. `appendix_prompt.txt` preserves the appendix wording. `prompt_template.txt` uses direction-neutral perturbation wording so the same code supports positive and negative changes.
