# NFS-CBM

This repository contains the inference and automatic concept-naming pipeline for **Non-negative Factorized Concept Bottleneck Models for Automatic Concept Discovery with Generative Models**.

The pipeline loads a trained NFS-CBM inference checkpoint, evaluates the test images, finds the most highly activated samples for every concept dimension, perturbs one dimension at a time, generates the corresponding reconstruction pairs, and asks GPT-5.4 to assign a semantic name. The checkpoint is intentionally excluded from Git.

## Quick start

```bash
python -m pip install -r requirements.txt
cp configs/naming_pipeline.example.json configs/naming_pipeline.json
python concept_naming/pipeline.py --config configs/naming_pipeline.json
```

Edit two local paths in `configs/naming_pipeline.json` before running:

- `checkpoint_path`: exported NFS-CBM inference checkpoint
- `dataset_root`: test images arranged in class folders

Set the API key in the shell:

```bash
export OPENAI_API_KEY="your_api_key"
```

The default virtual checkpoint path is `checkpoints/nfs_cbm_inference.pt`. Checkpoint files, datasets, generated image pairs, API responses, and concept dictionaries are ignored by Git.

See [concept_naming/README.md](concept_naming/README.md) for the checkpoint contract, complete configuration, and output files.

## Repository layout

```text
model/
  nfs_cbm.py                 NFS-CBM inference interface and FSB implementation
concept_naming/
  pipeline.py                Full all-dimension naming pipeline
  name_concepts.py           GPT-5.4 request and structured-output backend
  prompt_template.txt        Fixed execution prompt
configs/
  naming_pipeline.example.json
checkpoints/                 Local weights; excluded from Git
```

The code release does not include trained weights or benchmark datasets.

## Full-run naming result

The authors' all-dimension run is summarized in [results/semantic_statistics.pdf](results/semantic_statistics.pdf). Each dataset panel covers 128 concept units and reports the semantic-category distribution and repeated concept names.
