# Lightweight Multilingual Translation for Edge Devices

Research project focused on compressing multilingual machine translation models for edge-friendly deployment without giving up most of the original quality.

This work explores German, French, and Italian to English translation using LoRA adapters, multilingual knowledge distillation, quantization, and pruning. The project goal was to retain strong translation quality while making the final model substantially smaller and cheaper to run.

## Highlights

- Reduced trainable parameters by about 95% through LoRA-based adaptation.
- Distilled multiple language-specific teacher models into a single multilingual student model.
- Applied quantization and pruning to further optimize the final model for edge deployment.
- Targeted roughly 90% of base-model performance with about 60% model-size reduction.

## Technical Approach

### 1. LoRA Adapters
The project starts from MarianMT and adapts it for additional language pairs using low-rank updates instead of full fine-tuning.

### 2. Multilingual Distillation
Language-specific teachers are used to train a shared student model so one smaller model can serve multiple translation directions.

### 3. Compression for Deployment
The workflow includes ONNX export, task-aware quantization, and optional pruning to improve deployment efficiency on resource-constrained devices.

## Repository Structure

- `train_lora.py`: trains language-specific LoRA adapters
- `lora_adapter.py`: LoRA adapter implementation for MarianMT
- `distillation.py`: multilingual distillation logic
- `train_student.py`: student-model training with knowledge distillation
- `quantization.py`: quantization, pruning, and edge optimization workflow
- `evaluate.py`: BLEU and inference-oriented evaluation helpers
- `download_opus.py` and `download_opus100.py`: data acquisition scripts
- `Experiments/`: exploratory notebooks and earlier experiments

## Workflow

1. Download and preprocess multilingual parallel data.
2. Train or load LoRA adapters for each language pair.
3. Distill those teacher models into a shared student model.
4. Quantize and optionally prune the distilled model.
5. Evaluate translation quality and model size tradeoffs.

## Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- MarianMT
- ONNX / model quantization
- sacreBLEU

## Why This Project Matters

Most translation systems assume server-scale compute. This project focuses on a more practical constraint: how to make multilingual translation lightweight enough for edge and low-resource environments while keeping the model useful.

## Notes

This repository is research-oriented and centers on experimentation, model compression, and evaluation rather than production packaging.
