# Pyton-Generator-PHPAiModel-Transformer

Pyton-Generator-PHPAiModel-Transformer is a Python-based toolkit for training and inferring character-level Transformer models. It includes a Flask web server for model training with GPU/CPU support, a shared model architecture, and a separate web chat interface for interactive inference. Models are exported in JSON format for compatibility with PHP runtimes like PHPAiModel-Transformer.

This project enables easy training of Transformer-based language models on text datasets and provides a seamless way to deploy them for generation tasks.

## Overview
The repository consists of three main components:
- **app.py**: Flask server for training Transformer models on character-level text datasets. Supports configurable hyperparameters and exports trained models to `.pt` (PyTorch) and `.json` (for PHP).
- **model.py**: Defines the Transformer architecture (pre-LayerNorm, GELU, Multi-Head Attention, tied weights).
- **interface.py**: Flask server for a web-based chat UI to interact with loaded `.json` models, supporting streaming generation.

Key aspects:
- Character-level tokenization for simplicity.
- Causal self-attention for autoregressive text generation.
- Training with AMP (Automatic Mixed Precision) for faster GPU performance.
- Inference with temperature and top-k sampling.

## Features
- Web-based training interface with real-time progress updates via SSE (Server-Sent Events).
- Configurable model dimensions: d_model, heads, layers, d_ff, max_seq.
- Automatic dataset splitting for train/validation.
- Export to JSON weights compatible with PHP Transformer runtimes.
- Streaming chat inference with token-by-token output.
- CUDA support if available; falls back to CPU.
- MIT licensed, open-source.

## Requirements
- Python 3.8+ (tested with 3.10+ for Torch compatibility).
- PyTorch 2.0+ (with CUDA for GPU acceleration).
- Flask for web servers.
- Recommended: NVIDIA GPU with CUDA for training large models.
- Directories: `Datasets/` for .txt input files, `Models/` for output .pt/.json.