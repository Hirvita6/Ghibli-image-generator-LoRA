# Ghibli Image Transformer 🎨✨

Transform your photos into magical Studio Ghibli-style illustrations using a Flux-based deep learning model with LoRA weights for efficient fine-tuning.

## 🧠 Overview

This project leverages a Flux model trained with LoRA (Low-Rank Adaptation) to convert real-world images into stylized Ghibli-like visuals. It supports fast inference, GPU acceleration, and batch processing.

## 🎨 Image Style Transformation: Original vs Ghibli

<table>
  <tr>
    <th>Original Image</th>
    <th>Ghibli Style</th>
  </tr>
  <tr>
    <td><img src="input/img1.jpg" width="300"/></td>
    <td><img src="output/ghibli_img1.png" width="300"/></td>
  </tr>
  <tr>
    <td><img src="input/img2.jpg" width="300"/></td>
    <td><img src="output/ghibli_img2.png" width="300"/></td>
  </tr>
</table>

## 📂 Project Structure

```plaintext
.
├── main.py                            # Main entry point file
├── input/                             # input directory for original images
│   └── img.jpg                        # Original Image
├── models/                            # Folder for model files (.safetensors)
│   └── Ghibli.safetensors             # Pretrained Ghibli-style model
├── output/                            # output ghibli images 
│   └── ghibli_img.png
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└──src/
│   ├── layers_cache.py        # Caches transformer layers to speed up inference
│   ├── lora_helper.py         # Applies LoRA (Low-Rank Adaptation) to the model
│   ├── pipeline.py            # Defines the full image generation pipeline
│   ├── prompt_helper.py       # Handles and formats prompts for generation
│   └── transformer_flux.py    # Core Flux-style transformer model architecture

```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hirvita6/ghibli-image-transformer.git
cd ghibli-image-transformer
```

### 2. Create a Model Directory

```bash
mkdir -p model
```

### 3. Download the Ghibli.safetensors model
```bash
wget -P model/ https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli/resolve/main/models/Ghibli.safetensors
```
or
```bash
curl -L -o model/Ghibli.safetensors https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli/resolve/main/models/Ghibli.safetensors
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Run project
```bash
python main.py
```
