import os
import torch
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import torchvision.transforms as T

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

from huggingface_hub import login

# Paste your token here (you can generate one at https://huggingface.co/settings/tokens)
login("HuggingFace_ACCESS_Token")

# Paths (auto-downloads model from Hugging Face)
base_path = "black-forest-labs/FLUX.1-dev"
lora_path = "./models/Ghibli.safetensors"  # <-- Upload this manually to Kaggle!

# Load FLUX model
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
    base_path, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)


# Prompt (Ghibli trigger words)
prompt = "Ghibli Studio style, Charming hand-drawn anime-style illustration"

# Safe settings 
height = 512
width = 512
steps = 25
guidance_scale = 3.5
seed = 42

# Upload Image in Notebook Cell
from pathlib import Path
upload_path = Path("input/img1.jpg")

# Load user image
input_img = Image.open(upload_path).convert("RGB")
input_img = input_img.resize((width, height))

# Run pipeline
generator = torch.Generator(device="cuda").manual_seed(seed)
result = pipe(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=steps,
    max_sequence_length=512,
    generator=generator,
    spatial_images=[input_img],  # 👈 required for image-to-image!
    cond_size=512,
).images[0]

# Show and Save
output_path = "output/ghibli_img1.png"
result.save(output_path)