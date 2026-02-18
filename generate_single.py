#!/usr/bin/env python3
"""
Single face generation script using ByteDance/SDXL-Lightning.
Generates one face image from a text prompt.
"""

import torch
import argparse
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REPO = "ByteDance/SDXL-Lightning"
MODEL_CKPT = "sdxl_lightning_4step_unet.safetensors"
NUM_STEPS = 4  # Must match checkpoint (4-step)
GUIDANCE_SCALE = 0  # CFG-free distillation requires 0


def load_pipeline(cache_dir="models"):
    """Load SDXL-Lightning pipeline with memory optimizations for 8GB VRAM."""
    print("Loading SDXL-Lightning (4-step) model...")

    cache_path = Path(__file__).parent / cache_dir
    cache_path.mkdir(exist_ok=True)
    torch.cuda.empty_cache()

    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Load distilled UNet checkpoint
    print("Downloading/loading Lightning UNet checkpoint...")
    unet_config = UNet2DConditionModel.from_config(
        MODEL_BASE, subfolder="unet"
    )
    ckpt_path = hf_hub_download(
        MODEL_REPO, MODEL_CKPT, cache_dir=str(cache_path)
    )
    unet_config.load_state_dict(load_file(ckpt_path, device="cpu"))
    unet_config = unet_config.to(dtype=torch.float16)  # Must cast before pipeline wraps it

    # Load SDXL base pipeline with Lightning UNet
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_BASE,
        unet=unet_config,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=str(cache_path),
    )

    # Configure scheduler for Lightning (trailing timesteps required)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Apply memory optimizations for 8GB VRAM (RTX 3070 Laptop)
    if gpu_memory < 12:
        print(f"GPU VRAM: {gpu_memory:.1f}GB - Applying memory optimizations...")
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing(1)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        print(f"GPU VRAM: {gpu_memory:.1f}GB - Loading to GPU...")
        pipe = pipe.to("cuda")

    torch.cuda.empty_cache()

    print("✅ Model loaded successfully!\n")
    return pipe


def generate_face(
    pipe,
    prompt,
    output_path=None,
    height=1024,
    width=1024,
    num_steps=NUM_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    seed=None
):
    """Generate a single face image."""

    # Set seed for reproducibility if provided
    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)
        print(f"Using seed: {seed}")

    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}")
    print(f"Steps: {num_steps}")
    print(f"Generating image...\n")

    # Clear cache before generation
    torch.cuda.empty_cache()

    # Generate image
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Save image
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"output_{timestamp}.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    # Save metadata
    metadata = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "model": "ByteDance/SDXL-Lightning (4-step)"
    }

    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Image saved to: {output_path}")
    print(f"✅ Metadata saved to: {metadata_path}")

    return image, output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a single synthetic face using SDXL-Lightning")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic portrait of a person with neutral expression, professional headshot, natural lighting",
        help="Text prompt describing the face to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_STEPS,
        help=f"Number of inference steps (default: {NUM_STEPS}, must match checkpoint)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=GUIDANCE_SCALE,
        help=f"Guidance scale (default: {GUIDANCE_SCALE}, SDXL-Lightning requires 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        return

    try:
        # Load pipeline
        pipe = load_pipeline()

        # Generate face
        generate_face(
            pipe,
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )

        print("\n✅ Generation complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
