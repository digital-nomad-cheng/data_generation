#!/usr/bin/env python3
"""
Batch dataset generation script for creating synthetic face datasets
using ByteDance/SDXL-Lightning.
"""

import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import signal
import sys

from prompt_templates import generate_balanced_prompts, generate_random_prompt

MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REPO = "ByteDance/SDXL-Lightning"
MODEL_CKPT = "sdxl_lightning_4step_unet.safetensors"
NUM_STEPS = 4  # Must match checkpoint (4-step)
GUIDANCE_SCALE = 0  # CFG-free distillation requires 0
MODEL_NAME = "ByteDance/SDXL-Lightning (4-step)"


class DatasetGenerator:
    """Generator for synthetic face datasets using SDXL-Lightning."""

    def __init__(self, output_dir: str, cache_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.pipe = None
        self.interrupted = False

        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)

        # Create directory structure
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Load or create progress file
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()

    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully."""
        print("\n\n⚠️  Interruption detected. Saving progress...")
        self.interrupted = True

    def _load_progress(self):
        """Load progress from previous run if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"generated_count": 0, "last_index": -1}

    def _save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def load_pipeline(self):
        """Load SDXL-Lightning pipeline with optimizations for 8GB VRAM."""
        if self.pipe is not None:
            return

        print("Loading SDXL-Lightning (4-step) model...")

        cache_path = self.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Load distilled UNet checkpoint
        print("Downloading/loading Lightning UNet checkpoint...")
        unet = UNet2DConditionModel.from_config(
            MODEL_BASE, subfolder="unet"
        )
        ckpt_path = hf_hub_download(
            MODEL_REPO, MODEL_CKPT, cache_dir=str(cache_path)
        )
        unet.load_state_dict(load_file(ckpt_path, device="cpu"))
        unet = unet.to(dtype=torch.float16)  # Must cast before pipeline wraps it

        # Load SDXL base pipeline with Lightning UNet
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_BASE,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=str(cache_path),
        )

        # Configure scheduler for Lightning (trailing timesteps required)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

        # Apply optimizations
        if gpu_memory < 12:
            print(f"GPU VRAM: {gpu_memory:.1f}GB - Applying memory optimizations...")
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_attention_slicing(1)
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        else:
            print(f"GPU VRAM: {gpu_memory:.1f}GB - Loading to GPU...")
            self.pipe = self.pipe.to("cuda")

        print("✅ Model loaded!\n")

    def generate_image(
        self,
        prompt: str,
        index: int,
        attributes: dict,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = NUM_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        seed: int = None
    ):
        """Generate a single image and save with metadata."""

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        else:
            # Use index-based seed for reproducibility
            generator = torch.Generator("cpu").manual_seed(index)

        # Generate image
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save image
        image_filename = f"face_{index:06d}.png"
        image_path = self.images_dir / image_filename
        image.save(image_path)

        # Save metadata
        metadata = {
            "index": index,
            "filename": image_filename,
            "prompt": prompt,
            "attributes": attributes,
            "generation_params": {
                "height": height,
                "width": width,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "seed": seed if seed is not None else index
            },
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME
        }

        metadata_path = self.metadata_dir / f"face_{index:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return image_path

    def generate_dataset(
        self,
        num_images: int,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = NUM_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        balanced: bool = True,
        resume: bool = True
    ):
        """
        Generate a complete dataset of synthetic faces.

        Args:
            num_images: Total number of images to generate
            height: Image height in pixels
            width: Image width in pixels
            num_steps: Number of inference steps (must match checkpoint)
            guidance_scale: Guidance scale (SDXL-Lightning requires 0)
            balanced: Use balanced demographic distribution
            resume: Resume from previous progress if exists
        """

        # Load pipeline
        self.load_pipeline()

        # Determine starting point
        start_idx = 0
        if resume and self.progress["generated_count"] > 0:
            start_idx = self.progress["last_index"] + 1
            print(f"📁 Resuming from image {start_idx}")
            print(f"   Already generated: {self.progress['generated_count']} images\n")

        # Generate prompts
        print(f"🎨 Generating {num_images} prompts...")
        if balanced:
            prompt_data = generate_balanced_prompts(num_images)
        else:
            prompt_data = [generate_random_prompt() for _ in range(num_images)]
        print(f"✅ Prompts ready!\n")

        # Generate images
        print(f"🖼️  Generating {num_images - start_idx} face images...")
        print(f"   Output: {self.output_dir}\n")

        # Create progress bar
        pbar = tqdm(
            range(start_idx, num_images),
            initial=start_idx,
            total=num_images,
            desc="Generating faces",
            unit="img"
        )

        try:
            for i in pbar:
                if self.interrupted:
                    print("\n⚠️  Generation interrupted by user.")
                    break

                prompt_info = prompt_data[i]
                prompt = prompt_info["prompt"]
                attributes = prompt_info["attributes"]

                # Update progress bar with current demographic
                pbar.set_postfix({
                    'age': attributes.get('age', 'N/A')[:10],
                    'ethnicity': attributes.get('ethnicity', 'N/A')[:10]
                })

                # Generate image
                self.generate_image(
                    prompt=prompt,
                    index=i,
                    attributes=attributes,
                    height=height,
                    width=width,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale
                )

                # Update progress
                self.progress["generated_count"] = i + 1
                self.progress["last_index"] = i

                # Save progress every 10 images
                if (i + 1) % 10 == 0:
                    self._save_progress()

        finally:
            # Save final progress
            self._save_progress()
            pbar.close()

        # Generate dataset info
        self._generate_dataset_info(num_images)

        print(f"\n✅ Dataset generation complete!")
        print(f"   Total images: {self.progress['generated_count']}")
        print(f"   Location: {self.output_dir}")

    def _generate_dataset_info(self, total_images: int):
        """Generate dataset information file."""

        # Collect statistics from metadata
        ages = {}
        ethnicities = {}
        genders = {}

        for i in range(self.progress["generated_count"]):
            metadata_path = self.metadata_dir / f"face_{i:06d}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    attrs = data.get("attributes", {})

                    age = attrs.get("age", "unknown")
                    ethnicity = attrs.get("ethnicity", "unknown")
                    gender = attrs.get("gender", "unknown")

                    ages[age] = ages.get(age, 0) + 1
                    ethnicities[ethnicity] = ethnicities.get(ethnicity, 0) + 1
                    genders[gender] = genders.get(gender, 0) + 1

        # Create dataset info
        dataset_info = {
            "name": self.output_dir.name,
            "total_images": self.progress["generated_count"],
            "target_images": total_images,
            "creation_date": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "statistics": {
                "ages": ages,
                "ethnicities": ethnicities,
                "genders": genders
            },
            "directory_structure": {
                "images": "images/",
                "metadata": "metadata/",
                "progress": "progress.json"
            }
        }

        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\n📊 Dataset Statistics:")
        print(f"   Total images: {dataset_info['total_images']}")
        print(f"   Genders: {genders}")
        print(f"   Ethnicities: {len(ethnicities)} categories")
        print(f"   Age groups: {len(ages)} categories")

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic face dataset using SDXL-Lightning"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/synthetic_faces",
        help="Output directory for dataset (default: outputs/synthetic_faces)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
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
        "--no-balanced",
        action="store_true",
        help="Disable balanced demographic distribution (fully random)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, ignore previous progress"
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        sys.exit(1)

    print("=" * 80)
    print("SDXL-Lightning Synthetic Face Dataset Generator")
    print("=" * 80)
    print(f"Target images: {args.num_images}")
    print(f"Output directory: {args.output}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Inference steps: {args.steps}")
    print(f"Balanced distribution: {not args.no_balanced}")
    print("=" * 80)
    print()

    try:
        generator = DatasetGenerator(
            output_dir=args.output,
            cache_dir="models"
        )

        generator.generate_dataset(
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            balanced=not args.no_balanced,
            resume=not args.no_resume
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Progress has been saved.")
        print("   Run the same command again to resume.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
