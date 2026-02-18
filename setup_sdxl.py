#!/usr/bin/env python3
"""
Setup and verification script for ByteDance/SDXL-Lightning model.
Checks GPU availability, loads the model, and generates a test image.
"""

import torch
import sys
from pathlib import Path


def check_gpu():
    """Check GPU availability and specs."""
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available. SDXL-Lightning requires GPU.")
        print("   Please ensure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print("   - PyTorch installed with CUDA support")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    print(f"✅ GPU Found: {gpu_name}")
    print(f"✅ Total VRAM: {gpu_memory:.2f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")

    if gpu_memory < 6:
        print("\n⚠️  WARNING: Less than 6GB VRAM detected.")
        print("   SDXL-Lightning may not run on this GPU.")
        return False
    elif gpu_memory < 12:
        print(f"\n⚠️  Note: {gpu_memory:.0f}GB VRAM detected.")
        print("   Will use memory optimizations:")
        print("   - float16 precision")
        print("   - CPU offloading")
        print("   - VAE slicing & tiling")
    else:
        print("\n✅ Sufficient VRAM for optimal operation.")

    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)

    required = {
        'diffusers': '0.25.0',
        'transformers': '4.40.0',
        'accelerate': '0.30.0',
        'safetensors': '0.4.0',
        'huggingface_hub': '0.20.0',
    }

    all_ok = True
    for package, min_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            all_ok = False

    if not all_ok:
        print("\n❌ Missing dependencies. Please run:")
        print("   uv sync")
        return False

    return True


def test_model_loading():
    """Attempt to load SDXL-Lightning model and generate a test image."""
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)

    try:
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
        MODEL_REPO = "ByteDance/SDXL-Lightning"
        MODEL_CKPT = "sdxl_lightning_4step_unet.safetensors"

        cache_dir = str(Path(__file__).parent / "models")

        print("Loading SDXL-Lightning (4-step) model...")
        print("(First run will download ~6.5GB of model weights)")
        print("This may take several minutes...\n")

        # Determine optimization strategy based on VRAM
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Load distilled UNet in float16
        print("Loading UNet checkpoint...")
        unet = UNet2DConditionModel.from_config(
            MODEL_BASE, subfolder="unet"
        )
        ckpt_path = hf_hub_download(
            MODEL_REPO, MODEL_CKPT, cache_dir=cache_dir
        )
        unet.load_state_dict(load_file(ckpt_path, device="cpu"))
        unet = unet.to(dtype=torch.float16)  # Must cast before pipeline wraps it

        # Load SDXL base with Lightning UNet
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_BASE,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir,
        )

        # Configure scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        # Apply memory optimizations
        if gpu_memory < 12:
            print("Applying memory optimizations for 8GB VRAM...")
            pipe.enable_sequential_cpu_offload()  # Offloads layer-by-layer, lowest peak VRAM
            pipe.enable_attention_slicing(1)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe = pipe.to("cuda")

        print("✅ Model loaded successfully!")

        # Generate a test image
        print("\nGenerating test image (512x512, 4 steps)...")
        prompt = "A photorealistic portrait of a young woman with a neutral expression, professional headshot, studio lighting"

        image = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0,
        ).images[0]

        # Save test image
        output_path = Path(__file__).parent / "test_output.png"
        image.save(output_path)

        print(f"✅ Test image generated successfully!")
        print(f"   Saved to: {output_path}")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def main():
    """Run all setup checks."""
    print("\n🚀 SDXL-Lightning Setup & Verification")
    print("=" * 60)

    # Check GPU
    if not check_gpu():
        print("\n❌ GPU check failed. Cannot proceed.")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install requirements.")
        sys.exit(1)

    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading failed. Please check the errors above.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 60)
    print("\nYou're ready to generate synthetic faces!")
    print("Next steps:")
    print("  - Single image: python generate_single.py --prompt 'your prompt'")
    print("  - Dataset: python generate_dataset.py --num_images 100")
    print("=" * 60)


if __name__ == "__main__":
    main()
