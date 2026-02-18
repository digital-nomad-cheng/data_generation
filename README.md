# Synthetic Face Dataset Generation with SDXL-Lightning

A GDPR-compliant synthetic face dataset generator using ByteDance's **SDXL-Lightning** model. This pipeline creates diverse, photorealistic synthetic faces for training face detection and recognition systems without privacy concerns.

## 🎯 Features

- **100% Synthetic**: No real person data - fully GDPR compliant
- **Diverse Demographics**: Balanced distribution across age, ethnicity, and gender
- **High Quality**: Photorealistic 1024x1024 images using SDXL-Lightning
- **Blazing Fast**: Only 4 inference steps thanks to distilled model (~4-8× faster than standard SDXL)
- **Open Model**: No authentication or gated access required
- **Resumable**: Automatic progress tracking and resume capability
- **Optimized for 8GB VRAM**: Memory-efficient implementation for RTX 3070 Laptop and similar
- **Metadata Rich**: JSON metadata for each image with prompts and attributes

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: Minimum 8GB (runs well on RTX 3070 Laptop and similar)
- **Disk Space**: ~7GB for model + generated images

### Software
- Python 3.10+
- CUDA 11.8+ or 12.x

## 🚀 Setup

### 1. Install Dependencies

Using **uv** (recommended):

```bash
uv sync
```

Or with pip:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python3 setup_sdxl.py
```

This will:
- Check GPU availability and VRAM
- Download SDXL base + Lightning UNet checkpoint (~6.5GB - first run only)
- Generate a test image
- Verify everything is working

**Note**: First run will take a few minutes to download the model files.

## 📖 Usage

### Single Image Generation

Generate a single face with a custom prompt:

```bash
python3 generate_single.py --prompt "A photorealistic portrait of a young Asian woman with glasses, neutral expression, professional lighting"
```

Options:
- `--prompt`: Text description of the face to generate
- `--output`: Output file path (default: auto-generated)
- `--height`: Image height in pixels (default: 1024)
- `--width`: Image width in pixels (default: 1024)
- `--steps`: Inference steps (default: 4, must match checkpoint)
- `--guidance`: Guidance scale (default: 0, SDXL-Lightning requires 0)
- `--seed`: Random seed for reproducibility

### Dataset Generation

Generate a complete dataset with balanced demographics:

```bash
# Generate 100 diverse faces
python3 generate_dataset.py --num_images 100 --output outputs/my_dataset

# Generate 1000 faces with smaller resolution for speed
python3 generate_dataset.py --num_images 1000 --output outputs/large_dataset --height 512 --width 512
```

Options:
- `--num_images`: Number of images to generate (required)
- `--output`: Output directory (default: outputs/synthetic_faces)
- `--height`: Image height (default: 1024)
- `--width`: Image width (default: 1024)
- `--steps`: Inference steps (default: 4)
- `--guidance`: Guidance scale (default: 0)
- `--no-balanced`: Disable balanced demographics (fully random)
- `--no-resume`: Start from scratch, ignore progress

**Resume Support**: If interrupted, simply run the same command again to resume from where you left off.

## 📁 Dataset Structure

```
outputs/synthetic_faces/
├── images/                    # Generated face images
│   ├── face_000000.png
│   ├── face_000001.png
│   └── ...
├── metadata/                  # JSON metadata per image
│   ├── face_000000.json      # Contains prompt, attributes, params
│   ├── face_000001.json
│   └── ...
├── progress.json             # Generation progress (for resume)
└── dataset_info.json         # Dataset statistics and info
```

### Metadata Format

Each `face_XXXXXX.json` file contains:

```json
{
  "index": 0,
  "filename": "face_000000.png",
  "prompt": "A photorealistic portrait of...",
  "attributes": {
    "age": "young adult",
    "ethnicity": "East Asian",
    "gender": "woman",
    "hair": "long hair",
    "accessory": "glasses",
    "expression": "neutral expression",
    "lighting": "studio lighting",
    "angle": "frontal view"
  },
  "generation_params": {
    "height": 1024,
    "width": 1024,
    "num_steps": 4,
    "guidance_scale": 0,
    "seed": 0
  },
  "timestamp": "2026-01-17T23:30:00",
  "model": "ByteDance/SDXL-Lightning (4-step)"
}
```

## 🎨 Customization

### Custom Prompts

You can use the `prompt_templates.py` module to create custom prompts:

```python
from prompt_templates import custom_prompt, generate_random_prompt

# Generate random diverse prompt
prompt_data = generate_random_prompt()
print(prompt_data['prompt'])

# Create specific prompt
prompt_data = custom_prompt(
    age="young adult",
    ethnicity="African",
    gender="man",
    accessory="glasses",
    expression="slight smile"
)
```

### Demographic Categories

The prompt system includes:
- **Ages**: infant, toddler, child, teenager, young adult, middle-aged adult, senior adult, elderly
- **Ethnicities**: Caucasian, African, East Asian, South Asian, Hispanic, Middle Eastern, Southeast Asian, Indigenous, Pacific Islander, mixed
- **Genders**: man, woman, person (gender-neutral)
- **Accessories**: glasses, sunglasses, hat, cap, beanie, headband, earrings, scarf
- **Expressions**: neutral, slight smile, happy, serious, confident, friendly, calm, thoughtful
- **Lighting**: natural, studio, soft, bright, dramatic, golden hour, overcast, indoor

## ⚡ Performance

On RTX 3070 Laptop (8GB VRAM) with SDXL-Lightning (4-step):
- **Single image**: ~3-8 seconds (1024x1024, 4 steps)
- **100 images**: ~5-15 minutes
- **1000 images**: ~1-2.5 hours

Tips for even faster generation:
- Use smaller resolution (512x512 or 768x768)
- Fast but lower resolution: `--height 512 --width 512`

## 🔒 GDPR Compliance & License

✅ **Fully GDPR Compliant**: All generated faces are 100% synthetic
- No real person data used or stored
- No biometric data of actual individuals
- Safe for training commercial systems
- No consent required

✅ **Open Model**: SDXL-Lightning is based on Stable Diffusion XL (CreativeML OpenRAIL-M)

## 🐛 Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. **Reduce image size**: `--height 512 --width 512`
2. **Close other applications** using GPU
3. **Restart and try again** (clears GPU memory)

### Model Download Issues

If model download fails:
- Check internet connection
- Manually download from https://huggingface.co/ByteDance/SDXL-Lightning

### Slow Generation

- First image is always slower (model loading)
- Subsequent images are faster
- Use smaller resolution for speed: `--height 512 --width 512`

## 📚 Next Steps

After generating your dataset:

1. **Inspect samples**: Check `images/` directory for quality
2. **Review statistics**: Check `dataset_info.json` for demographic balance
3. **Train face detector**: Use generated images for training
4. **Iterate**: Generate more images with different settings if needed

## 🆚 Why SDXL-Lightning?

| Feature | SDXL-Lightning | FLUX.2-klein 4B | FLUX.1-dev |
|---------|---------------|-----------------|------------|
| **Steps** | 4 (distilled) | 28 | 28 |
| **Speed** | ~3-8s/image | ~20-40s/image | ~30-60s/image |
| **VRAM** | 8GB+ (with opts) | 8GB+ (with opts) | 16GB+ |
| **Access** | Open (no login) | Gated | Gated |
| **Quality** | Excellent | Excellent | Excellent |

**SDXL-Lightning is the fastest option for batch dataset generation!**

## 📄 License

- **This code**: MIT License (free to use)
- **SDXL-Lightning model**: Based on SDXL (CreativeML OpenRAIL-M)
- **Generated images**: Usable for any purpose including commercial

## 🤝 Support

For issues or questions about:
- SDXL-Lightning: https://huggingface.co/ByteDance/SDXL-Lightning
- This pipeline: Create an issue in your repository
