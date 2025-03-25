# PG-SAM: Prior-Guided Segment Anything Model for Medical Image Segmentation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

Segment Anything Model (SAM) demonstrates powerful zero-shot capabilities, but struggles with medical image segmentation due to domain gaps. PG-SAM addresses this through **fine-grained modality alignment** using medical text priors from LLMs and a **multi-level feature fusion decoder**, achieving state-of-the-art performance on the Synapse dataset.

![PG-SAM Architecture](figure2.pdf)

## Key Features
- 🧩 **Modular Design**  
  Componentized architecture for data processing, model training, and evaluation
- 🚀 **Unprompted Learning**  
  Integrated iterative mask optimizer for boundary refinement
- 📦 **Large File Support**  
  Git LFS management for model weights (>2GB)
- 🏥 **Medical Specialization**  
  Fine-grained alignment of radiology text reports and imaging data

## Installation
### 1. Clone Repository
```bash
git clone https://github.com/logan-0623/PG-SAM.git
cd PG-SAM
git lfs install  # Required for model weights
git lfs pull
```

### 2. Set Up Environment
```bash
conda create -n pg-sam python=3.10 -y
conda activate pg-sam
pip install -r requirements.txt
```

## Dataset Preparation
### Directory Structure
```text
PG-SAM/
├── trainset/               # Training data
│   ├── output_image_text_pairs/     # 10% subset (text+image)
│   ├── output_image_text_pairs_all_1/  # Full dataset
│   └── train_npz_new_224/          # Original 224px images
├── testset/
│   └── test_vol_h5/            # Test data
│       └── output_image_text_pairs/texts/  # Annotations
```

### Setup Steps
1. **Download Data** from [Google Drive](https://drive.google.com/drive/folders/1Wu-OjKifrVth_I5VLHK6pA7IuAo4Rp2d):
   ```bash
   gdown "1Wu-OjKifrVth_I5VLHK6pA7IuAo4Rp2d" -O ./datasets/ --folder
   ```
2. **Unzip Files**:
   ```bash
   unzip trainset.zip -d PG-SAM/trainset/
   unzip testset.zip -d PG-SAM/testset/
   ```

### Verification
```text
 Expected output:
 
 PG-SAM/
 ├── datasets/     # Dataloader code
 ├── trainset/     # Training set 
 │   ├── output_image_text_pairs/   # 10% of train data
 │   ├── output_image_text_pairs_all_1/    # full train data
 │   └── train_npz_new_224/    # the original data (without text)
 ├── testset/      # Test set 
 │   ├── test_vol_h5/   # Test images and texts
 │   │   ├── output_image_text_pairs/    
 │   │   │   └── texts/    
...
```

## Usage
### Quick Inference Demo
```bash
CUDA_VISIBLE_DEVICES="0"  python LORA.py 
```

### Training
Default configuration (single GPU):
```bash
CUDA_VISIBLE_DEVICES="0"  python train.py \
  --base_lr 0.0026 \
  --img_size 224 \ 
  --warmup \
  --AdamW \
  --max_epochs 300
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES="0" python test.py \
  --checkpoint model_weights/pg-sam-final.pt \
  --data_dir testset/test_vol_h5
```

## Configuration Options
| Parameter          | Default | Description                  |
|--------------------|---------|------------------------------|
| `--base_lr`        | 0.0026  | Base learning rate           |
| `--img_size`       | 224     | Input image resolution       |
| `--warmup`         | True    | Enable learning rate warmup  |
| `--max_epochs`     | 300     | Total training epochs        |

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
