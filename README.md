# PG-SAM

Segment Anything Model (SAM) demonstrates powerful zero-shot capabilities; however, its accuracy and robustness significantly decrease when applied to medical image segmentation. Recent approaches focus on modality fusion to integrate textual and image information, providing more detailed priors to address this issue. In this study, we argue that the granularity of text and the domain gap influence the accuracy of the priors. Furthermore, there exists a discrepancy between the high-level abstract semantics and the pixel-level boundary information of the image. To address this, we propose Prior-Guided SAM (PG-SAM), where the fine-grained modality prior aligner leverages specialized medical information to facilitate modality alignment. The core of this approach lies in efficiently addressing domain gap issues using fine-grained text provided by a Medical LLM, while also improving the quality of the priors produced after modality alignment. In addition, our decoder enhances the model’s learning and expressive capabilities through multi-level feature fusion and iterative mask optimizer operations, supporting unprompted learning. We also propose a unified pipeline that efficiently provides high-quality semantic information to SAM. Results on the Synapse dataset demonstrate that PG-SAM achieves state-of-the-art performance.

## Features

- **Modular Design**  
  The project is divided into modules for data processing, model construction, training, and testing, making it easy to extend and maintain.

- **Large File Management**  
  Uses Git LFS to manage model weight files that exceed GitHub's file size limit.

- **Comprehensive Dependencies**  
  All required dependencies are listed in [requirements.txt](requirements.txt) for easy environment setup.

## Directory Structure

```text
PG-SAM/
├── datasets/              # Data loading and preprocessing modules
├── model_weights/         # Model weight files (managed with Git LFS)
├── segment_anything/      # Segmentation model and related code
├── tests/                 # Test scripts and unit tests
├── train.py               # Training script
├── test.py                # Testing script
├── utils.py               # Utility functions
├── requirements.txt       # Python dependency list
└── README.md              # Project documentation 
```
### 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/logan-0623/PG-SAM.git
cd PG-SAM
```


### 2. Install the required Python packages using the requirements file:
```
conda create -n PG-SAM python=3.10
conda activate PG-SAM
pip install -r requirements.txt
```

### 3. Data Preparation

Update Soon ！！

### 4. Run the Project
Train the Model
```
CUDA_VISIBLE_DEVICES="0" python train.py --base_lr=0.0026 --img_size=224 --warmup --AdamW --max_epochs=300 --stop_epoch=300 
```
Test the Model
```
CUDA_VISIBLE_DEVICES="0" python test.py
```






