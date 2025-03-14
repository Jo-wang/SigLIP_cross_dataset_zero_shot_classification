# SigLIP & SigLIPv2 Cross-Dataset Benchmark

This repository provides a standardized setup for benchmarking SigLIP and SigLIPv2 models on cross-dataset benchmark. 


## Directory Structure

```
.
├── data/                 # Directory for dataset storage
├── datasets
│   ├── __pycache__
│   ├── __init__.py
│   ├── augmix_ops.py
│   ├── caltech101.py
│   ├── cifar100c.py
│   ├── dtd.py
│   ├── eurosat.py
│   ├── fgvc.py
│   ├── food101.py
│   ├── imagenet_a.py
│   ├── imagenet_r.py
│   ├── imagenet_sketch.py
│   ├── imagenet.py
│   ├── imagenetv2.py
│   ├── oxford_flowers.py
│   ├── oxford_pets.py
│   ├── stanford_cars.py
│   ├── sun397.py
│   ├── ucf101.py
│   └── utils.py
├── data/
├── siglip_inference.py  # Main inference script
├── .gitignore

## Usage

Run inference using the provided inference script:
```bash
python siglip_inference.py --dataset fgvc --data-root /path/to/datasets --ckpt ViT-B-16-SigLIP # change ckpt to v2 if needed
```

### Arguments
- `--dataset`: Dataset name (e.g., `fgvc`, `food101`).
- `--data-root`: Path to datasets.
- `--ckpt`: Model checkpoint name.

## Output

The script evaluates the model and periodically reports accuracy during inference. Final accuracy results will be displayed upon completion of inference.

## Requirements
- Python 3.9.7
- PyTorch 2.4.0
- OpenCLIP 2.31.0
- tqdm


## Citation
If you use this benchmark or related code, please cite the corresponding SigLIP and SigLIPv2 papers.



