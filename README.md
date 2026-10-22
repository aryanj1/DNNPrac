# CycleGAN on your Mac GPU

A runnable PyTorch implementation of unpaired **apple ↔ orange** translation. Includes two ResNet generators, two 70×70 PatchGAN discriminators, least-squares adversarial loss, cycle consistency (weight 10), identity loss (weight 5), and 50-image replay pools. No pretrained weights are required.

## Run here

The local `.venv` and dataset were prepared for this project. From this folder:

```bash
source .venv/bin/activate
bash run.sh
```

This trains for 10 epochs on the Mac GPU, then translates 20 held-out images in each direction. Default training is 200 randomly selected images per domain, 128×128 pixels, batch size 1, 32 base channels, and six residual blocks. It is a small educational experiment, not a guarantee of photorealistic translation. Expect early outputs to look poor; judge changes across epoch previews. Even this small run may take significant time depending on your Mac.

## Setup on another Mac

Use native ARM64 Python 3.12 (not the macOS system Python 3.9 or a Rosetta Python):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

PyTorch 2.5.1 is pinned as the project's tested baseline. Training and inference default to `--device mps` and fail clearly if the Mac GPU is unavailable. `--device auto` permits CPU fallback; `--device cpu` explicitly uses CPU. Float32 and zero DataLoader workers keep this setup simple on macOS. No CUDA installation is needed. See [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Dataset

The downloader uses the [original CycleGAN authors' dataset host](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip), following their [dataset script](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh). The archive is roughly 75 MB; the complete dataset is downloaded once, while training uses a deterministic 200-image subset per domain by default. Actual archive counts: 995 train apples, 1,019 train oranges, 266 test apples and 248 test oranges. Test images are never used for gradient updates. A local SHA-256 and counts are recorded in `data/dataset.json` for traceability; the hash is not an independently supplied authenticity signature.

Images were collected from ImageNet by the original authors. Refer to the [original project](https://github.com/junyanz/CycleGAN#datasets) and underlying image terms before redistribution or commercial use.

```text
data/apple2orange/
  trainA/   # apples
  trainB/   # oranges
  testA/    # held-out apples
  testB/    # held-out oranges
```

You can use your own images with the same four-folder layout and `--data path/to/dataset`. Training samples domain B independently; paired images are not required. Images are converted to RGB, center-fit/cropped to squares, and normalized to [-1,1]. Training additionally applies random crops and flips. Inference produces square images at the training resolution by default.

## Train, stop, resume

```bash
python -m cyclegan train
python -m cyclegan train --resume runs/apple2orange/latest.pt
```

Each completed epoch saves an atomic `latest.pt` checkpoint, `losses.csv`, and `epoch_001.png` previews. Preview rows show an apple and an orange; columns show original, translation, and cycle reconstruction. Ctrl+C stops training; resume restarts after the last completed epoch (unfinished-epoch work is lost). Checkpoints contain both generators/discriminators, Adam states, replay pools and Python/CPU random states. MPS operations can be nondeterministic; resume is not promised to be bit-identical. Resume restores the saved architecture, subset and schedule unless --epochs or --decay-start explicitly overrides the schedule. --epochs is the total target, not additional epochs. The run.sh shortcut caps the total at 10; a checkpoint already at or above that count proceeds directly to inference. Set `--run` to the original run directory when resuming a custom run.

The default learning rate is 0.0002 through epoch 6, then decreases linearly over epochs 7–10. A run with an existing checkpoint cannot be accidentally restarted without `--resume`. Checkpoints are loaded with `weights_only=True`.

For a quick pipeline check (these weights will not produce meaningful translations):

```bash
python -m cyclegan train --run runs/quickcheck --epochs 2 --decay-start 1 --steps 2 --limit 4
python -m cyclegan infer --checkpoint runs/quickcheck/latest.pt --limit 2 --output outputs/quickcheck
```

For a longer experiment with the entire training set:

```bash
python -m cyclegan train --run runs/longer --limit 0 --epochs 100 --decay-start 50
```

Use a new run directory for different settings. To reduce memory or time further, try `--size 64 --channels 16 --blocks 3` with a new run. Resolution must be at least 32 and divisible by 4. Larger datasets/resolutions improve the opportunity to learn, but cost more computation; GAN losses alone do not establish visual quality.

## Test / translate your images

```bash
# Apples → oranges, held-out test set (20 images by default)
python -m cyclegan infer

# Oranges → apples
python -m cyclegan infer --direction ba --input data/apple2orange/testB --output outputs/orange2apple

# Your own file or directory
python -m cyclegan infer --input /path/to/apple.jpg --output outputs/custom

# All held-out apples
python -m cyclegan infer --limit 0 --output outputs/all_apples
```

Results include individual translated PNGs, a labeled `comparison.png` (up to eight rows), and `metrics.json`. Cycle MAE is reconstruction error on [0,1] pixels, **not** translation accuracy or realism. There is no paired target ground truth here; inspect the translated images. Both inference and reconstruction use the Mac GPU.

`bash run.sh` is the default end-to-end shortcut. For custom run directories, use the separate train/infer commands and pass the matching `--checkpoint`.

## Checks and implementation

```bash
python -m unittest discover -s tests -v
python -m cyclegan --help
python -m cyclegan train --help
```

`cyclegan/models.py` defines networks; `data.py` handles transforms/unpaired sampling; `__main__.py` contains training, checkpoints and inference. Dependencies are listed in `requirements.txt`. Training uses batch size one and no mixed precision. The replay pools live on CPU to reduce GPU memory pressure.

Reference: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/), Zhu et al., ICCV 2017. This is an independent compact implementation, not the authors' pretrained model.
# DNNPrac
