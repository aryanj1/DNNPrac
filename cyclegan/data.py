import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def images(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob('*'))
    result = [p for p in files if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not result:
        raise ValueError(f'No images found in {path}')
    return result


def read_image(path, size, augment=False):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    side = int(size * 1.12) if augment else size
    image = ImageOps.fit(image, (side, side), method=Image.Resampling.BICUBIC)
    if augment:
        left, top = random.randint(0, side-size), random.randint(0, side-size)
        image = image.crop((left, top, left+size, top+size))
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1)


def to_image(tensor):
    array = tensor.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(((array + 1) * 127.5).round().astype(np.uint8))


class UnpairedDataset(Dataset):
    def __init__(self, root, size=128, limit=200, seed=42):
        rng = random.Random(seed)
        self.a, self.b = images(Path(root)/'trainA'), images(Path(root)/'trainB')
        rng.shuffle(self.a)
        rng.shuffle(self.b)
        if limit:
            self.a, self.b = self.a[:limit], self.b[:limit]
        self.size = size

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, index):
        return (read_image(self.a[index % len(self.a)], self.size, True),
                read_image(random.choice(self.b), self.size, True))
