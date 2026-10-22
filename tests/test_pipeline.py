import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cyclegan.data import UnpairedDataset, read_image
from cyclegan.models import Generator, Discriminator
from cyclegan.__main__ import ReplayPool


class PipelineTests(unittest.TestCase):
    def test_cycle_backpropagates_through_both_generators(self):
        a, b = Generator(8, 1), Generator(8, 1)
        x = torch.randn(1, 3, 32, 32)
        translated = a(x)
        self.assertEqual(translated.shape, x.shape)
        loss = (b(translated)-x).abs().mean()
        loss.backward()
        for model in (a, b):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        logits = Discriminator(8)(translated.detach())
        self.assertEqual(logits.shape[1], 1)

    def test_dataset_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            for split in ('trainA', 'trainB'):
                directory = Path(folder)/split
                directory.mkdir()
                for i in range(3):
                    Image.new('L', (45, 60), 128).save(directory/f'{i}.png')
            dataset = UnpairedDataset(folder, 32, 2)
            self.assertEqual(len(dataset), 2)
            a, b = dataset[0]
            self.assertEqual(a.shape, (3, 32, 32))
            self.assertTrue(a.min() >= -1 and b.max() <= 1)
            self.assertTrue(torch.equal(read_image(dataset.a[0], 32), read_image(dataset.a[0], 32)))

    def test_pool_detaches_and_bounds_memory(self):
        pool = ReplayPool(2)
        for _ in range(5):
            output = pool.query(torch.randn(1, 3, 32, 32, requires_grad=True))
            self.assertFalse(output.requires_grad)
        self.assertEqual(len(pool.items), 2)


if __name__ == '__main__':
    unittest.main()
