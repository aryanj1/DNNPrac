"""Download the authors' apple2orange archive, safely extract and verify it."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip'
ROOT = Path(__file__).resolve().parents[1] / 'data'


def main():
    ROOT.mkdir(exist_ok=True)
    archive = ROOT / 'apple2orange.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.zip.part')
        print(f'Downloading {URL}', flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (ROOT/member.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe archive path')
            if member.file_size > 100_000_000:
                raise ValueError('Unexpectedly large archive member')
        zipped.extractall(ROOT)
    counts = {}
    for split in ('trainA', 'trainB', 'testA', 'testB'):
        counts[split] = len(list((ROOT/'apple2orange'/split).glob('*.jpg')))
        if counts[split] == 0:
            raise ValueError(f'Missing images: {split}')
    metadata = {'source': URL, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'counts': counts, 'domains': {'A': 'apple', 'B': 'orange'}}
    (ROOT/'dataset.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
