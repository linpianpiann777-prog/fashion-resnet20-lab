"""Offline, checksum-validated IDX loading. Never downloads implicitly."""
from pathlib import Path
import gzip
import hashlib
import struct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'FashionMNIST' / 'raw'
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
CLASSES_ZH = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
MD5 = {'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
       'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
       't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
       't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310'}


def load_part(train=True):
    prefix = 'train' if train else 't10k'
    arrays = []
    for kind in ('images-idx3', 'labels-idx1'):
        name = f'{prefix}-{kind}-ubyte.gz'
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f'请按 README 下载数据文件: {path}')
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != MD5[name]:
            raise ValueError(f'MD5 mismatch: {name}')
        content = gzip.decompress(content)
        if kind.startswith('images'):
            magic, n, h, w = struct.unpack('>IIII', content[:16])
            assert magic == 2051 and (h, w) == (28, 28)
            arrays.append(np.frombuffer(content, np.uint8, offset=16).reshape(n, h, w).copy())
        else:
            magic, n = struct.unpack('>II', content[:8])
            assert magic == 2049
            arrays.append(np.frombuffer(content, np.uint8, offset=8).copy())
        assert len(arrays[-1]) == n
    return arrays


def split_indices(labels, seed=2026):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        val.extend(idx[:600]); train.extend(idx[600:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def tensor_images(images, device):
    return torch.as_tensor(images, device=device).unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5)
