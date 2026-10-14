"""Build-time JIT warmup: executa os kernels uma vez para gravar o cache do
Numba na imagem (NUMBA_CACHE_DIR). Em runtime, o container ja acorda com o
codigo compilado, sem pagar compilacao no primeiro request nem depender de
CPU fora do window de request (Cloud Run so alloca CPU durante requests)."""

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "optimized")
import image  # noqa: E402

img = Image.fromarray(
    np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8), mode="RGB"
)
image.get_palette(img, 5)
image.rgb2lab(np.random.rand(10, 3))
image.lab2rgb(np.random.rand(10, 3))
print("[warmup_build] JIT cache gravado OK")
