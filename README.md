# YOLOX (Inference Fork)

## Overview

Inference-only fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). Training, evaluation, export tools, and COCO evaluation extensions were removed.

Install in editable mode from this directory:

```bash
pip install -e .
```

## Components

| Component | Description |
|-----------|-------------|
| [src/yolox/models](./src/yolox/models/) | YOLOX backbone, PA-FPN, and detection head modules |
| [src/yolox/utils](./src/yolox/utils/) | Box decoding helpers and PyTorch compatibility utilities |

## Examples

```python
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

backbone = YOLOPAFPN(depth=1.33, width=1.25)
head = YOLOXHead(num_classes=1, width=1.25)
model = YOLOX(backbone=backbone, head=head)
```
