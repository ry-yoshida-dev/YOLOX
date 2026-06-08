"""YOLOX model module for inference."""

import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model for object detection inference.

    Parameters
    ----------
    backbone : YOLOPAFPN | None
        Backbone network. Defaults to ``YOLOPAFPN()``.
    head : YOLOXHead | None
        Detection head. Defaults to ``YOLOXHead(80)``.
    """

    def __init__(
        self,
        backbone: YOLOPAFPN | None = None,
        head: YOLOXHead | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Decoded detections with shape ``(batch, num_anchors, 5 + num_classes)``.
        """
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs)
