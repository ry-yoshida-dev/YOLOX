"""YOLOX detection head for inference."""

import math

import torch
import torch.nn as nn

from yolox.utils import meshgrid

from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    """
    YOLOX detection head.

    Parameters
    ----------
    num_classes : int
        Number of object classes.
    width : float
        Width multiplier for channel dimensions.
    strides : list[int]
        Feature-map strides for each detection scale.
    in_channels : list[int]
        Input channel sizes for each detection scale.
    act : str
        Activation function name.
    depthwise : bool
        Whether to use depthwise convolutions in the head.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] | None = None,
        in_channels: list[int] | None = None,
        act: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()

        if strides is None:
            strides = [8, 16, 32]
        if in_channels is None:
            in_channels = [256, 512, 1024]

        self.num_classes = num_classes
        self.decode_in_inference = True
        self.strides = strides
        self.hw: list[tuple[int, int]] = []

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        conv_cls = DWConv if depthwise else BaseConv

        for channel_count in in_channels:
            self.stems.append(
                BaseConv(
                    in_channels=int(channel_count * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    conv_cls(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    conv_cls(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    conv_cls(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    conv_cls(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def initialize_biases(self, prior_prob: float) -> None:
        """
        Initialize classification and objectness biases.

        Parameters
        ----------
        prior_prob : float
            Prior probability used for bias initialization.
        """
        for conv in self.cls_preds:
            bias = conv.bias.view(1, -1)
            bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            bias = conv.bias.view(1, -1)
            bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def forward(self, xin: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Decode multi-scale predictions into bounding boxes.

        Parameters
        ----------
        xin : tuple[torch.Tensor, ...]
            Feature maps from the PA-FPN backbone.

        Returns
        -------
        torch.Tensor
            Decoded predictions with shape ``(batch, num_anchors, 5 + num_classes)``.
        """
        outputs: list[torch.Tensor] = []

        for index, (cls_conv, reg_conv, _, feature_map) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            feature_map = self.stems[index](feature_map)
            cls_feat = cls_conv(feature_map)
            cls_output = self.cls_preds[index](cls_feat)
            reg_feat = reg_conv(feature_map)
            reg_output = self.reg_preds[index](reg_feat)
            obj_output = self.obj_preds[index](reg_feat)
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()],
                1,
            )
            outputs.append(output)

        self.hw = [output.shape[-2:] for output in outputs]
        outputs_tensor = torch.cat(
            [output.flatten(start_dim=2) for output in outputs],
            dim=2,
        ).permute(0, 2, 1)

        if self.decode_in_inference:
            return self.decode_outputs(outputs_tensor, dtype=xin[0].type())
        return outputs_tensor

    def decode_outputs(self, outputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Decode raw head outputs into absolute box coordinates.

        Parameters
        ----------
        outputs : torch.Tensor
            Raw head outputs before grid decoding.
        dtype : torch.dtype
            Target dtype for grid tensors.

        Returns
        -------
        torch.Tensor
            Decoded predictions in center-size format.
        """
        grids: list[torch.Tensor] = []
        strides_tensors: list[torch.Tensor] = []

        for (height, width), stride in zip(self.hw, self.strides):
            y_values, x_values = meshgrid(
                [torch.arange(height), torch.arange(width)]
            )
            grid = torch.stack((x_values, y_values), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides_tensors.append(torch.full((*shape, 1), stride))

        grids_tensor = torch.cat(grids, dim=1).type(dtype)
        strides_tensor = torch.cat(strides_tensors, dim=1).type(dtype)

        return torch.cat(
            [
                (outputs[..., 0:2] + grids_tensor) * strides_tensor,
                torch.exp(outputs[..., 2:4]) * strides_tensor,
                outputs[..., 4:],
            ],
            dim=-1,
        )
