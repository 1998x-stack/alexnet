# -*- coding: utf-8 -*-
"""AlexNet (paper-faithful options) with LRN/overlapping pooling/dropout."""
from __future__ import annotations
from typing import Tuple, Optional
import torch
from torch import nn

from .registry import register_model


class LRN(nn.LocalResponseNorm):
    """封装，便于通过参数在配置中开/关 LRN."""
    pass


class AlexNet(nn.Module):
    """AlexNet 实现（支持消融：LRN、重叠池化、Dropout 等）.

    Attributes:
        features: 卷积与池化特征提取模块
        classifier: 全连接分类头
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_ch: int = 3,
        use_lrn: bool = True,
        overlap_pool: bool = True,
        dropout_p: float = 0.5,
        init_bias_ones: bool = True,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        k_pool = 3 if overlap_pool else 2  # 重叠池化 kernel=3,stride=2；非重叠则 2,2
        s_pool = 2

        def lrn() -> nn.Module:
            return LRN(size=5, alpha=1e-4, beta=0.75, k=2.0) if use_lrn else nn.Identity()

        # 论文中 conv1: 11x11 s=4, pad=2；本实现统一为 224 输入
        c1 = int(96 * width_mult)
        c2 = int(256 * width_mult)
        c3 = int(384 * width_mult)
        c4 = int(384 * width_mult)
        c5 = int(256 * width_mult)

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            lrn(),
            nn.MaxPool2d(kernel_size=k_pool, stride=s_pool),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            lrn(),
            nn.MaxPool2d(kernel_size=k_pool, stride=s_pool),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=k_pool, stride=s_pool),
        )

        # 论文 fc: 4096-4096-1000；输入 6x6x256（224 输入下）
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._init_weights(init_bias_ones=init_bias_ones)

    def _init_weights(self, init_bias_ones: bool = True) -> None:
        """按论文初始化：W~N(0,0.01)；conv2/4/5+fc隐藏层 bias=1，其余=0."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if isinstance(m, nn.Linear):
                    # fc 隐藏层 bias=1
                    if init_bias_ones and ("classifier.1" in name or "classifier.4" in name):
                        nn.init.constant_(m.bias, 1.0)
                    else:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Conv2d):
                    # conv2/4/5 bias=1，其余=0
                    if init_bias_ones and any(layer in name for layer in ["features.4", "features.10", "features.12"]):
                        nn.init.constant_(m.bias, 1.0)
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_model("alexnet")
def build_alexnet(**kwargs) -> AlexNet:
    """注册的构造函数，支持配置字典传参."""
    return AlexNet(**kwargs)
