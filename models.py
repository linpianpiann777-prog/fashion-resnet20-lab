"""CIFAR-style ResNet-20, adapted to 1-channel 28x28 Fashion-MNIST.

Original basic block and parameter-free option-A shortcuts; no pretrained weights.
DW replaces only the residual branch's 3x3 convolutions, not the input stem.
"""
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = ('baseline', 'dw', 'se', 'dw_se')


class SeparableConv(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                         nn.Conv2d(cin, cout, 1, bias=False))


class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(channels, max(channels // reduction, 1), 1),
                                nn.ReLU(),
                                nn.Conv2d(max(channels // reduction, 1), channels, 1),
                                nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3), keepdim=True))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, depthwise=False, attention=False):
        super().__init__()
        conv = SeparableConv if depthwise else lambda a, b, s: nn.Conv2d(a, b, 3, s, 1, bias=False)
        self.conv1, self.bn1 = conv(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = conv(cout, cout, 1), nn.BatchNorm2d(cout)
        self.attention = SE(cout) if attention else nn.Identity()
        self.stride, self.extra = stride, cout - cin

    def forward(self, x):
        shortcut = x[:, :, ::self.stride, ::self.stride]
        if self.extra:
            shortcut = F.pad(shortcut, (0, 0, 0, 0, self.extra // 2, self.extra - self.extra // 2))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        return F.relu(out + shortcut)


class ResNet20(nn.Module):
    def __init__(self, variant='baseline'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU())
        blocks, cin = [], 16
        for stage, cout in enumerate((16, 32, 64)):
            for block in range(3):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(BasicBlock(cin, cout, stride, variant in ('dw', 'dw_se'), variant in ('se', 'dw_se')))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.classifier(self.blocks(self.stem(x)).mean((2, 3)))


def complexity(model):
    """Batch=1 Conv/Linear multiply-accumulates; excludes BN/activation/pool/SE multiply.
    FLOPs convention: 2*MACs. Includes SE's two 1x1 convolutions.
    """
    macs, handles = [], []
    def count(m, inputs, output):
        if isinstance(m, nn.Conv2d):
            macs.append(output.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            macs.append(output.numel() * m.in_features)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(count))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=next(model.parameters()).device))
    for h in handles:
        h.remove()
    model.train(was_training)
    return {'parameters': sum(p.numel() for p in model.parameters()), 'macs': sum(macs), 'flops': 2 * sum(macs)}
