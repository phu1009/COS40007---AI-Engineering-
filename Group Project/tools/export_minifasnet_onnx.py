"""Export the pre-trained MiniFASNet baseline (Minivision Silent-Face-Anti-Spoofing, Apache-2.0)
from PyTorch .pth weights to ONNX for the Raspberry Pi edge service.

Run on a PC (needs torch + onnx), not on the Pi:

    git clone --depth 1 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git sf
    python tools/export_minifasnet_onnx.py --weights-dir sf/resources/anti_spoof_models --out models

The network is re-implemented here (same layer names, so the weights load with strict=True)
to avoid importing third-party code. Each exported model takes a 1x3x80x80 float32 BGR crop
with raw 0-255 pixel values (no /255, no mean/std) and returns softmax probabilities
[p_spoof_a, p_live, p_spoof_b] - class 1 is a live (bona fide) face.
"""

import argparse
import hashlib
from pathlib import Path

import torch
from torch import nn

KEEP = {
    "1.8M": [32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
             154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
             128, 26, 26, 128, 26, 26, 128, 512, 512],
    "1.8M_": [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128,
              231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308, 308,
              128, 26, 26, 128, 26, 26, 128, 512, 512],
}

# file name -> (architecture, keep key, use SE blocks, crop scale)
MODELS = {
    "2.7_80x80_MiniFASNetV2.pth": ("minifasnet_v2", "1.8M_", False, 2.7),
    "4_0_0_80x80_MiniFASNetV1SE.pth": ("minifasnet_v1se", "1.8M", True, 4.0),
}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.prelu = nn.PReLU(cout)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class SEModule(nn.Module):
    def __init__(self, ch, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.bn2(self.fc2(self.relu(self.bn1(self.fc1(self.avg_pool(x)))))))
        return x * w


class DepthWise(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, stride=2, se=False):
        super().__init__()
        self.conv = ConvBlock(c1[0], c1[1])
        self.conv_dw = ConvBlock(c2[0], c2[1], k=3, s=stride, p=1, g=c2[0])
        self.project = LinearBlock(c3[0], c3[1])
        self.residual = residual
        if se:
            self.se_module = SEModule(c3[1], 4)
        self.se = se

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        if self.residual:
            if self.se:
                out = self.se_module(out)
            out = x + out
        return out


class Residual(nn.Module):
    def __init__(self, keep, idx, n, se):
        super().__init__()
        blocks = []
        for i in range(n):
            j = idx + 3 * i
            c1, c2, c3 = (keep[j], keep[j + 1]), (keep[j + 1], keep[j + 2]), (keep[j + 2], keep[j + 3])
            blocks.append(DepthWise(c1, c2, c3, residual=True, stride=1, se=se and i == n - 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(nn.Module):
    def __init__(self, keep, se, conv6_kernel=(5, 5), embedding=128, classes=3):
        super().__init__()
        k = keep
        self.conv1 = ConvBlock(3, k[0], k=3, s=2, p=1)
        self.conv2_dw = ConvBlock(k[0], k[1], k=3, s=1, p=1, g=k[1])
        self.conv_23 = DepthWise((k[1], k[2]), (k[2], k[3]), (k[3], k[4]))
        self.conv_3 = Residual(k, 4, 4, se)
        self.conv_34 = DepthWise((k[16], k[17]), (k[17], k[18]), (k[18], k[19]))
        self.conv_4 = Residual(k, 19, 6, se)
        self.conv_45 = DepthWise((k[37], k[38]), (k[38], k[39]), (k[39], k[40]))
        self.conv_5 = Residual(k, 40, 2, se)
        self.conv_6_sep = ConvBlock(k[46], k[47])
        self.conv_6_dw = LinearBlock(k[47], k[48], k=conv6_kernel, g=k[48])
        self.linear = nn.Linear(512, embedding, bias=False)
        self.bn = nn.BatchNorm1d(embedding)
        self.drop = nn.Dropout(0.0)
        self.prob = nn.Linear(embedding, classes, bias=False)

    def forward(self, x):
        for layer in (self.conv1, self.conv2_dw, self.conv_23, self.conv_3, self.conv_34,
                      self.conv_4, self.conv_45, self.conv_5, self.conv_6_sep, self.conv_6_dw):
            x = layer(x)
        x = self.prob(self.drop(self.bn(self.linear(torch.flatten(x, 1)))))
        return torch.softmax(x, dim=1)


def load(path: Path, keep_key: str, se: bool) -> MiniFASNet:
    model = MiniFASNet(KEEP[keep_key], se)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--out", default=Path("models"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fname, (name, keep_key, se, scale) in MODELS.items():
        model = load(args.weights_dir / fname, keep_key, se)
        out = args.out / f"{name}_{scale}_80x80.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, 80, 80), str(out), input_names=["input"],
                          output_names=["probs"], opset_version=13, dynamo=False)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"{out}  scale={scale}  sha256={sha}")


if __name__ == "__main__":
    main()
