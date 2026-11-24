# Edge model files

The `.onnx` files are not committed (see `.gitignore`). `python -m src.edge.fetch_models` downloads them
from the GitHub Release `pad-model-v0.1.0` of this repo and checks each sha256 against
`configs/pad_config.yaml`.

| File | Role | Source | Licence | sha256 |
|---|---|---|---|---|
| `minifasnet_v2_2.7_80x80.onnx` | PAD model, crop scale 2.7 | Minivision [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) `2.7_80x80_MiniFASNetV2.pth`, exported with `tools/export_minifasnet_onnx.py` | Apache-2.0 | `67db1c98…41e1401` |
| `minifasnet_v1se_4.0_80x80.onnx` | PAD model, crop scale 4.0 | same repo, `4_0_0_80x80_MiniFASNetV1SE.pth` | Apache-2.0 | `3ac219aa…d41e342` |
| `face_detection_yunet_2023mar.onnx` | Face detector | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (same file as `yunet_s_640_640.onnx` in [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)) | MIT / BSD-3-Clause | `8f2383e4…2552fa4` |

Licence texts are in `licenses/`. The MiniFASNet weights are the authors' pre-trained baseline (trained on
Minivision's own data), not a model trained by Group 3.

## Model contract

- Input `input`: `1x3x80x80` float32, **BGR**, raw pixel values 0–255 (no scaling, no mean/std).
  The crop is the face box enlarged `crop_scale` times around its centre (Silent-Face `CropImage`).
- Output `probs`: softmax over 3 classes; index 1 = live (bona fide), 0 and 2 = spoof.
- `p_live` = mean of index 1 over the two models.

## Re-create the files on a PC

```bash
pip install torch onnx
git clone --depth 1 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git sf
python tools/export_minifasnet_onnx.py --weights-dir sf/resources/anti_spoof_models --out models
```

Parity check (24/11/2026): ONNX Runtime vs PyTorch max abs difference 8.3e-7. On the Silent-Face sample
images the edge pipeline gives p_live 0.98 (real face), 0.08 and 0.29 (photos of printed faces).
