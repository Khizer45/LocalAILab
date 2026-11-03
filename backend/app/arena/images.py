"""Image loading, preprocessing and ONNX helpers for the Model Arena.

* :func:`preprocess` turns image files into the tensor layout a model expects
  (size, NCHW/NHWC, normalisation, RGB/BGR, resize or centre crop).
* :class:`OnnxSession` wraps ONNX Runtime (CPU) and handles models whose batch
  dimension is fixed to 1.
* :func:`feature_session` turns any ONNX *classifier* into a feature extractor
  by exposing the input of its final dense (Gemm/MatMul) layer – the classic
  transfer-learning recipe: a pretrained backbone + a small classifier trained
  on the user's own classes.
* Built-in backbones (MobileNetV2, EfficientNet-Lite4 from the ONNX Model Zoo)
  are downloaded once on request and verified by SHA-256.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.core.errors import AppError, ValidationFailed

NORMALIZATIONS: dict[str, tuple[list[float], list[float]]] = {
    # name: (mean, std) applied to pixel values in 0..255
    "imagenet": ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
    "zero_one": ([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]),
    "minus_one_one": ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
    "efficientnet_lite": ([127.0, 127.0, 127.0], [128.0, 128.0, 128.0]),
    "raw": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
NORMALIZATION_HELP = {
    "imagenet": "PyTorch / torchvision models (mean/std of ImageNet, pixels 0–1)",
    "zero_one": "Keras models trained with Rescaling(1/255)",
    "minus_one_one": "Keras MobileNet/EfficientNetV2 'preprocess_input' style (−1…1)",
    "efficientnet_lite": "TF-Lite EfficientNet-Lite ((x−127)/128)",
    "raw": "No normalisation (pixels 0–255), e.g. Keras EfficientNet with built-in rescaling",
}


@dataclass
class ImagePreprocess:
    size: int = 224
    layout: str = "nchw"  # nchw | nhwc
    normalization: str = "imagenet"
    color: str = "rgb"  # rgb | bgr
    resize_mode: str = "resize"  # resize | center_crop
    channels: int = 3

    def validate(self) -> ImagePreprocess:
        if self.layout not in ("nchw", "nhwc"):
            raise ValidationFailed("layout must be nchw or nhwc.")
        if self.normalization not in NORMALIZATIONS:
            raise ValidationFailed(f"normalization must be one of {', '.join(NORMALIZATIONS)}.")
        if self.color not in ("rgb", "bgr"):
            raise ValidationFailed("color must be rgb or bgr.")
        if self.resize_mode not in ("resize", "center_crop"):
            raise ValidationFailed("resize_mode must be resize or center_crop.")
        if not 8 <= int(self.size) <= 1024:
            raise ValidationFailed("Image size must be between 8 and 1024 pixels.")
        if self.channels not in (1, 3):
            raise ValidationFailed("Only 1- or 3-channel models are supported.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _load_rgb(path: Path, size: int, resize_mode: str):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if resize_mode == "center_crop":
            short = size * 256 // 224
            w, h = im.size
            scale = short / min(w, h)
            im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
            w, h = im.size
            left, top = (w - size) // 2, (h - size) // 2
            im = im.crop((left, top, left + size, top + size))
        else:
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def preprocess(paths: list[Path], pp: ImagePreprocess) -> np.ndarray:
    arrs = []
    mean, std = (np.array(v, dtype=np.float32) for v in NORMALIZATIONS[pp.normalization])
    for p in paths:
        a = _load_rgb(p, pp.size, pp.resize_mode)
        a = (a - mean) / std
        if pp.color == "bgr":
            a = a[..., ::-1]
        if pp.channels == 1:
            a = a.mean(axis=-1, keepdims=True)
        arrs.append(a)
    batch = np.stack(arrs).astype(np.float32)
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2) if pp.layout == "nchw" else batch)


def raw_pixel_features(paths: list[Path], size: int = 32) -> np.ndarray:
    """Baseline image features without any neural network: a tiny RGB thumbnail."""
    return np.stack([(_load_rgb(p, size, "resize") / 255.0).ravel() for p in paths]).astype(np.float32)


# ----------------------------------------------------------------------------- ONNX
def _ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise AppError("ONNX Runtime is not installed.", code="dependency_missing", status_code=500,
                       hint="Run: pip install onnxruntime") from exc
    return ort


class OnnxSession:
    def __init__(self, model: str | bytes | Path, threads: int | None = None):
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        try:
            src = str(model) if isinstance(model, Path) else model
            self.session = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - invalid / unsupported model
            raise ValidationFailed(f"ONNX Runtime could not load the model: {str(exc)[:300]}") from exc
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        shape = self.inputs[0].shape
        self.fixed_batch = isinstance(shape[0], int) and shape[0] == 1 if shape else False

    def describe(self) -> dict:
        return {"inputs": [{"name": i.name, "shape": [d if isinstance(d, int) else str(d) for d in i.shape],
                            "type": i.type} for i in self.inputs],
                "outputs": [{"name": o.name, "shape": [d if isinstance(d, int) else str(d) for d in (o.shape or [])],
                             "type": o.type} for o in self.outputs]}

    def run(self, feeds: dict, output_names: list[str] | None = None) -> list[np.ndarray]:
        return self.session.run(output_names, feeds)

    def run_batched(self, x: np.ndarray, output: str | None = None, batch: int = 32) -> np.ndarray:
        """Run the first input on `x` in batches and return one output (stacked)."""
        name = self.inputs[0].name
        step = 1 if self.fixed_batch else batch
        outs = []
        names = [output] if output else [self.outputs[0].name]
        for i in range(0, len(x), step):
            outs.append(np.asarray(self.session.run(names, {name: x[i:i + step]})[0]))
        return np.concatenate(outs, axis=0)


def infer_image_preprocess(sess: OnnxSession, base: ImagePreprocess) -> ImagePreprocess:
    """Fill layout/size/channels from the model's input shape when they are fixed."""
    shape = sess.inputs[0].shape
    pp = ImagePreprocess(**base.to_dict())
    if len(shape) != 4:
        raise ValidationFailed(f"Image models need a 4-D input (batch, …); this model's input is {shape}.")
    dims = [d if isinstance(d, int) else None for d in shape]
    if dims[1] in (1, 3):
        pp.layout, pp.channels = "nchw", dims[1]
        if dims[2]:
            pp.size = dims[2]
    elif dims[3] in (1, 3):
        pp.layout, pp.channels = "nhwc", dims[3]
        if dims[1]:
            pp.size = dims[1]
    return pp.validate()


FEATURE_PASSTHROUGH = {"Softmax", "LogSoftmax", "Sigmoid", "Add", "Identity", "Relu", "Mul", "Reshape", "Squeeze",
                       "Flatten", "Cast", "Dropout"}


def find_feature_tensor(model) -> str | None:
    """Walk back from the first graph output to the input of the last dense layer."""
    producers = {o: n for n in model.graph.node for o in n.output}
    name = model.graph.output[0].name
    for _ in range(12):
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type in ("Gemm", "MatMul"):
            return node.input[0]
        if node.op_type in FEATURE_PASSTHROUGH and node.input:
            name = node.input[0]
            continue
        return None
    return None


def feature_session(path: Path) -> tuple[OnnxSession, str]:
    """An ONNX session whose extra output is the penultimate (embedding) tensor."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise AppError("The 'onnx' package is not installed.", code="dependency_missing", status_code=500,
                       hint="Run: pip install onnx") from exc
    model = onnx.load(str(path))
    feat = find_feature_tensor(model)
    if feat is None:
        raise ValidationFailed("Could not find the final dense layer of this ONNX model, so it cannot be used as a "
                               "feature extractor.", hint="Upload it as a classifier instead.")
    if feat not in {o.name for o in model.graph.output}:
        model.graph.output.append(onnx.helper.make_tensor_value_info(feat, onnx.TensorProto.FLOAT, None))
    return OnnxSession(model.SerializeToString()), feat


def count_parameters(path: Path) -> int | None:
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        return int(sum(int(np.prod(t.dims)) for t in model.graph.initializer))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- built-in backbones
_ZOO = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification"
BACKBONE_SOURCE = ("ONNX Model Zoo (repository licence Apache-2.0). The weights were trained on ImageNet, whose terms "
                   "limit the dataset to non-commercial research; check the model card / get advice before commercial "
                   "use, or upload your own feature extractor.")
BACKBONES: dict[str, dict] = {
    "mobilenetv2": {
        "name": "MobileNetV2 (ImageNet)", "file": "mobilenetv2-12.onnx",
        "url": f"{_ZOO}/mobilenet/model/mobilenetv2-12.onnx",
        "sha256": "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5", "size_mb": 13.3,
        "feature_dim": 1280, "source": BACKBONE_SOURCE,
        "preprocess": {"size": 224, "layout": "nchw", "normalization": "imagenet", "color": "rgb",
                       "resize_mode": "center_crop", "channels": 3},
        "notes": "Fast and small – good default on laptops without a GPU.",
    },
    "efficientnet_lite4": {
        "name": "EfficientNet-Lite4 (ImageNet)", "file": "efficientnet-lite4-11.onnx",
        "url": f"{_ZOO}/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "sha256": "d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12", "size_mb": 49.5,
        "feature_dim": 1280, "source": BACKBONE_SOURCE,
        "preprocess": {"size": 224, "layout": "nhwc", "normalization": "efficientnet_lite", "color": "rgb",
                       "resize_mode": "center_crop", "channels": 3},
        "notes": "More accurate, slower (batch size fixed to 1 in this export).",
    },
}


def backbone_path(arena_dir: Path, backbone_id: str) -> Path:
    return arena_dir / "backbones" / BACKBONES[backbone_id]["file"]


def backbone_status(arena_dir: Path) -> list[dict]:
    out = []
    for bid, b in BACKBONES.items():
        p = backbone_path(arena_dir, bid)
        out.append({"id": bid, "name": b["name"], "size_mb": b["size_mb"], "feature_dim": b["feature_dim"],
                    "source": b["source"], "notes": b["notes"], "installed": p.exists(), "url": b["url"],
                    "file": b["file"], "preprocess": b["preprocess"]})
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_backbone(arena_dir: Path, backbone_id: str, http_client) -> dict:
    """Download a backbone through the (offline-guarded) HTTP client and verify it."""
    if backbone_id not in BACKBONES:
        raise ValidationFailed(f"Unknown backbone '{backbone_id}'.")
    b = BACKBONES[backbone_id]
    dest = backbone_path(arena_dir, backbone_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    t0 = time.perf_counter()
    with http_client.stream("GET", b["url"], follow_redirects=True) as r:
        if r.status_code != 200:
            raise AppError(f"Download failed (HTTP {r.status_code}).", code="download_failed", status_code=502)
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                f.write(chunk)
    digest = sha256_file(tmp)
    if digest != b["sha256"]:
        tmp.unlink(missing_ok=True)
        raise AppError("The downloaded file failed the SHA-256 check and was deleted.", code="checksum_mismatch",
                       status_code=502)
    tmp.replace(dest)
    return {"id": backbone_id, "installed": True, "seconds": round(time.perf_counter() - t0, 1)}
