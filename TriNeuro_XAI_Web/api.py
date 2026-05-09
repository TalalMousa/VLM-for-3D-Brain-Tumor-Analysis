from __future__ import annotations

import base64
import io
import os
import re
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from monai.networks.nets import ResNet, ResNetBottleneck, resnet18
from PIL import Image
from scipy import ndimage as ndi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "Report Generation Files" / ".env")
BINARY_CKPT = ROOT / "Final_Resnet_Tumor_NoTumor_93_ACC" / "NoTumor vs Tumor 93_.pt"
TYPE_CKPT = (
    ROOT
    / "mask_guided_resnet18_gated_attention_raw_cache112"
    / "best_mask_guided_resnet18_gated_attention_t1c_t2f_raw112.pt"
)
SEG_CKPT = ROOT / "Segmentation" / "seg_model.pth"
REPORT_DIR = ROOT / "Report Generation Files"
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

try:
    from module2_feature_formatter import format_clinical_findings
except Exception:  # pragma: no cover
    format_clinical_findings = None


BINARY_SHAPE = (128, 128, 128)
TYPE_SHAPE = (112, 112, 112)
SEG_SHAPE = (128, 160, 112)
FAST_SLICE_COUNT = 24
FULL_SLICE_COUNT = 64
FAST_SLICE_SIZE = 420
FULL_SLICE_SIZE = 900
BINARY_LABELS = {0: "no_tumor", 1: "tumor"}
TYPE_LABELS = {0: "glioma", 1: "meningioma"}


app = FastAPI(title="TriNeuro AI API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_upload(upload: Optional[UploadFile], folder: Path, fallback_name: str) -> Optional[Path]:
    if upload is None:
        return None
    filename = upload.filename or fallback_name
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif lower.endswith(".npy"):
        suffix = ".npy"
    else:
        suffix = Path(filename).suffix or ".nii"
    path = folder / f"{Path(filename).stem.replace('.', '_')}{suffix}"
    with open(path, "wb") as handle:
        handle.write(upload.file.read())
    return path


def load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if path.suffix.lower() == ".npy":
        data = np.load(str(path), allow_pickle=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 4:
            # Accept either channel-first or channel-last arrays and use the first channel.
            if data.shape[0] <= 8:
                data = data[0]
            elif data.shape[-1] <= 8:
                data = data[..., 0]
            else:
                data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D or 4D .npy MRI array, got shape {data.shape}")
        return data, (1.0, 1.0, 1.0)

    nii = nib.as_closest_canonical(nib.load(str(path)))
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return np.asarray(data, dtype=np.float32), tuple(float(x) for x in nii.header.get_zooms()[:3])


def is_brats_file(path: Optional[Path]) -> bool:
    if path is None:
        return False
    name = path.name.lower()
    return bool(re.search(r"(^|[^a-z0-9])brats[-_](gli|men)([^a-z0-9]|$)", name))


def center_crop_or_pad_3d(arr: np.ndarray, target_shape: tuple[int, int, int] = (240, 240, 155)) -> np.ndarray:
    out = np.zeros(target_shape, dtype=arr.dtype)
    in_slices = []
    out_slices = []
    for in_size, target_size in zip(arr.shape, target_shape):
        if in_size >= target_size:
            start_in = (in_size - target_size) // 2
            end_in = start_in + target_size
            start_out = 0
            end_out = target_size
        else:
            start_in = 0
            end_in = in_size
            start_out = (target_size - in_size) // 2
            end_out = start_out + in_size
        in_slices.append(slice(start_in, end_in))
        out_slices.append(slice(start_out, end_out))
    out[out_slices[0], out_slices[1], out_slices[2]] = arr[in_slices[0], in_slices[1], in_slices[2]]
    return out


def ixi_step1_register_or_shape_normalize(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 from preprocess_ixi_v2: rigid registration when possible, otherwise shape-normalize."""
    if path.suffix.lower() == ".npy":
        arr, _spacing = load_nifti(path)
        arr = center_crop_or_pad_3d(np.asarray(arr, dtype=np.float32), (240, 240, 155))
        return arr, np.eye(4, dtype=np.float32)

    template_path = Path(os.getenv("TRINEURO_IXI_TEMPLATE", str(ROOT / "sri24_t2.nii.gz")))
    if template_path.exists():
        try:
            import ants

            moving = ants.image_read(str(path))
            fixed = ants.image_read(str(template_path))
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
            warped = reg["warpedmovout"]
            warped_1mm = ants.resample_image(warped, (1.0, 1.0, 1.0), use_voxels=False, interp_type=1)
            arr = center_crop_or_pad_3d(warped_1mm.numpy().astype(np.float32), (240, 240, 155))
            return arr, np.eye(4, dtype=np.float32)
        except Exception:
            pass

    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    arr = center_crop_or_pad_3d(np.asarray(arr, dtype=np.float32), (240, 240, 155))
    return arr, np.eye(4, dtype=np.float32)


def ixi_step2_skull_strip(arr: np.ndarray) -> np.ndarray:
    """Step 2 from preprocess_ixi_v2: skull strip, using the local fallback."""
    brain = estimate_ixi_brain_mask(arr)
    return np.where(brain, arr, 0.0).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def estimate_ixi_brain_mask(volume: np.ndarray) -> np.ndarray:
    """More aggressive single-volume skull-strip fallback for non-BraTS IXI T2 scans."""
    arr = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    abs_arr = np.abs(arr)
    values = abs_arr[abs_arr > 1e-6]
    if values.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    low_threshold = max(float(np.percentile(values, 8)), 1e-6)
    head = abs_arr > low_threshold
    head = ndi.binary_opening(head, structure=np.ones((3, 3, 3), dtype=bool))
    head = ndi.binary_closing(head, structure=np.ones((5, 5, 5), dtype=bool))
    head = largest_component(head)
    head = ndi.binary_fill_holes(head)
    if not np.any(head):
        return estimate_brain_mask(arr)

    edge_distance = ndi.distance_transform_edt(head)
    strip_distance = float(os.getenv("TRINEURO_IXI_STRIP_DISTANCE", "12"))
    inner = edge_distance >= strip_distance
    if int(np.count_nonzero(inner)) < 5000:
        inner = ndi.binary_erosion(head, structure=np.ones((3, 3, 3), dtype=bool), iterations=8)

    inner = largest_component(inner)
    inner = ndi.binary_fill_holes(inner)
    intensity_floor = abs_arr >= float(np.percentile(values, 12))
    refined = largest_component(inner & intensity_floor)
    if int(np.count_nonzero(refined)) >= 5000:
        inner = ndi.binary_fill_holes(refined)
    inner = ndi.binary_dilation(inner, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    inner &= head

    if int(np.count_nonzero(inner)) < 5000:
        fallback = estimate_brain_mask(arr)
        if int(np.count_nonzero(fallback)) >= 5000:
            return fallback
        return head.astype(bool)
    return inner.astype(bool)


def preprocess_non_brats_mri(path: Path, output_dir: Path) -> Path:
    """Single-upload IXI-style preprocessing for non-BraTS MRI files."""
    registered, affine = ixi_step1_register_or_shape_normalize(path)
    stripped = ixi_step2_skull_strip(registered)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{path.stem.replace('.', '_')}_ixi_v2.nii.gz"
    nib.save(nib.Nifti1Image(stripped.astype(np.float32), affine), str(out_path))
    return out_path


def preprocess_non_brats_uploads(
    files: dict[str, Optional[Path]],
    folder: Path,
) -> tuple[dict[str, Optional[Path]], dict[str, dict[str, str]]]:
    processed = dict(files)
    info: dict[str, dict[str, str]] = {}
    output_dir = folder / "ixi_v2_preprocessed"
    for label, path in files.items():
        if path is None:
            continue
        if label == "Segmentation mask":
            info[label] = {"status": "skipped", "reason": "segmentation mask", "input": path.name}
            continue
        if is_brats_file(path):
            info[label] = {"status": "skipped", "reason": "BraTS filename", "input": path.name}
            continue
        if path.suffix.lower() == ".npy":
            info[label] = {
                "status": "skipped",
                "reason": "preprocessed numpy cache",
                "input": path.name,
            }
            continue
        out_path = preprocess_non_brats_mri(path, output_dir)
        processed[label] = out_path
        info[label] = {
            "status": "applied",
            "reason": "non-BraTS filename",
            "input": path.name,
            "output": out_path.name,
        }
    return processed, info


def estimate_brain_mask(volume: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    abs_arr = np.abs(arr)
    values = abs_arr[abs_arr > 1e-6]
    if values.size == 0:
        return np.zeros_like(volume, dtype=bool)
    threshold = max(float(np.percentile(values, 20)), 1e-6)
    mask = abs_arr > threshold
    if mask.sum() == 0:
        mask = abs_arr > 1e-6
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3), dtype=bool))
    labeled, count = ndi.label(mask)
    if count > 0:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
    mask = ndi.binary_fill_holes(mask)
    return ndi.binary_dilation(mask, iterations=2).astype(bool)


def normalize01(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def normalize01_exact(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x, dtype=np.float32) if mx - mn <= eps else np.clip((x - mn) / (mx - mn), 0, 1).astype(np.float32)


def make_brain_mask_multichannel(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    abs_vol = np.max(np.abs(vol), axis=0)
    vals = abs_vol[abs_vol > eps]
    if vals.size < 100:
        return abs_vol > eps
    return (abs_vol >= max(float(np.percentile(vals, 5)), eps)).astype(bool)


def display_slice_exact(x: np.ndarray, brain: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if brain is None:
        brain = np.abs(x) > 1e-6
    brain = np.asarray(brain, dtype=bool)
    if np.count_nonzero(brain) > 0.85 * brain.size:
        abs_x = np.abs(x)
        vals = abs_x[abs_x > 1e-6]
        if vals.size > 20:
            brain = abs_x >= max(float(np.percentile(vals, 8)), 1e-6)
    vals = x[brain] if np.count_nonzero(brain) > 20 else x.reshape(-1)
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    out = (np.clip(x, lo, hi) - lo) / (hi - lo)
    out[~brain] = 0
    return np.clip(out, 0, 1).astype(np.float32)


def rotate_show(x: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(x))


def max_focus_slices(cam: np.ndarray, seg: Optional[np.ndarray] = None, brain: Optional[np.ndarray] = None) -> tuple[int, int, int]:
    score = np.asarray(cam, dtype=np.float32).copy()
    if brain is not None:
        score *= brain.astype(np.float32)
    if seg is not None and np.count_nonzero(seg) > 0:
        score = 0.65 * score + 0.35 * seg.astype(np.float32)
    if not np.isfinite(score).all() or float(score.max()) <= 0:
        d, h, w = score.shape
        return d // 2, h // 2, w // 2
    return (
        int(np.argmax(score.sum(axis=(1, 2)))),
        int(np.argmax(score.sum(axis=(0, 2)))),
        int(np.argmax(score.sum(axis=(0, 1)))),
    )


def robust_zscore(volume: np.ndarray) -> np.ndarray:
    mask = estimate_brain_mask(volume)
    if not np.any(mask):
        mask = np.abs(volume) > 1e-6
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[mask]
    lo, hi = np.percentile(values, [1, 99])
    clipped = np.clip(volume, lo, hi)
    mean = float(clipped[mask].mean())
    std = float(clipped[mask].std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    return np.where(mask, (clipped - mean) / std, 0).astype(np.float32)


def brain_bbox(mask: np.ndarray, margin: int = 8) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    low = np.maximum(coords.min(axis=0) - margin, 0)
    high = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))  # type: ignore[return-value]


def resize3d(volume: np.ndarray, shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    zoom = [shape[i] / volume.shape[i] for i in range(3)]
    return ndi.zoom(volume, zoom=zoom, order=order).astype(np.float32)


def preprocess_binary(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    raw, spacing = load_nifti(path)
    normed = robust_zscore(raw)
    cropped = normed[brain_bbox(np.abs(normed) > 1e-6, margin=8)]
    return resize3d(cropped, BINARY_SHAPE, order=1), spacing, raw


def preprocess_display_volume(path: Path) -> np.ndarray:
    raw, _spacing = load_nifti(path)
    return resize3d(robust_zscore(raw), BINARY_SHAPE, order=1)


def preprocess_type(t1c_path: Path, t2f_path: Path) -> np.ndarray:
    t1c, _ = load_nifti(t1c_path)
    t2f, _ = load_nifti(t2f_path)
    t1c = resize3d(robust_zscore(t1c), TYPE_SHAPE, order=1)
    t2f = resize3d(robust_zscore(t2f), TYPE_SHAPE, order=1)
    return np.stack([t1c, t2f], axis=0).astype(np.float32)


def preprocess_segmentation(files: dict[str, Optional[Path]]) -> np.ndarray:
    primary = files["Primary MRI"]
    if primary is None:
        raise ValueError("Primary MRI is required for segmentation.")

    # seg_model.pth was trained with 4 BraTS-style channels:
    # T1n, T1c, T2w, and T2-FLAIR. If a modality is missing, reuse the
    # primary upload so the request can still run from a single MRI file.
    modality_paths = [
        files["T1n"] or primary,
        files["T1c"] or primary,
        files["T2w"] or primary,
        files["T2F / FLAIR"] or files["T2w"] or primary,
    ]
    channels = []
    for path in modality_paths:
        volume, _ = load_nifti(path)
        channels.append(resize3d(robust_zscore(volume), SEG_SHAPE, order=1))
    return np.stack(channels, axis=0).astype(np.float32)


def build_binary_model() -> nn.Module:
    return ResNet(
        block=ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=[32, 64, 128, 256],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
        conv1_t_size=7,
        conv1_t_stride=2,
        no_max_pool=False,
        shortcut_type="B",
        feed_forward=True,
    )


def choose_group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def replace_batchnorm_with_groupnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            channels = child.num_features
            setattr(module, name, nn.GroupNorm(choose_group_count(channels), channels))
        else:
            replace_batchnorm_with_groupnorm(child)


class MaskGuidedResNet18GatedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=2,
            feed_forward=False,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
        )
        replace_batchnorm_with_groupnorm(self.encoder)
        self.feature_channels = 512
        self.attn = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1),
            nn.GroupNorm(choose_group_count(128), 128),
            nn.SiLU(inplace=True),
            nn.Conv3d(128, 1, kernel_size=1),
        )
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(0.25), nn.Linear(1024, 2))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.relu(x) if hasattr(self.encoder, "relu") else F.relu(x, inplace=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self._activation(x)
        if hasattr(self.encoder, "maxpool") and not getattr(self.encoder, "no_max_pool", False):
            x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.forward_features(x)
        attn_logits = self.attn(feat)
        attn = torch.softmax(attn_logits.flatten(2), dim=-1).view_as(attn_logits)
        attentive = (feat * attn).sum(dim=(2, 3, 4))
        global_avg = F.adaptive_avg_pool3d(feat, 1).flatten(1)
        logits = self.head(torch.cat([global_avg, attentive], dim=1))
        if return_attention:
            return {"logits": logits, "attention": attn, "attention_logits": attn_logits, "features": feat}
        return logits


@lru_cache(maxsize=1)
def binary_model() -> nn.Module:
    model = build_binary_model()
    state = torch.load(BINARY_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device()).eval()


@lru_cache(maxsize=1)
def type_model() -> MaskGuidedResNet18GatedAttention:
    model = MaskGuidedResNet18GatedAttention()
    state = torch.load(TYPE_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device()).eval()


@lru_cache(maxsize=1)
def segmentation_model() -> nn.Module:
    try:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "nnunetv2 is required to run Segmentation/seg_model.pth. "
            "Install the project requirements before starting FastAPI."
        ) from exc

    checkpoint = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    init_args = checkpoint["init_args"]
    plans = init_args["plans"]
    configuration = plans["configurations"][init_args["configuration"]]
    architecture = configuration["architecture"]
    output_channels = int(checkpoint["network_weights"]["decoder.seg_layers.0.weight"].shape[0])

    model = get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture.get("_kw_requires_import", []),
        input_channels=4,
        output_channels=output_channels,
        allow_init=False,
        deep_supervision=False,
    )
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    return model.to(device()).eval()


def predict_binary(volume: np.ndarray) -> dict:
    model = binary_model()
    x = torch.from_numpy(volume[None, None]).float().to(device())
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {"label": BINARY_LABELS[idx], "confidence": float(probs[idx]), "probabilities": {BINARY_LABELS[i]: float(probs[i]) for i in range(2)}}


def predict_type(volume: np.ndarray) -> dict:
    model = type_model()
    x = torch.from_numpy(volume[None]).float().to(device())
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {"label": TYPE_LABELS[idx], "confidence": float(probs[idx]), "probabilities": {TYPE_LABELS[i]: float(probs[i]) for i in range(2)}}


def predict_segmentation(volume: np.ndarray) -> np.ndarray:
    model = segmentation_model()
    x = torch.from_numpy(volume[None]).float().to(device())
    use_amp = x.is_cuda
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()

    whole_tumor = probs[0] > 0.5
    tumor_core = (probs[1] > 0.5) & whole_tumor if probs.shape[0] > 1 else np.zeros_like(whole_tumor)
    enhancing_tumor = (probs[2] > 0.5) & tumor_core if probs.shape[0] > 2 else np.zeros_like(whole_tumor)

    mask = np.zeros(whole_tumor.shape, dtype=np.float32)
    mask[whole_tumor] = 1.0
    mask[tumor_core] = 2.0
    mask[enhancing_tumor] = 3.0

    if np.count_nonzero(mask) > 0:
        binary = ndi.binary_opening(mask > 0, structure=np.ones((3, 3, 3), dtype=bool))
        binary = ndi.binary_fill_holes(binary)
        mask = np.where(binary, mask, 0).astype(np.float32)
    return mask


class GradCAMPlusPlus3D:
    def __init__(self, model: MaskGuidedResNet18GatedAttention, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(x, return_attention=True)
        logits = outputs["logits"]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        logits[:, class_idx].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed.")
        acts = self.activations
        grads = self.gradients
        grads2 = grads.pow(2)
        grads3 = grads.pow(3)
        denom = 2 * grads2 + (acts * grads3).sum(dim=(2, 3, 4), keepdim=True)
        alpha = grads2 / denom.clamp_min(1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=TYPE_SHAPE, mode="trilinear", align_corners=False)
        return normalize01(cam[0, 0].detach().cpu().numpy())


def compute_gradcam(type_volume: np.ndarray) -> np.ndarray:
    model = type_model()
    cam = GradCAMPlusPlus3D(model, model.encoder.layer4[-1].conv2)
    try:
        x = torch.from_numpy(type_volume[None]).float().to(device()).clone()
        with torch.enable_grad():
            return cam(x)
    finally:
        cam.close()


def preview_mask(volume: np.ndarray) -> np.ndarray:
    img = normalize01(volume)
    brain = estimate_brain_mask(volume)
    values = img[brain] if brain.any() else img.reshape(-1)
    threshold = float(np.percentile(values, 97.5)) if values.size else 1.0
    mask = (img >= threshold) & brain
    labeled, count = ndi.label(mask)
    if count > 0:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
    return mask.astype(np.float32)


def png_data_url(rgb: np.ndarray, size: int = FAST_SLICE_SIZE) -> str:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_rgb(heat: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    heat = normalize01(heat)
    return colormaps[cmap_name](heat)[..., :3].astype(np.float32)


def overlay_slice(
    base: np.ndarray,
    overlay: Optional[np.ndarray] = None,
    alpha: float = 0.62,
    mode: str = "gradcam",
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    brain = np.abs(base) > 1e-6
    if np.count_nonzero(brain) > 0.85 * brain.size:
        abs_base = np.abs(base)
        vals = abs_base[abs_base > 1e-6]
        if vals.size > 20:
            brain = abs_base >= max(float(np.percentile(vals, 8)), 1e-6)
    if np.count_nonzero(brain) > 20:
        brain = ndi.binary_fill_holes(brain)
        brain = ndi.binary_opening(brain, structure=np.ones((3, 3), dtype=bool))
    else:
        brain = None
    gray = display_slice_exact(base, brain)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if overlay is not None:
        heat = normalize01(overlay)
        if brain is not None:
            heat = heat * brain.astype(np.float32)
        if mode == "mask_only":
            rgb = np.zeros_like(rgb)
            mask = heat > 0.15
            color = np.zeros_like(rgb)
            color[..., 0] = 0.05
            color[..., 1] = 0.95
            color[..., 2] = 1.0
            rgb = np.where(mask[..., None], color, rgb)
        elif mode == "segmentation":
            mask = heat > 0.15
            color = np.zeros_like(rgb)
            color[..., 0] = 0.05
            color[..., 1] = 0.95
            color[..., 2] = 1.0
            alpha_map = mask.astype(np.float32) * 0.58
            rgb = rgb * (1 - alpha_map[..., None]) + color * alpha_map[..., None]
        elif mode == "attention":
            heat = ndi.gaussian_filter(heat, sigma=1.15)
            heat = normalize01(heat)
            cam_color = heatmap_rgb(heat, "turbo")
            alpha_map = np.clip((heat - 0.04) / 0.58, 0, 1) ** 0.62
            alpha_map *= 0.66
            rgb = rgb * (1 - alpha_map[..., None]) + cam_color * alpha_map[..., None]
        else:
            # Notebook-style Grad-CAM: vivid jet colors, transparent low activation,
            # and high opacity only at the model's strongest focus region.
            cam_color = heatmap_rgb(heat, "jet")
            alpha_map = np.clip((heat - 0.08) / 0.62, 0, 1) ** 0.55
            alpha_map *= alpha
            rgb = rgb * (1 - alpha_map[..., None]) + cam_color * alpha_map[..., None]
    return np.rot90(rgb)


def pick_indices(size: int, count: int = 64) -> list[int]:
    if size <= count:
        return list(range(size))
    return [int(x) for x in np.linspace(0, size - 1, count)]


def slice_pack(
    volume: np.ndarray,
    overlay: Optional[np.ndarray] = None,
    mode: str = "gradcam",
    count: int = FAST_SLICE_COUNT,
    size: int = FAST_SLICE_SIZE,
) -> dict:
    planes = {
        "axial": 2,
        "coronal": 1,
        "sagittal": 0,
    }
    out = {}
    for name, axis in planes.items():
        images = []
        for idx in pick_indices(volume.shape[axis], count=count):
            if axis == 0:
                base = volume[idx, :, :]
                ov = overlay[idx, :, :] if overlay is not None else None
            elif axis == 1:
                base = volume[:, idx, :]
                ov = overlay[:, idx, :] if overlay is not None else None
            else:
                base = volume[:, :, idx]
                ov = overlay[:, :, idx] if overlay is not None else None
            images.append(png_data_url(overlay_slice(base, ov, mode=mode), size=size))
        out[name] = images
    return out


def extract_plane(volume: np.ndarray, plane: str) -> np.ndarray:
    if plane == "sagittal":
        return volume[volume.shape[0] // 2, :, :]
    if plane == "coronal":
        return volume[:, volume.shape[1] // 2, :]
    return volume[:, :, volume.shape[2] // 2]


def render_tile(slice_2d: np.ndarray, overlay: Optional[np.ndarray] = None, mode: str = "gradcam", size: int = 220) -> Image.Image:
    rgb = overlay_slice(slice_2d, overlay, mode=mode)
    img = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), mode="RGB")
    return img.resize((size, size), Image.Resampling.LANCZOS)


@torch.inference_mode()
def compute_learned_attention(type_volume: np.ndarray) -> np.ndarray:
    model = type_model()
    x = torch.from_numpy(type_volume[None]).float().to(device())
    outputs = model(x, return_attention=True)
    attn = outputs["attention"]
    attn = F.interpolate(attn, size=TYPE_SHAPE, mode="trilinear", align_corners=False)
    return normalize01(attn[0, 0].detach().cpu().numpy())


def mri_column_png(type_volume: np.ndarray, cam: np.ndarray, mask: np.ndarray) -> str:
    brain_3d = make_brain_mask_multichannel(type_volume)
    x_idx, y_idx, z_idx = max_focus_slices(cam, seg=mask, brain=brain_3d)
    views = [
        (type_volume[0, x_idx], brain_3d[x_idx]),
        (type_volume[0, :, y_idx, :], brain_3d[:, y_idx]),
        (type_volume[0, :, :, z_idx], brain_3d[:, :, z_idx]),
    ]
    fig = plt.figure(figsize=(3.2, 9.4), facecolor="#000000", dpi=180)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08, top=0.995, bottom=0.005, left=0.005, right=0.995)
    for row, (sl, brain) in enumerate(views):
        ax = fig.add_subplot(gs[row, 0])
        ax.axis("off")
        ax.set_facecolor("#000000")
        ax.imshow(rotate_show(display_slice_exact(sl, brain)), cmap="gray", vmin=0, vmax=1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#000000", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def xai_panel_png(type_volume: np.ndarray, cam: np.ndarray, mask: np.ndarray, attention: np.ndarray) -> str:
    # Visual logic for the compact "T1c + Grad-CAM++" column:
    # dark background, display_slice(), rotate_show(), and cmap="jet", alpha=0.58.
    brain_3d = make_brain_mask_multichannel(type_volume)
    x_idx, y_idx, z_idx = max_focus_slices(cam, seg=mask, brain=brain_3d)
    views = [
        ("Sagittal", type_volume[:, x_idx], cam[x_idx], brain_3d[x_idx], x_idx),
        ("Coronal", type_volume[:, :, y_idx, :], cam[:, y_idx], brain_3d[:, y_idx], y_idx),
        ("Axial", type_volume[:, :, :, z_idx], cam[:, :, z_idx], brain_3d[:, :, z_idx], z_idx),
    ]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(4.6, 11.8), facecolor="#0e1117", dpi=170)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08, top=0.95, bottom=0.025, left=0.02, right=0.98)
    fig.suptitle("T1c + Grad-CAM++", color="#aaaaaa", fontsize=12, y=0.99)

    for row, (view_name, mri_2ch, cam_sl, brain_sl, slice_idx) in enumerate(views):
        t1c_show = rotate_show(display_slice_exact(mri_2ch[0], brain_sl))
        cam_show = rotate_show(normalize01_exact(cam_sl) * brain_sl.astype(np.float32))
        ax = fig.add_subplot(gs[row, 0])
        ax.axis("off")
        ax.imshow(t1c_show, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam_show, cmap="jet", alpha=0.58, vmin=0, vmax=1)
        ax.text(
            0.02,
            0.97,
            f"{view_name} slice {slice_idx}",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            va="top",
            bbox=dict(facecolor="#0e1117", edgecolor="#00cc66", linewidth=0.7, alpha=0.85, pad=2),
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def xai_full_panel_png(type_volume: np.ndarray, cam: np.ndarray, mask: np.ndarray, attention: np.ndarray) -> str:
    """Render an organized 3-plane AI panel with same-slice mask and Grad-CAM overlays."""
    brain_3d = make_brain_mask_multichannel(type_volume)
    x_idx, y_idx, z_idx = max_focus_slices(cam, seg=mask, brain=brain_3d)
    heat_cmap = LinearSegmentedColormap.from_list(
        "brain_heat",
        ["#000033", "#0000aa", "#0055ff", "#00ccff", "#00ff99", "#aaff00", "#ffcc00", "#ff4400", "#cc0000"],
        N=256,
    )
    views = [
        {
            "name": "Sagittal",
            "slice": x_idx,
            "t1c": type_volume[0, x_idx],
            "t2f": type_volume[1, x_idx],
            "seg": mask[x_idx],
            "cam": cam[x_idx],
            "attn": attention[x_idx],
            "brain": brain_3d[x_idx],
        },
        {
            "name": "Coronal",
            "slice": y_idx,
            "t1c": type_volume[0, :, y_idx, :],
            "t2f": type_volume[1, :, y_idx, :],
            "seg": mask[:, y_idx],
            "cam": cam[:, y_idx],
            "attn": attention[:, y_idx],
            "brain": brain_3d[:, y_idx],
        },
        {
            "name": "Axial",
            "slice": z_idx,
            "t1c": type_volume[0, :, :, z_idx],
            "t2f": type_volume[1, :, :, z_idx],
            "seg": mask[:, :, z_idx],
            "cam": cam[:, :, z_idx],
            "attn": attention[:, :, z_idx],
            "brain": brain_3d[:, :, z_idx],
        },
    ]
    titles = ["T1c", "T2F / FLAIR", "Segmentation", "Grad-CAM++ + Segmentation", "Guided Attention"]

    with plt.rc_context(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "savefig.facecolor": "#0e1117",
            "text.color": "#dddddd",
            "axes.labelcolor": "#dddddd",
            "xtick.color": "#dddddd",
            "ytick.color": "#dddddd",
        }
    ):
        fig = plt.figure(figsize=(19, 10.5), facecolor="#0e1117", dpi=140)
        gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.08, wspace=0.045, top=0.86, bottom=0.045, left=0.035, right=0.99)
        fig.suptitle(
            "Same-Slice AI Summary: MRI, Segmentation, Grad-CAM++, and Guided Attention",
            color="#e8f4ff",
            fontsize=15,
            y=0.985,
            fontweight="bold",
        )

        for row, view in enumerate(views):
            t1c = rotate_show(display_slice_exact(view["t1c"], view["brain"]))
            t2f = rotate_show(display_slice_exact(view["t2f"], view["brain"]))
            seg = rotate_show((view["seg"] > 0).astype(np.float32))
            cam_show = rotate_show(normalize01_exact(view["cam"]) * view["brain"].astype(np.float32))
            attn_show = rotate_show(normalize01_exact(view["attn"]) * view["brain"].astype(np.float32))

            panels = [
                ("gray", t1c),
                ("gray", t2f),
                ("mask", (t1c, seg)),
                ("segcam", (t1c, cam_show, seg)),
                ("attn", (t1c, attn_show)),
            ]

            for col, (kind, data) in enumerate(panels):
                ax = fig.add_subplot(gs[row, col])
                ax.axis("off")
                ax.set_facecolor("#000000")
                if row == 0:
                    ax.set_title(titles[col], color="#aeb6c3", fontsize=10, pad=10, fontweight="bold")

                if kind == "gray":
                    ax.imshow(data, cmap="gray", vmin=0, vmax=1)
                elif kind == "mask":
                    base, seg_show = data
                    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(np.ma.masked_where(seg_show <= 0, seg_show), cmap="winter", alpha=0.62, vmin=0, vmax=1)
                    if np.count_nonzero(seg_show) > 0:
                        ax.contour(seg_show, levels=[0.5], colors=["#7ffcff"], linewidths=1.2)
                elif kind == "segcam":
                    base, cam_img, seg_show = data
                    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(cam_img, cmap="jet", alpha=0.58, vmin=0, vmax=1)
                    if np.count_nonzero(seg_show) > 0:
                        ax.imshow(np.ma.masked_where(seg_show <= 0, seg_show), cmap="winter", alpha=0.28, vmin=0, vmax=1)
                        ax.contour(seg_show, levels=[0.5], colors=["#ffffff"], linewidths=1.25)
                else:
                    base, attn_img = data
                    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(attn_img, cmap=heat_cmap, alpha=0.62, vmin=0, vmax=1)
                    if np.count_nonzero(attn_img) > 0:
                        yy, xx = np.unravel_index(int(np.argmax(attn_img)), attn_img.shape)
                        ax.plot(xx, yy, marker="+", markersize=10, markeredgewidth=1.4, color="white")

                if col == 0:
                    ax.text(
                        0.02,
                        0.97,
                        f"{view['name']} slice {view['slice']}",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        va="top",
                        bbox=dict(facecolor="#0e1117", edgecolor="#00cc66", linewidth=0.8, alpha=0.9, pad=2),
                    )
                if col == 3:
                    ax.text(
                        0.98,
                        0.04,
                        "white outline = segmentation",
                        transform=ax.transAxes,
                        color="#f6fbff",
                        fontsize=7,
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor="#050a10", edgecolor="#6ad7ff", linewidth=0.5, alpha=0.72, pad=2),
                    )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def modality_lines(files: dict[str, Optional[Path]]) -> str:
    lines = [f"- {label}: {path.name}" for label, path in files.items() if path is not None]
    return "\n".join(lines) if lines else "- No modality metadata"


def modality_sentence(files: dict[str, Optional[Path]]) -> str:
    used = [label for label, path in files.items() if path is not None and label != "Segmentation mask"]
    if not used:
        return "Uploaded MRI modality not specified"
    return ", ".join(used)


def clinical_findings(label: str, mask: np.ndarray, spacing: Tuple[float, float, float], patient_id: str) -> dict:
    report_label = "No tumor" if label == "no_tumor" else ("Glioma" if label == "glioma" else "Meningioma")
    if format_clinical_findings is None:
        return {"patient_id": patient_id, "classification_label": report_label}
    return format_clinical_findings(report_label, mask, spacing, patient_id)


def report_text(patient_id: str, label: str, files: dict[str, Optional[Path]], clinical: dict) -> str:
    report_label = clinical.get("classification_label", label)
    tumor_present = bool(clinical.get("tumor_present", False))
    lesion_volume = clinical.get("segmented_lesion_volume_cm3", clinical.get("tumor_volume_cm3", 0.0))
    core_volume = clinical.get("tumor_volume_cm3", 0.0)
    diameter = clinical.get("max_diameter_mm", 0.0)
    axial_span = clinical.get("axial_span_mm", 0.0)
    axial_slices = clinical.get("axial_span_slices", 0)
    hemisphere = clinical.get("hemisphere", "N/A")
    sphericity = clinical.get("sphericity", 0.0)
    bbox = clinical.get("bounding_box_size_mm", None)
    bbox_text = " x ".join(str(x) for x in bbox) + " mm" if bbox else "N/A"
    subregions = clinical.get("subregion_volumes", {}) or {}
    if subregions:
        subregion_text = "; ".join(
            f"{name}: {values.get('volume_cm3', 0)} cm3 ({values.get('percentage', 0)}%)"
            for name, values in subregions.items()
            if isinstance(values, dict)
        )
    else:
        subregion_text = "No subregion volume breakdown available."

    if not tumor_present:
        return f"""Patient: {patient_id}
Volume: 0 cm3

TECHNIQUE:
Automated brain MRI review was performed using the following uploaded modality set: {modality_sentence(files)}. The tumor screening model evaluated the available MRI input. No segmentation-derived tumor measurements were required because the case was routed as no tumor.

COMPARISON:
No prior comparison study was provided.

FINDINGS:
- No intracranial tumor was detected by the automated screening stage.
- The reviewed modality set was documented for traceability: {modality_sentence(files)}.
- No measurable tumor volume, diameter, axial span, or subregion burden is reported.

IMPRESSION:
- No tumor detected by the automated TriNeuro AI workflow.
- Radiologist review remains required before clinical use.

RECOMMENDATIONS:
Correlate with the complete MRI study and clinical context."""

    return f"""Patient: {patient_id}
Volume: {lesion_volume} cm3

TECHNIQUE:
Automated brain MRI review was performed using the following uploaded modality set: {modality_sentence(files)}. The workflow included tumor screening, glioma/meningioma classification when routed to the tumor branch, Grad-CAM++ explainability, segmentation-mask processing, and structured report generation.

COMPARISON:
No prior comparison study was provided.

FINDINGS:
- Automated classification is most consistent with {report_label}.
- Segmented lesion volume is {lesion_volume} cm3, including estimated tumor core volume of {core_volume} cm3.
- Maximum lesion diameter is {diameter} mm, with axial span of {axial_span} mm across {axial_slices} slices.
- Laterality estimate is {hemisphere}. Bounding box dimensions are {bbox_text}.
- Shape analysis shows sphericity of {sphericity}, indicating the degree of lesion irregularity.
- Subregion measurements: {subregion_text}
- Grad-CAM++ visualization was generated to show the classifier focus region.

IMPRESSION:
- Automated MRI pipeline detects a measurable tumor burden most consistent with {report_label}.
- Segmentation-derived measurements support a structured estimate of lesion size, extent, and morphology.
- Findings are decision-support outputs and should not replace radiologist interpretation.

RECOMMENDATIONS:
Review the original MRI sequences, segmentation overlay, and Grad-CAM++ map. Correlate with clinical findings and histopathology when available."""


def build_report(patient_id: str, label: str, files: dict[str, Optional[Path]], clinical: dict) -> tuple[str, str]:
    """Generate the final report with OpenAI gpt-4.1."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    fallback = report_text(patient_id, label, files, clinical)
    if not api_key:
        fallback += (
            "\n\nREPORT GENERATION NOTE:\n"
            "OPENAI_API_KEY is empty. Add your key to the project .env file and restart the FastAPI server. "
            "The final website is configured to use gpt-4.1 for report generation."
        )
        return fallback, "structured_fallback_missing_openai_key"

    try:
        from module4_medgemma_report import OpenAILLM, generate_mri_report_with_graph

        clinical_for_report = dict(clinical)
        clinical_for_report["patient_id"] = patient_id
        clinical_for_report["uploaded_modalities"] = {name: path.name for name, path in files.items() if path is not None}
        clinical_for_report["modality_summary"] = modality_sentence(files)

        draft_llm = OpenAILLM(api_key=api_key, max_output_tokens=900)
        critique_llm = OpenAILLM(api_key=api_key, max_output_tokens=900)
        refine_llm = OpenAILLM(api_key=api_key, max_output_tokens=900)
        result = generate_mri_report_with_graph(
            clinical_for_report,
            report_llm=draft_llm,
            critique_llm=critique_llm,
            refine_llm=refine_llm,
        )
        final_report = (result.get("final_report") or result.get("draft_report") or "").strip()
        if final_report:
            return final_report, "openai_gpt_4_1"
    except Exception as exc:
        fallback += (
            "\n\nREPORT GENERATION NOTE:\n"
            f"gpt-4.1 report generation could not complete locally: {exc}"
        )
        return fallback, "structured_fallback_after_openai_error"

    return fallback, "structured_fallback_after_empty_openai_report"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "TriNeuro AI API", "device": str(device())}


@app.post("/predict")
def predict(
    primary: UploadFile = File(...),
    patient_id: str = Form("TRINEURO_CASE"),
    force_tumor_branch: bool = Form(False),
    fast_mode: bool = Form(True),
    include_full_panel: bool = Form(False),
    include_xai_slices: bool = Form(False),
    t1n: Optional[UploadFile] = File(None),
    t1c: Optional[UploadFile] = File(None),
    t2w: Optional[UploadFile] = File(None),
    t2f: Optional[UploadFile] = File(None),
    mask: Optional[UploadFile] = File(None),
) -> dict:
    started_at = time.perf_counter()
    timings: dict[str, float] = {}

    def mark(stage: str) -> None:
        timings[stage] = round(time.perf_counter() - started_at, 3)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        try:
            files = {
                "Primary MRI": save_upload(primary, folder, "primary.nii.gz"),
                "T1n": save_upload(t1n, folder, "t1n.nii.gz"),
                "T1c": save_upload(t1c, folder, "t1c.nii.gz"),
                "T2w": save_upload(t2w, folder, "t2w.nii.gz"),
                "T2F / FLAIR": save_upload(t2f, folder, "t2f.nii.gz"),
                "Segmentation mask": save_upload(mask, folder, "mask.nii.gz"),
            }
            files, preprocessing_info = preprocess_non_brats_uploads(files, folder)
            mark("uploads_and_preprocessing")
            primary_path = files["Primary MRI"]
            if primary_path is None:
                raise HTTPException(status_code=400, detail="Primary MRI is required.")

            binary_source = files["T2w"] or files["T2F / FLAIR"] or primary_path
            t1c_source = files["T1c"] or primary_path
            t2f_source = files["T2F / FLAIR"] or files["T2w"] or primary_path

            binary_volume, spacing, raw_volume = preprocess_binary(binary_source)
            tumor_result = predict_binary(binary_volume)
            mark("tumor_screen")
            route_to_tumor = force_tumor_branch or tumor_result["label"] == "tumor"
            preview_volume = binary_volume

            tumor_type_result = None
            gradcam_slices = None
            attention_slices = None
            segmentation_source = "not needed"
            mask_volume = np.zeros_like(binary_volume, dtype=np.float32)

            if route_to_tumor:
                type_volume = preprocess_type(t1c_source, t2f_source)
                tumor_type_result = predict_type(type_volume)
                mark("tumor_type")
                cam = compute_gradcam(type_volume)
                mark("gradcam")
                if files["Segmentation mask"] is not None:
                    uploaded_mask, _ = load_nifti(files["Segmentation mask"])
                    mask_volume = resize3d((uploaded_mask > 0).astype(np.float32), BINARY_SHAPE, order=0)
                    preview_volume = preprocess_display_volume(t2f_source)
                    segmentation_source = "uploaded segmentation mask"
                else:
                    seg_input = preprocess_segmentation(files)
                    seg_mask = predict_segmentation(seg_input)
                    mask_volume = resize3d(seg_mask, BINARY_SHAPE, order=0)
                    preview_volume = resize3d(seg_input[3], BINARY_SHAPE, order=1)
                    segmentation_source = "Segmentation/seg_model.pth"
                mark("segmentation")
                type_mask = resize3d(mask_volume, TYPE_SHAPE, order=0)
                attention = compute_learned_attention(type_volume)
                mark("attention")
                preview_count = FAST_SLICE_COUNT if fast_mode else FULL_SLICE_COUNT
                preview_size = FAST_SLICE_SIZE if fast_mode else FULL_SLICE_SIZE
                if include_xai_slices:
                    gradcam_slices = slice_pack(type_volume[0], cam, mode="gradcam", count=preview_count, size=preview_size)
                    attention_slices = slice_pack(type_volume[0], attention, mode="attention", count=preview_count, size=preview_size)
                xai_panel = xai_panel_png(type_volume, cam, type_mask, attention)
                xai_full_panel = xai_full_panel_png(type_volume, cam, type_mask, attention) if include_full_panel else None
                mri_column = None
                mark("xai_images")
            else:
                xai_panel = None
                xai_full_panel = None
                mri_column = None

            final_label = tumor_type_result["label"] if tumor_type_result else tumor_result["label"]
            clinical = clinical_findings(final_label, mask_volume, spacing, patient_id)
            report, report_source = build_report(patient_id, final_label, files, clinical)
            mark("report")
            preview_count = FAST_SLICE_COUNT if fast_mode else FULL_SLICE_COUNT
            preview_size = FAST_SLICE_SIZE if fast_mode else FULL_SLICE_SIZE
            mri_slices = slice_pack(preview_volume, count=preview_count, size=preview_size)
            segmentation_slices = (
                slice_pack(preview_volume, mask_volume, mode="segmentation", count=preview_count, size=preview_size)
                if route_to_tumor
                else None
            )
            mark("slice_previews")

            return {
                "tumor_result": tumor_result,
                "tumor_type_result": tumor_type_result,
                "preprocessing_info": preprocessing_info,
                "segmentation_source": segmentation_source,
                "clinical_findings": clinical,
                "report": report,
                "report_source": report_source,
                "performance": timings,
                "mri_slices": mri_slices,
                "segmentation_slices": segmentation_slices,
                "gradcam_slices": gradcam_slices,
                "attention_slices": attention_slices,
                "xai_panel_png": xai_panel,
                "xai_full_panel_png": xai_full_panel,
                "mri_column_png": mri_column,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
