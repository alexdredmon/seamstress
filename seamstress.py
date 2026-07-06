#!/usr/bin/env python3
"""
Make the cuts between consecutively generated clips of one continuous shot
invisible.

Drop two or more clips (or a directory of clips) on the CLI and it fixes every
seam automatically:

  1. Probes both sides of each seam and auto-conforms resolution / frame rate.
  2. Estimates a guarded global affine alignment (ECC) between the boundary
     frames.
  3. Builds dense per-pixel correspondence with DIS optical flow plus a
     forward/backward consistency check, so color is only measured on pixels
     that truly correspond and genuinely changed elements are detected and
     excluded.
  4. Fits a three-layer color model on the validated correspondences:
     a robust linear-light RGB matrix, smooth per-channel refinement curves,
     and a spatially varying low-frequency residual field (masked, inpainted,
     smoothed) for region-dependent mismatch.
  5. Anchors the seam with corrections that travel with the content: a
     decaying dense geometric morph absorbs subtle element shifts between the
     two generations, and a flow-advected detail residual makes the boundary
     frame match exactly without ghosting on motion.
  6. Re-renders with triangular-PDF dithering and a high-quality encode.

Every stage measures itself and falls back to a safer model when it does not
improve the seam, so no manual tuning is required.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

cv2 = None
np = None

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def load_python_dependencies() -> None:
    global cv2, np
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "missing Python dependency; install requirements with "
            "`python -m pip install -r requirements.txt`"
        ) from exc
    cv2 = cv2_module
    np = np_module


# --------------------------------------------------------------------------
# Probing and IO
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps: float
    frames: int
    duration: float
    has_audio: bool

    @property
    def size_arg(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def rate_arg(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"


def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def ffprobe_video(path: Path) -> VideoInfo:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream found in {path}")
    stream = streams[0]

    def parse_rate(text: str | None) -> tuple[int, int]:
        if not text or "/" not in text:
            return 0, 1
        num, den = text.split("/")
        return int(num), max(1, int(den))

    fps_num, fps_den = parse_rate(stream.get("r_frame_rate"))
    if not fps_num:
        fps_num, fps_den = parse_rate(stream.get("avg_frame_rate"))
    fps = fps_num / fps_den if fps_den else 0.0
    duration = float(stream.get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or 0)
    if not frames:
        frames = int(stream.get("nb_read_packets") or 0)
    if not frames and fps and duration:
        frames = int(round(duration * fps))

    audio = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ]
    )
    has_audio = bool(audio.get("streams"))

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        fps=fps,
        frames=frames,
        duration=duration,
        has_audio=has_audio,
    )


def decode_window(path: Path, info: VideoInfo, which: str, count: int) -> "np.ndarray":
    """Decode the first or last `count` frames as an (n, h, w, 3) RGB array."""
    frame_bytes = info.width * info.height * 3
    if which == "head":
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        raw = subprocess.check_output(cmd)
    elif which == "tail":
        window = (count + 6) / max(info.fps, 1.0)
        raw = b""
        for _ in range(4):
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-sseof",
                f"-{window:.6f}",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            raw = subprocess.check_output(cmd)
            if len(raw) // frame_bytes >= count:
                break
            window *= 2.0
    else:
        raise ValueError(f"unknown window side: {which}")

    n = len(raw) // frame_bytes
    if n < 1:
        raise RuntimeError(f"could not decode {which} frames from {path}")
    frames = np.frombuffer(raw[: n * frame_bytes], np.uint8).reshape(
        n, info.height, info.width, 3
    )
    return frames[:count].copy() if which == "head" else frames[-count:].copy()


def decode_boundary_frame(path: Path, info: VideoInfo, which: str) -> "np.ndarray":
    side = "head" if which == "first" else "tail"
    return decode_window(path, info, side, 1)[0]


def write_png(path: Path, rgb: "np.ndarray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def conform_clip(source: Path, target: VideoInfo, workdir: Path) -> Path:
    """Rescale / retime a clip so it matches the reference clip's geometry."""
    out = workdir / f"{source.stem}_conformed.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"scale={target.width}:{target.height}:flags=lanczos,fps={target.rate_arg}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(out),
        ]
    )
    return out


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

_LIN_LUT = None
_ENC_LUT = None
_DIS_CACHE: dict[str, object] = {}


def linear_lut() -> "np.ndarray":
    """uint8 sRGB value -> linear light 0..1."""
    global _LIN_LUT
    if _LIN_LUT is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        _LIN_LUT = np.where(
            c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4
        ).astype(np.float32)
    return _LIN_LUT


def encode_lut() -> "np.ndarray":
    """linear light 0..1 (16384 steps) -> sRGB 0..255 float."""
    global _ENC_LUT
    if _ENC_LUT is None:
        lin = np.linspace(0.0, 1.0, 16384, dtype=np.float64)
        g = np.where(
            lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055
        )
        _ENC_LUT = (g * 255.0).astype(np.float32)
    return _ENC_LUT


def dis_flow(gray_a: "np.ndarray", gray_b: "np.ndarray") -> "np.ndarray":
    if "dis" not in _DIS_CACHE:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
        _DIS_CACHE["dis"] = dis
    return _DIS_CACHE["dis"].calc(gray_a, gray_b, None)


def flow_between(rgb_a: "np.ndarray", rgb_b: "np.ndarray", work_width: int) -> "np.ndarray":
    """Dense flow such that a(x) corresponds to b(x + flow(x)), full resolution."""
    h, w = rgb_a.shape[:2]
    scale = min(1.0, work_width / w)
    if scale < 1.0:
        size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
        ga = cv2.cvtColor(cv2.resize(rgb_a, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(cv2.resize(rgb_b, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        flow = dis_flow(ga, gb)
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= np.float32([w / size[0], h / size[1]])
    else:
        ga = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
        flow = dis_flow(ga, gb)
    return flow


def make_grid(height: int, width: int) -> tuple["np.ndarray", "np.ndarray"]:
    gx = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    gy = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1)
    return gx, gy


def remap_by_flow(img: "np.ndarray", flow: "np.ndarray", grid: tuple) -> "np.ndarray":
    gx, gy = grid
    return cv2.remap(
        img,
        gx + flow[..., 0],
        gy + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def edge_feather(height: int, width: int, px: int) -> "np.ndarray":
    px = max(1, px)
    ry = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    rx = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    return np.minimum(
        np.clip(ry / px, 0.0, 1.0)[:, None], np.clip(rx / px, 0.0, 1.0)[None, :]
    )


def soft_clip(x: "np.ndarray", cap: float) -> "np.ndarray":
    return (cap * np.tanh(x / cap)).astype(np.float32)


def smoothstep_decay(i: int, n: int) -> float:
    """1.0 at i == 0 easing to 0.0 at i >= n."""
    if n <= 0:
        return 0.0
    t = min(1.0, max(0.0, i / float(n)))
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def delta_e(ref: "np.ndarray", cand: "np.ndarray", mask: "np.ndarray" = None) -> dict:
    """CIE76 delta-E statistics between two RGB images (uint8 or float 0..255)."""
    a = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    b = cv2.cvtColor(np.clip(cand, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    de = np.sqrt(np.sum((a - b) ** 2, axis=2))
    if mask is not None:
        de = de[mask]
    if de.size == 0:
        return {"mean": 0.0, "p95": 0.0}
    return {"mean": float(np.mean(de)), "p95": float(np.percentile(de, 95))}


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def highpass_gray(rgb: "np.ndarray") -> "np.ndarray":
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    return np.clip(gray - blur + 0.5, 0.0, 1.0).astype(np.float32)


def margin_mask(height: int, width: int, margin: int) -> "np.ndarray":
    mask = np.zeros((height, width), np.uint8)
    margin = max(0, min(margin, min(height, width) // 3))
    mask[margin : height - margin, margin : width - margin] = 255
    return mask


def affine_is_sane(matrix: "np.ndarray", width: int, height: int) -> bool:
    return (
        abs(matrix[0, 0] - 1.0) < 0.12
        and abs(matrix[1, 1] - 1.0) < 0.12
        and abs(matrix[0, 1]) < 0.12
        and abs(matrix[1, 0]) < 0.12
        and abs(matrix[0, 2]) < 0.10 * width
        and abs(matrix[1, 2]) < 0.10 * height
    )


def estimate_affine(
    reference: "np.ndarray",
    source: "np.ndarray",
    mask_margin: int,
    ecc_width: int,
) -> tuple["np.ndarray", float, str]:
    """Guarded global alignment: affine ECC, translation fallback, identity fallback."""
    h, w = reference.shape[:2]
    scale = min(1.0, ecc_width / w)
    if scale < 1.0:
        size = (int(round(w * scale)), int(round(h * scale)))
        ref_s = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        src_s = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    else:
        ref_s, src_s = reference, source
    mask_s = margin_mask(ref_s.shape[0], ref_s.shape[1], int(round(mask_margin * scale)))
    ref_hp = highpass_gray(ref_s)
    src_hp = highpass_gray(src_s)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1500, 1e-8)

    for motion, label in ((cv2.MOTION_AFFINE, "affine"), (cv2.MOTION_TRANSLATION, "translation")):
        try:
            init = np.eye(2, 3, dtype=np.float32)
            cc, matrix = cv2.findTransformECC(
                ref_hp, src_hp, init, motion, criteria, mask_s, 5
            )
        except cv2.error:
            continue
        matrix = matrix.copy()
        matrix[0, 2] /= scale
        matrix[1, 2] /= scale
        if affine_is_sane(matrix, w, h):
            return matrix.astype(np.float32), float(cc), label
    return np.eye(2, 3, dtype=np.float32), 0.0, "identity"


def warp_frame(frame: "np.ndarray", matrix: "np.ndarray", width: int, height: int) -> "np.ndarray":
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


# --------------------------------------------------------------------------
# Color model
# --------------------------------------------------------------------------


@dataclass
class ColorModel:
    """Linear-light RGB matrix followed by smooth per-channel gamma curves."""

    matrix: "np.ndarray"  # (4, 3): rows are R, G, B gains plus bias, linear light
    luts: "np.ndarray"  # (3, 1024): refinement curves over gamma 0..255

    @classmethod
    def identity(cls) -> "ColorModel":
        matrix = np.zeros((4, 3), np.float32)
        matrix[:3, :] = np.eye(3, dtype=np.float32)
        domain = np.linspace(0.0, 255.0, 1024, dtype=np.float32)
        return cls(matrix=matrix, luts=np.stack([domain] * 3))

    def apply(self, frame_u8: "np.ndarray") -> "np.ndarray":
        """uint8 RGB frame -> corrected float32 RGB in 0..255 gamma space."""
        lin = linear_lut()[frame_u8].reshape(-1, 3)
        out = lin @ self.matrix[:3, :] + self.matrix[3, :]
        np.clip(out, 0.0, 1.0, out)
        gamma = encode_lut()[(out * 16383.0 + 0.5).astype(np.int32)]
        idx = np.clip(gamma * (1023.0 / 255.0) + 0.5, 0, 1023).astype(np.int32)
        for c in range(3):
            gamma[:, c] = self.luts[c][idx[:, c]]
        return gamma.reshape(frame_u8.shape).astype(np.float32)


def fit_color_matrix(src_lin: "np.ndarray", ref_lin: "np.ndarray") -> "np.ndarray":
    """Robust trimmed least squares for the linear-light matrix, (N,3) inputs."""
    identity = np.zeros((4, 3), np.float64)
    identity[:3, :] = np.eye(3)
    if src_lin.shape[0] < 5000:
        return identity.astype(np.float32)
    src = np.concatenate([src_lin, np.ones((src_lin.shape[0], 1), src_lin.dtype)], axis=1)
    ref = ref_lin
    usable = np.ones(src.shape[0], bool)
    selected = usable.copy()
    matrix = identity
    for _ in range(5):
        matrix = np.linalg.lstsq(src[selected], ref[selected], rcond=None)[0]
        residual = np.sqrt(np.mean((src @ matrix - ref) ** 2, axis=1))
        threshold = np.percentile(residual[usable], 80.0)
        selected = usable & (residual <= threshold)
        if selected.sum() < 5000:
            break
    return matrix.astype(np.float32)


def fit_refinement_luts(
    src_gamma: "np.ndarray",
    ref_gamma: "np.ndarray",
    weights: "np.ndarray",
    bins: int = 64,
    max_shift: float = 16.0,
) -> "np.ndarray":
    """Smooth confidence-weighted per-channel curves, (N,3) gamma inputs."""
    domain = np.linspace(0.0, 255.0, 1024, dtype=np.float32)
    luts = np.stack([domain.copy() for _ in range(3)])
    centers = (np.arange(bins) + 0.5) * (255.0 / bins)
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    kernel /= kernel.sum()
    for c in range(3):
        idx = np.clip(src_gamma[:, c] * (bins / 255.0), 0, bins - 1e-3).astype(np.int64)
        count = np.bincount(idx, weights=weights, minlength=bins)
        total = np.bincount(idx, weights=weights * ref_gamma[:, c], minlength=bins)
        mean = np.where(count > 1e-3, total / np.maximum(count, 1e-3), centers)
        confidence = count / (count + 300.0)
        delta = confidence * (mean - centers)
        delta = np.convolve(np.pad(delta, 4, mode="edge"), kernel, mode="valid")
        delta = np.clip(delta, -max_shift, max_shift)
        curve = domain + np.interp(domain, centers, delta)
        curve = np.maximum.accumulate(curve)
        luts[c] = np.clip(curve, 0.0, 255.0)
    return luts.astype(np.float32)


# --------------------------------------------------------------------------
# Seam analysis
# --------------------------------------------------------------------------


@dataclass
class SeamCorrection:
    affine: "np.ndarray"
    morph_flow: "np.ndarray"  # (h, w, 2); frame0(x) = warped(x + s * flow(x))
    color: ColorModel
    field: "np.ndarray"  # (h, w, 3) low-frequency residual, gamma 0..255
    detail: "np.ndarray"  # (h, w, 3) masked detail residual at the boundary
    valid_fraction: float
    stats: dict = dataclass_field(default_factory=dict)


def build_lowfreq_field(
    residual: "np.ndarray",
    weight: "np.ndarray",
    cell: int = 32,
    cap: float = 30.0,
) -> "np.ndarray":
    """Weighted coarse residual grid, diffusion-inpainted, smoothed, upsampled."""
    h, w = residual.shape[:2]
    gw = max(4, int(round(w / cell)))
    gh = max(4, int(round(h / cell)))
    weight_c = cv2.resize(weight, (gw, gh), interpolation=cv2.INTER_AREA)
    values_c = cv2.resize(residual * weight[..., None], (gw, gh), interpolation=cv2.INTER_AREA)
    known = weight_c > 0.02
    values = np.where(
        known[..., None], values_c / np.maximum(weight_c, 1e-6)[..., None], 0.0
    ).astype(np.float32)
    mask = known.astype(np.float32)
    filled = values.copy()
    for _ in range(200):
        vb = cv2.GaussianBlur(filled * mask[..., None], (0, 0), 1.5)
        mb = cv2.GaussianBlur(mask, (0, 0), 1.5)
        update = mb > 1e-4
        spread = vb / np.maximum(mb, 1e-6)[..., None]
        filled = np.where(known[..., None], values, np.where(update[..., None], spread, filled))
        mask = np.minimum(1.0, mb * 4.0 + mask)
        if bool(np.all(mask > 1e-4)):
            filled = np.where(known[..., None], values, spread)
            break
    filled = cv2.GaussianBlur(filled, (0, 0), 1.0)
    field = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    return soft_clip(field, cap)


def analyze_seam(
    reference: "np.ndarray",
    source: "np.ndarray",
    mask_margin: int,
    ecc_width: int,
) -> SeamCorrection:
    """Fit every correction layer from one boundary frame pair.

    `reference` is the frame the altered clip must continue from; `source` is
    the altered clip's boundary frame. All fields are expressed in the
    reference geometry that rendering reproduces at the cut.
    """
    h, w = reference.shape[:2]
    grid = make_grid(h, w)
    stats: dict = {}

    affine, ecc_cc, ecc_model = estimate_affine(reference, source, mask_margin, ecc_width)
    warped = warp_frame(source, affine, w, h)
    stats["alignment"] = {"model": ecc_model, "ecc": ecc_cc, "matrix": affine.tolist()}

    # Dense correspondence with forward/backward consistency.
    flow_rs = flow_between(reference, warped, 960)  # ref(x) ~ warped(x + flow_rs)
    flow_sr = flow_between(warped, reference, 960)
    flow_sr_at = remap_by_flow(flow_sr, flow_rs, grid)
    fb_err = np.linalg.norm(flow_rs + flow_sr_at, axis=2)
    fb_mag = np.linalg.norm(flow_rs, axis=2) + np.linalg.norm(flow_sr_at, axis=2)
    consistent = fb_err < (1.0 + 0.05 * fb_mag)
    interior = margin_mask(h, w, max(8, mask_margin // 2)).astype(bool)
    valid = consistent & interior
    valid_fraction = float(valid.mean())
    stats["dense"] = {
        "valid_fraction": valid_fraction,
        "median_fb_error": float(np.median(fb_err[interior])),
    }

    morph_ok = valid_fraction > 0.30 and stats["dense"]["median_fb_error"] < 4.0
    if morph_ok:
        # Gated, smoothed, capped morph field: pulls the boundary frame onto
        # the reference geometry, easing out over the first frames.
        weight_soft = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 4.0)
        morph = flow_rs * weight_soft[..., None]
        morph[..., 0] = cv2.GaussianBlur(morph[..., 0], (0, 0), 9.0)
        morph[..., 1] = cv2.GaussianBlur(morph[..., 1], (0, 0), 9.0)
        smooth_w = cv2.GaussianBlur(weight_soft, (0, 0), 9.0)
        morph /= np.maximum(smooth_w, 0.25)[..., None]
        mag = np.linalg.norm(morph, axis=2)
        morph *= np.minimum(1.0, 14.0 / np.maximum(mag, 1e-6))[..., None]
        morph *= edge_feather(h, w, 24)[..., None]
        morph = morph.astype(np.float32)
    else:
        morph = np.zeros((h, w, 2), np.float32)
    stats["morph_enabled"] = bool(morph_ok)

    # Color pairs from exact dense correspondences.
    aligned_src = remap_by_flow(warped, flow_rs, grid)  # source content, reference geometry
    unsaturated = (
        ((aligned_src > 3) & (aligned_src < 252)).all(axis=2)
        & ((reference > 3) & (reference < 252)).all(axis=2)
    )
    fit_mask = valid & unsaturated
    if fit_mask.sum() < 20000:
        fit_mask = interior & unsaturated
        if fit_mask.sum() < 20000:
            fit_mask = interior

    stats["seam_raw"] = delta_e(reference, aligned_src, valid if valid.any() else interior)

    color = ColorModel.identity()
    src_lin = linear_lut()[aligned_src[fit_mask]].astype(np.float64)
    ref_lin = linear_lut()[reference[fit_mask]].astype(np.float64)
    matrix = fit_color_matrix(src_lin, ref_lin)
    candidate = ColorModel(matrix=matrix, luts=color.luts.copy())
    eval_mask = valid if valid.any() else interior
    after_matrix = candidate.apply(aligned_src)
    stats["seam_after_matrix"] = delta_e(reference, after_matrix, eval_mask)
    if stats["seam_after_matrix"]["mean"] <= stats["seam_raw"]["mean"] + 1e-6:
        color = candidate
        matched = after_matrix
    else:
        matched = color.apply(aligned_src)
        stats["seam_after_matrix"] = dict(stats["seam_raw"])
        stats["matrix_reverted"] = True

    luts = fit_refinement_luts(
        matched[fit_mask],
        reference[fit_mask].astype(np.float32),
        weights=np.ones(int(fit_mask.sum()), np.float32),
    )
    candidate = ColorModel(matrix=color.matrix.copy(), luts=luts)
    after_luts = candidate.apply(aligned_src)
    stats["seam_after_curves"] = delta_e(reference, after_luts, eval_mask)
    if stats["seam_after_curves"]["mean"] <= stats["seam_after_matrix"]["mean"] + 1e-6:
        color = candidate
    else:
        stats["seam_after_curves"] = dict(stats["seam_after_matrix"])
        stats["curves_reverted"] = True

    # Residuals are measured in the exact geometry rendering produces at the
    # cut: global warp followed by the full-strength morph.
    boundary_geometry = remap_by_flow(warped, morph, grid)
    corrected = color.apply(boundary_geometry)
    residual = reference.astype(np.float32) - corrected
    residual_mag = np.linalg.norm(residual, axis=2)
    change_gate = np.exp(-((residual_mag / 45.0) ** 2)).astype(np.float32)
    weight = valid.astype(np.float32) * change_gate

    field = build_lowfreq_field(residual, weight)
    after_field = corrected + field
    stats["seam_after_field"] = delta_e(reference, after_field, eval_mask)
    if stats["seam_after_field"]["mean"] > stats["seam_after_curves"]["mean"] + 1e-6:
        field = np.zeros_like(field)
        after_field = corrected
        stats["seam_after_field"] = dict(stats["seam_after_curves"])
        stats["field_reverted"] = True

    detail_weight = cv2.GaussianBlur(weight, (0, 0), 2.0)
    detail_weight *= edge_feather(h, w, 16)
    detail = ((reference.astype(np.float32) - after_field) * detail_weight[..., None]).astype(
        np.float32
    )
    stats["seam_anchored"] = delta_e(reference, after_field + detail, eval_mask)
    stats["changed_content_fraction"] = float(
        np.mean((~consistent | (change_gate < 0.4)) & interior) / max(interior.mean(), 1e-6)
    )

    return SeamCorrection(
        affine=affine,
        morph_flow=morph,
        color=color,
        field=field.astype(np.float32),
        detail=detail,
        valid_fraction=valid_fraction,
        stats=stats,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlendWindows:
    morph: int
    detail: int
    lowfreq: int


def blend_windows(fps: float, total_frames: int, blend_seconds: float) -> BlendWindows:
    fps = max(fps, 1.0)
    cap = max(1, total_frames - 1)

    def frames(seconds: float, minimum: int) -> int:
        return min(cap, max(minimum, int(round(seconds * fps))))

    return BlendWindows(
        morph=frames(0.40 * blend_seconds, 2),
        detail=frames(0.33 * blend_seconds, 2),
        lowfreq=frames(1.00 * blend_seconds, 4),
    )


def build_encoder_cmd(
    output: Path, source: Path, info: VideoInfo, crf: int, preset: str
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        info.size_arg,
        "-r",
        info.rate_arg,
        "-i",
        "-",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]


class DitherBank:
    """Rolled triangular-PDF dither so gradients quantize without banding."""

    def __init__(self, height: int, width: int, seed: int):
        rng = np.random.default_rng(seed)
        self.noise = (
            rng.random((height, width, 3), dtype=np.float32)
            + rng.random((height, width, 3), dtype=np.float32)
            - 1.0
        )
        self.rng = rng

    def frame(self) -> "np.ndarray":
        return np.roll(
            self.noise,
            (int(self.rng.integers(0, self.noise.shape[0])), int(self.rng.integers(0, self.noise.shape[1]))),
            axis=(0, 1),
        )


def advect_gray(frame_u8: "np.ndarray") -> "np.ndarray":
    h, w = frame_u8.shape[:2]
    scale = min(1.0, 512.0 / w)
    size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
    return cv2.cvtColor(
        cv2.resize(frame_u8, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
    )


def advect_flow_full(gray_from: "np.ndarray", gray_to: "np.ndarray", width: int, height: int) -> "np.ndarray":
    flow = dis_flow(gray_from, gray_to)
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow *= np.float32([width / gray_from.shape[1], height / gray_from.shape[0]])
    return flow


def render_corrected_clip(
    source: Path,
    output: Path,
    info: VideoInfo,
    correction: SeamCorrection,
    windows: BlendWindows,
    anchor_side: str,
    total_frames: int,
    crf: int,
    preset: str,
    seed: int,
) -> int:
    """Stream-decode, correct, dither, and re-encode the altered clip.

    The global alignment and color model apply to every frame. The morph and
    low-frequency field ease out from the cut on a smoothstep schedule. The
    detail residual is advected with per-frame optical flow so the exact-match
    anchor travels with the content instead of ghosting.
    """
    h, w = info.height, info.width
    grid = make_grid(h, w)
    dither = DitherBank(h, w, seed)
    has_morph = bool(np.any(correction.morph_flow))

    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        build_encoder_cmd(output, source, info, crf, preset),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    frame_bytes = w * h * 3

    def strength(index: int, window: int) -> float:
        offset = index if anchor_side == "start" else (total_frames - 1 - index)
        return smoothstep_decay(offset, window)

    def quantize(frame_float: "np.ndarray") -> bytes:
        stamped = np.clip(frame_float + dither.frame(), 0.0, 255.0)
        return (stamped + 0.5).clip(0.0, 255.49).astype(np.uint8).tobytes()

    detail_state = correction.detail.copy()
    prev_gray = None
    tail_buffer: list[tuple[int, "np.ndarray", "np.ndarray"]] = []
    frame_index = 0

    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"partial frame from decoder at frame {frame_index}")
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)

            geo = warp_frame(frame, correction.affine, w, h)
            s_morph = strength(frame_index, windows.morph) if has_morph else 0.0
            if s_morph > 0.0:
                geo = remap_by_flow(geo, correction.morph_flow * np.float32(s_morph), grid)

            out = correction.color.apply(geo)
            s_low = strength(frame_index, windows.lowfreq)
            if s_low > 0.0:
                out += correction.field * np.float32(s_low)

            s_detail = strength(frame_index, windows.detail)
            if anchor_side == "start":
                if s_detail > 0.0:
                    gray = advect_gray(geo)
                    if frame_index > 0:
                        flow = advect_flow_full(gray, prev_gray, w, h)
                        detail_state = remap_by_flow(detail_state, flow, grid)
                    prev_gray = gray
                    out += detail_state * np.float32(s_detail)
                encoder.stdin.write(quantize(out))
            else:
                if s_detail > 0.0:
                    tail_buffer.append((frame_index, out.astype(np.float16), advect_gray(geo)))
                else:
                    encoder.stdin.write(quantize(out))
            frame_index += 1

        if anchor_side == "end" and tail_buffer:
            # Walk backward from the final frame, advecting the detail
            # residual with frame-to-frame flow, then emit in order.
            staged: dict[int, "np.ndarray"] = {}
            detail_state = correction.detail.copy()
            for pos in range(len(tail_buffer) - 1, -1, -1):
                index, out_half, gray = tail_buffer[pos]
                if pos < len(tail_buffer) - 1:
                    flow = advect_flow_full(gray, tail_buffer[pos + 1][2], w, h)
                    detail_state = remap_by_flow(detail_state, flow, grid)
                s_detail = strength(index, windows.detail)
                staged[index] = out_half.astype(np.float32) + detail_state * np.float32(s_detail)
            for index in sorted(staged):
                encoder.stdin.write(quantize(staged[index]))
    except BrokenPipeError as exc:
        raise RuntimeError("encoder closed before all frames were written") from exc
    finally:
        decoder.stdout.close()
        encoder.stdin.close()

    decoder_status = decoder.wait()
    encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    encoder_status = encoder.wait()
    if decoder_status:
        raise RuntimeError(f"decoder failed with exit code {decoder_status}")
    if encoder_status:
        raise RuntimeError(f"encoder failed with exit code {encoder_status}\n{encoder_stderr}")
    return frame_index


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def write_seam_preview(
    timeline_prev: Path,
    timeline_prev_original: Path,
    timeline_next: Path,
    timeline_next_original: Path,
    out_path: Path,
    fps: float,
    seconds: float = 0.7,
) -> None:
    """Before/after seam preview: original seam on top, corrected below, 3x slow."""
    prev_info = ffprobe_video(timeline_prev)
    next_info = ffprobe_video(timeline_next)
    prev_orig_info = ffprobe_video(timeline_prev_original)
    next_orig_info = ffprobe_video(timeline_next_original)
    count = max(4, int(round(seconds * fps)))
    after = list(decode_window(timeline_prev, prev_info, "tail", count)) + list(
        decode_window(timeline_next, next_info, "head", count)
    )
    before = list(decode_window(timeline_prev_original, prev_orig_info, "tail", count)) + list(
        decode_window(timeline_next_original, next_orig_info, "head", count)
    )
    n = min(len(before), len(after))
    before, after = before[:n], after[:n]

    h, w = before[0].shape[:2]
    scale = min(1.0, 960.0 / w)
    size = (int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2)
    separator = np.full((6, size[0], 3), 96, np.uint8)

    out_fps = max(4.0, fps / 3.0)
    out_h = size[1] * 2 + 6
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size[0]}x{out_h}",
            "-r",
            f"{out_fps:.4f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    try:
        for top, bottom in zip(before, after):
            top_r = cv2.resize(top, size, interpolation=cv2.INTER_AREA)
            bottom_r = cv2.resize(bottom, size, interpolation=cv2.INTER_AREA)
            encoder.stdin.write(np.vstack([top_r, separator, bottom_r]).tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("seam preview encoding failed")


def write_diagnostics(
    diagnostics: Path,
    reference: "np.ndarray",
    source: "np.ndarray",
    correction: SeamCorrection,
    output_boundary: "np.ndarray",
) -> None:
    write_png(diagnostics / "01_reference_boundary.png", reference)
    write_png(diagnostics / "02_source_boundary.png", source)
    h, w = reference.shape[:2]
    grid = make_grid(h, w)
    warped = warp_frame(source, correction.affine, w, h)
    morphed = remap_by_flow(warped, correction.morph_flow, grid)
    corrected = np.clip(correction.color.apply(morphed) + correction.field, 0, 255).astype(np.uint8)
    write_png(diagnostics / "03_source_aligned.png", morphed)
    write_png(diagnostics / "04_source_corrected.png", corrected)
    write_png(diagnostics / "05_output_boundary_encoded.png", output_boundary)
    absdiff = np.abs(reference.astype(np.int16) - output_boundary.astype(np.int16))
    write_png(
        diagnostics / "06_encoded_absdiff_amplified.png",
        np.clip(absdiff * 6, 0, 255).astype(np.uint8),
    )
    field_vis = np.clip(correction.field * 4 + 128, 0, 255).astype(np.uint8)
    write_png(diagnostics / "07_lowfreq_field_amplified.png", field_vis)


# --------------------------------------------------------------------------
# Seam driver
# --------------------------------------------------------------------------


@dataclass
class SeamResult:
    output: Path
    report: dict


def process_seam(
    previous: Path,
    next_clip: Path,
    alter: str,
    output: Path,
    diagnostics_dir: Path | None,
    args: argparse.Namespace,
    workdir: Path,
) -> SeamResult:
    altered = next_clip if alter == "next" else previous
    anchor_clip = previous if alter == "next" else next_clip

    anchor_info = ffprobe_video(anchor_clip)
    altered_info = ffprobe_video(altered)

    conformed = None
    needs_conform = (
        (anchor_info.width, anchor_info.height) != (altered_info.width, altered_info.height)
        or abs(anchor_info.fps - altered_info.fps) > 1e-6
        or altered_info.width % 2
        or altered_info.height % 2
    )
    if needs_conform:
        target = anchor_info
        if target.width % 2 or target.height % 2:
            raise RuntimeError(
                f"reference clip {anchor_clip.name} has odd dimensions; "
                "re-export it with even width and height"
            )
        print(f"  conforming {altered.name} to {target.size_arg} @ {target.rate_arg} fps")
        conformed = conform_clip(altered, target, workdir)
        altered = conformed
        altered_info = ffprobe_video(altered)

    if alter == "next":
        reference = decode_boundary_frame(anchor_clip, anchor_info, "last")
        source_frame = decode_boundary_frame(altered, altered_info, "first")
        anchor_side = "start"
    else:
        reference = decode_boundary_frame(anchor_clip, anchor_info, "first")
        source_frame = decode_boundary_frame(altered, altered_info, "last")
        anchor_side = "end"

    correction = analyze_seam(reference, source_frame, args.mask_margin, args.ecc_width)
    stats = correction.stats
    print(
        f"  alignment: {stats['alignment']['model']} (ecc {stats['alignment']['ecc']:.4f}), "
        f"dense valid {correction.valid_fraction * 100:.1f}%, "
        f"changed content {stats['changed_content_fraction'] * 100:.1f}%"
    )
    print(
        "  seam dE mean: raw {raw:.2f} -> matrix {m:.2f} -> curves {c:.2f} -> "
        "field {f:.2f} -> anchored {a:.2f}".format(
            raw=stats["seam_raw"]["mean"],
            m=stats["seam_after_matrix"]["mean"],
            c=stats["seam_after_curves"]["mean"],
            f=stats["seam_after_field"]["mean"],
            a=stats["seam_anchored"]["mean"],
        )
    )

    windows = blend_windows(altered_info.fps, altered_info.frames, args.blend_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_written = render_corrected_clip(
        altered,
        output,
        altered_info,
        correction,
        windows,
        anchor_side,
        altered_info.frames,
        args.crf,
        args.preset,
        args.seed,
    )
    print(f"  rendered {frames_written} frames -> {output}")

    output_info = ffprobe_video(output)
    output_boundary = decode_boundary_frame(
        output, output_info, "first" if anchor_side == "start" else "last"
    )
    encoded = delta_e(reference, output_boundary)
    print(f"  encoded boundary dE mean {encoded['mean']:.2f}, p95 {encoded['p95']:.2f}")

    if diagnostics_dir is not None:
        write_diagnostics(diagnostics_dir, reference, source_frame, correction, output_boundary)
        preview = diagnostics_dir / "seam_preview_before_after.mp4"
        if alter == "next":
            write_seam_preview(previous, previous, output, next_clip, preview, altered_info.fps)
        else:
            write_seam_preview(output, previous, next_clip, next_clip, preview, altered_info.fps)
        print(f"  diagnostics: {diagnostics_dir}")

    report = {
        "previous": str(previous),
        "next": str(next_clip),
        "alter": alter,
        "output": str(output),
        "conformed_input": str(conformed) if conformed else None,
        "frames_written": frames_written,
        "windows": {
            "morph_frames": windows.morph,
            "detail_frames": windows.detail,
            "lowfreq_frames": windows.lowfreq,
        },
        "analysis": stats,
        "encoded_boundary_delta_e": encoded,
        "settings": {
            "mask_margin": args.mask_margin,
            "ecc_width": args.ecc_width,
            "blend_seconds": args.blend_seconds,
            "crf": args.crf,
            "preset": args.preset,
            "seed": args.seed,
        },
    }
    return SeamResult(output=output, report=report)


def concat_clips(paths: list[Path], out: Path, crf: int, preset: str) -> None:
    infos = [ffprobe_video(p) for p in paths]
    with_audio = all(info.has_audio for info in infos)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    n = len(paths)
    if with_audio:
        streams = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
        cmd += [
            "-filter_complex",
            f"{streams}concat=n={n}:v=1:a=1[v][a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    else:
        streams = "".join(f"[{i}:v:0]" for i in range(n))
        cmd += ["-filter_complex", f"{streams}concat=n={n}:v=1:a=0[v]", "-map", "[v]"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.check_call(cmd)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def expand_clip_args(items: list[Path], sort: str) -> list[Path]:
    clips: list[Path] = []
    for item in items:
        path = item.expanduser().resolve()
        if path.is_dir():
            found = sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in VIDEO_SUFFIXES
                and p.is_file()
                and not p.name.startswith(".")
                and not p.stem.endswith(("_seamless", "_conformed"))
            )
            if not found:
                raise RuntimeError(f"no video files found in directory: {path}")
            clips.extend(found)
        else:
            if not path.is_file():
                raise RuntimeError(f"not found: {path}")
            clips.append(path)
    if sort == "name":
        clips.sort(key=lambda p: p.name)
    elif sort == "mtime":
        clips.sort(key=lambda p: p.stat().st_mtime)
    return clips


def default_output_path(source: Path, outdir: Path | None) -> Path:
    name = f"{source.stem}_seamless{source.suffix if source.suffix.lower() == '.mp4' else '.mp4'}"
    return (outdir or source.parent) / name


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seamstress.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Make the cuts between consecutively generated clips of one "
            "continuous shot invisible. Fully automatic: drop two or more "
            "clips (or a directory) and every seam is analyzed and fixed."
        ),
        epilog="""\
Typical use:
  ./seamstress.py previous.mp4 next.mp4

Fix every seam in a whole take and write the stitched result:
  ./seamstress.py clip01.mp4 clip02.mp4 clip03.mp4 --concat final.mp4

Drop a directory (clips are taken in name order):
  ./seamstress.py my_take_folder/ --concat final.mp4

Preserve the next clip and alter the previous clip's ending instead:
  ./seamstress.py previous.mp4 next.mp4 --alter previous

No configuration is required: alignment, dense correspondence, the color
model, the spatial correction field, and the seam blend windows are all
estimated per seam, and each stage falls back to a safer model when it does
not measurably improve the seam.
""",
    )
    parser.add_argument(
        "clips",
        nargs="+",
        type=Path,
        help=(
            "Two or more clips in timeline order, or directories of clips "
            "(directory contents are added in name order)."
        ),
    )
    parser.add_argument(
        "--alter",
        choices=("next", "previous"),
        default="next",
        help=(
            "Which side of the seam to re-render (two-clip mode only). "
            "Default: next."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for the corrected clip (two-clip mode only).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Directory for corrected clips and reports. Default: next to each input.",
    )
    parser.add_argument(
        "--concat",
        type=Path,
        help="Also write the full stitched timeline (first clip plus corrected clips).",
    )
    parser.add_argument(
        "--sort",
        choices=("given", "name", "mtime"),
        default="given",
        help=(
            "Order for the clip list. 'given' keeps the command-line order "
            "(directories always expand in name order). Default: given."
        ),
    )
    parser.add_argument(
        "--blend-seconds",
        type=float,
        default=1.0,
        help=(
            "Overall duration scale for easing the seam corrections out. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=10,
        help="libx264 CRF for corrected clips. Default: 10.",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="libx264 preset for corrected clips. Default: slow.",
    )
    parser.add_argument(
        "--mask-margin",
        type=int,
        default=32,
        help="Pixels ignored at each frame edge during analysis. Default: 32.",
    )
    parser.add_argument(
        "--ecc-width",
        type=int,
        default=960,
        help="Working width for global ECC alignment. Default: 960.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Dither noise seed. Default: 7.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip diagnostic PNGs and the before/after seam preview video.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")
    load_python_dependencies()

    clips = expand_clip_args(args.clips, args.sort)
    if len(clips) < 2:
        raise RuntimeError("need at least two clips (or a directory containing them)")
    if len(clips) > 2 and args.alter == "previous":
        raise RuntimeError("--alter previous is only supported with exactly two clips")
    if args.output and len(clips) > 2:
        raise RuntimeError("use --outdir instead of --output when fixing more than one seam")

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    results: list[SeamResult] = []
    with tempfile.TemporaryDirectory(prefix="seamstress_") as tmp:
        workdir = Path(tmp)
        if len(clips) == 2 and args.alter == "previous":
            previous, next_clip = clips
            output = (
                args.output.expanduser().resolve()
                if args.output
                else default_output_path(previous, args.outdir)
            )
            if output in (previous, next_clip):
                raise RuntimeError("output must be different from both input paths")
            print(f"seam 1/1: {previous.name} <- {next_clip.name} (altering previous)")
            diagnostics = None if args.no_diagnostics else output.with_name(f"{output.stem}_diagnostics")
            result = process_seam(previous, next_clip, "previous", output, diagnostics, args, workdir)
            results.append(result)
            timeline = [output, next_clip]
        else:
            timeline = [clips[0]]
            total = len(clips) - 1
            for index in range(1, len(clips)):
                previous = timeline[-1]
                next_clip = clips[index]
                output = (
                    args.output.expanduser().resolve()
                    if args.output and len(clips) == 2
                    else default_output_path(next_clip, args.outdir)
                )
                if output in (previous, next_clip):
                    raise RuntimeError("output must be different from both input paths")
                print(f"seam {index}/{total}: {previous.name} -> {next_clip.name}")
                diagnostics = (
                    None if args.no_diagnostics else output.with_name(f"{output.stem}_diagnostics")
                )
                result = process_seam(previous, next_clip, "next", output, diagnostics, args, workdir)
                results.append(result)
                timeline.append(output)

        for result in results:
            report_path = result.output.with_name(f"{result.output.stem}_report.json")
            report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")

        if args.concat:
            concat_path = args.concat.expanduser().resolve()
            print(f"stitching {len(timeline)} clips -> {concat_path}")
            concat_clips(timeline, concat_path, args.crf, args.preset)

    print("done:")
    for result in results:
        seam = result.report["encoded_boundary_delta_e"]
        raw = result.report["analysis"]["seam_raw"]
        print(
            f"  {Path(result.report['output']).name}: seam dE {raw['mean']:.2f} -> {seam['mean']:.2f}"
        )
    if args.concat:
        print(f"  stitched timeline: {args.concat}")
    return 0


def main(argv: list[str]) -> int:
    try:
        return run(parse_args(argv))
    except subprocess.CalledProcessError as exc:
        cmd = exc.cmd if isinstance(exc.cmd, str) else " ".join(str(c) for c in exc.cmd)
        print(f"error: command failed with exit code {exc.returncode}: {cmd}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
