#!/usr/bin/env python3
"""
Detect and repair brief geometric glitches inside a single clip.

Seamstress makes the cut between two clips seamless; stabilizer fixes glitches
within one clip: sudden scale pops, position shifts, and stutter/frame-jump
artifacts that break an otherwise smooth continuous shot. Only frames inside a
repaired neighborhood are altered, so the clip keeps its content and look.

By default the tool scans the whole clip and repairs what it finds. Use
--range/--at to restrict detection to known trouble spots (in seconds), which
also lowers the detection threshold inside those windows.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AUTO_SENSITIVITY = 6.0
RANGE_SENSITIVITY = 3.5
AT_WINDOW_SECONDS = 0.75
TRANS_SIGMA_FLOOR = 1.0
LOG_SCALE_SIGMA_FLOOR = 0.0035
ROT_SIGMA_FLOOR = math.radians(0.12)
RAW_SIGMA_FLOOR = 0.6
FIT_SIGMA_FLOOR = 0.6
DUP_ABS_DIFF = 0.5
DUP_REL_DIFF = 0.3
DUP_CADENCE_FRACTION = 0.15
MIN_TRACK_POINTS = 12
MIN_CORRECTION_PX = 0.05
MIN_EVENT_CORRECTION_PX = 1.5
GROUP_GAP = 3
JUMP_EXPLAINED_RATIO = 0.8
JUMP_MIN_INLIERS = 0.25


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps: float
    frames: int
    duration: float

    @property
    def size_arg(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def rate_arg(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"


@dataclass
class Transition:
    index: int
    ok: bool
    vec: tuple[float, float, float, float, float, float] | None
    n_points: int
    inlier_ratio: float
    raw_diff: float
    warped_mae: float


@dataclass
class GlitchEvent:
    t0: int
    t1: int
    transitions: list[int]
    kind: str
    repair: str
    z_peak: float
    shift_px: float
    scale_factor: float
    stretch_factor: float
    rot_deg: float
    note: str = ""

    @property
    def interior_frames(self) -> list[int]:
        return list(range(self.t0, self.t1))


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
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = data["streams"][0]
    fps_num, fps_den = (int(part) for part in stream["r_frame_rate"].split("/"))
    fps = fps_num / fps_den if fps_den else 0.0
    duration = float(stream.get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or 0)
    if not frames and fps and duration:
        frames = int(round(duration * fps))
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_num=fps_num,
        fps_den=fps_den,
        fps=fps,
        frames=frames,
        duration=duration,
    )


def decode_rgb_frames(path: Path, info: VideoInfo):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    frame_bytes = info.width * info.height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"partial frame decoded from {path}")
            yield np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
        proc.stdout.close()
        if proc.wait():
            raise RuntimeError(f"decoder failed with exit code {proc.returncode} on {path}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


XF_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def xf_matrix(vec) -> "np.ndarray":
    m00, m01, m10, m11, tx, ty = vec
    return np.array([[m00, m01, tx], [m10, m11, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def xf_vec(matrix) -> tuple[float, float, float, float, float, float]:
    return (
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )


def xf_components(vec, width: int, height: int) -> tuple[float, float, float, float, float]:
    m00, m01, m10, m11, tx, ty = vec
    cx, cy = width / 2.0, height / 2.0
    dx = m00 * cx + m01 * cy + tx - cx
    dy = m10 * cx + m11 * cy + ty - cy
    det = max(m00 * m11 - m01 * m10, 1e-6)
    dlog = 0.5 * math.log(det)
    rot = math.atan2(m10 - m01, m00 + m11)
    singular = np.linalg.svd(
        np.array([[m00, m01], [m10, m11]], dtype=np.float64),
        compute_uv=False,
    )
    aniso = math.log(max(float(singular[0]), 1e-9) / max(float(singular[1]), 1e-9))
    return dx, dy, dlog, rot, aniso


def xf_displacement(vec, width: int, height: int) -> float:
    matrix = xf_matrix(vec)
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]],
        dtype=np.float64,
    ).T
    moved = matrix @ corners
    return float(np.max(np.hypot(moved[0] - corners[0], moved[1] - corners[1])))


def estimate_transition(prev_gray, cur_gray, index: int, to_native: float) -> Transition:
    raw_diff = float(cv2.absdiff(prev_gray, cur_gray).mean())
    failed = Transition(index, False, None, 0, 0.0, raw_diff, float("nan"))

    points = cv2.goodFeaturesToTrack(prev_gray, 600, 0.01, 8, blockSize=7)
    if points is None or len(points) < MIN_TRACK_POINTS:
        return failed
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        points.astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    keep = status.ravel() == 1
    if keep.sum() < MIN_TRACK_POINTS:
        return failed
    src = points[keep].reshape(-1, 2)
    dst = moved[keep].reshape(-1, 2)
    matrix, inliers = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.995,
    )
    if matrix is None or inliers is None:
        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.995,
        )
    if matrix is None or inliers is None:
        return failed

    height, width = prev_gray.shape
    warped = cv2.warpAffine(prev_gray, matrix, (width, height), flags=cv2.INTER_LINEAR)
    coverage = cv2.warpAffine(
        np.full((height, width), 255, np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    valid = coverage == 255
    if valid.sum() < 1000:
        valid = np.ones_like(valid)
    diff = cv2.absdiff(warped, cur_gray)
    warped_mae = float(diff[valid].mean())

    vec = (
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]) * to_native,
        float(matrix[1, 2]) * to_native,
    )
    return Transition(
        index=index,
        ok=True,
        vec=vec,
        n_points=int(keep.sum()),
        inlier_ratio=float(inliers.ravel().mean()),
        raw_diff=raw_diff,
        warped_mae=warped_mae,
    )


def analyze_clip(path: Path, info: VideoInfo, analysis_width: int) -> tuple[list[Transition], int]:
    scale = min(1.0, analysis_width / info.width)
    size = (
        max(2, int(round(info.width * scale))),
        max(2, int(round(info.height * scale))),
    )
    to_native = info.width / size[0]

    transitions: list[Transition] = []
    prev_gray = None
    count = 0
    for frame in decode_rgb_frames(path, info):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        if prev_gray is not None:
            transitions.append(estimate_transition(prev_gray, gray, count, to_native))
        prev_gray = gray
        count += 1
    if count < 3:
        raise RuntimeError(f"clip too short to stabilize: {count} frames")
    return transitions, count


def refine_transitions(
    clip: Path,
    info: VideoInfo,
    transitions: list[Transition],
    windows: list[tuple[int, int, list[GlitchEvent]]],
) -> tuple[int, int]:
    groups: list[list[int]] = []
    needed: set[int] = set()
    for _, _, window_events in windows:
        ts = sorted(
            {t for event in window_events if event.repair == "warp" for t in event.transitions}
        )
        if not ts:
            continue
        groups.append(ts)
        for t in ts:
            needed.update((t - 1, t))
        needed.update((ts[0] - 1, ts[-1]))
    if not groups:
        return 0, 0

    grays: dict[int, "np.ndarray"] = {}
    for idx, frame in enumerate(decode_rgb_frames(clip, info)):
        if idx in needed:
            grays[idx] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if len(grays) == len(needed):
            break

    refined = 0
    closed = 0
    for ts in groups:
        for t in ts:
            tr = transitions[t - 1]
            if not tr.ok or t - 1 not in grays or t not in grays:
                continue
            full = estimate_transition(grays[t - 1], grays[t], t, 1.0)
            if full.ok and full.inlier_ratio >= 0.2:
                tr.vec = full.vec
                refined += 1

        if len(ts) < 2 or ts[0] - 1 not in grays or ts[-1] not in grays:
            continue
        if any(not transitions[t - 1].ok for t in range(ts[0], ts[-1] + 1)):
            continue
        direct = estimate_transition(grays[ts[0] - 1], grays[ts[-1]], ts[-1], 1.0)
        if not direct.ok or direct.inlier_ratio < 0.25:
            continue
        chain = np.eye(3)
        for t in range(ts[0], ts[-1]):
            chain = xf_matrix(transitions[t - 1].vec) @ chain
        transitions[ts[-1] - 1].vec = xf_vec(xf_matrix(direct.vec) @ np.linalg.inv(chain))
        closed += 1
    return refined, closed


def track_points(gray_a, gray_b, max_corners: int = 1200):
    points = cv2.goodFeaturesToTrack(gray_a, max_corners, 0.01, 8, blockSize=7)
    if points is None or len(points) < 40:
        return None
    forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, points.astype(np.float32), None, winSize=(21, 21), maxLevel=4
    )
    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
        gray_b, gray_a, forward, None, winSize=(21, 21), maxLevel=4
    )
    src = points.reshape(-1, 2)
    dst = forward.reshape(-1, 2)
    round_trip = backward.reshape(-1, 2)
    keep = (status_f.ravel() == 1) & (status_b.ravel() == 1)
    keep &= np.linalg.norm(round_trip - src, axis=1) < 0.8
    if keep.sum() < 40:
        return None
    return src[keep], dst[keep]


def poly_design(xs, ys, width: int, height: int):
    xn = xs / width - 0.5
    yn = ys / height - 0.5
    return np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=1)


def fit_field(src, dst, width: int, height: int):
    displacement = dst - src
    design = poly_design(src[:, 0], src[:, 1], width, height)
    weights = np.ones(len(src))
    coeffs = None
    residual = None
    sigma = 1.0
    for _ in range(6):
        weighted = weights[:, None]
        coeffs, *_ = np.linalg.lstsq(design * weighted, displacement * weighted, rcond=None)
        residual = np.linalg.norm(design @ coeffs - displacement, axis=1)
        sigma = max(1.4826 * float(np.median(residual)), 0.3)
        weights = 1.0 / np.maximum(residual / (2.5 * sigma), 1.0)
    inliers = residual <= max(1.5, 3.0 * sigma)
    if inliers.sum() < 60:
        return None
    cell_x = np.clip((src[inliers, 0] / width * 4).astype(int), 0, 3)
    cell_y = np.clip((src[inliers, 1] / height * 4).astype(int), 0, 3)
    cells = set(zip(cell_x.tolist(), cell_y.tolist()))
    if len(cells) < 8 or len(set(cell_x.tolist())) < 3 or len(set(cell_y.tolist())) < 3:
        return None
    return coeffs, inliers


def field_between(gray_a, gray_b, width: int, height: int):
    tracked = track_points(gray_a, gray_b)
    if tracked is None:
        return None
    fitted = fit_field(tracked[0], tracked[1], width, height)
    return fitted[0] if fitted is not None else None


def static_drift(gray_a, gray_b, width: int, height: int) -> float | None:
    tracked = track_points(gray_a, gray_b)
    if tracked is None:
        return None
    src, dst = tracked
    fitted = fit_field(src, dst, width, height)
    if fitted is None:
        return None
    coeffs, inliers = fitted
    design = poly_design(src[inliers, 0], src[inliers, 1], width, height)
    field = np.linalg.norm(design @ coeffs, axis=1)
    return float(np.percentile(field, 95))


def field_max_displacement(coeffs, width: int, height: int, margin: float = 0.0) -> float:
    xs, ys = np.meshgrid(
        np.linspace(margin * width, (1.0 - margin) * width, 9, dtype=np.float64),
        np.linspace(margin * height, (1.0 - margin) * height, 17, dtype=np.float64),
    )
    design = poly_design(xs.ravel(), ys.ravel(), width, height)
    displacement = design @ coeffs
    return float(np.max(np.linalg.norm(displacement, axis=1)))


def field_disp_maps(coeffs, width: int, height: int):
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    xn = xs / width - 0.5
    yn = ys / height - 0.5
    terms = (np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn)
    disp_x = np.zeros_like(xn)
    disp_y = np.zeros_like(yn)
    for c, term in zip(coeffs, terms):
        disp_x += np.float32(c[0]) * term
        disp_y += np.float32(c[1]) * term
    return disp_x, disp_y


def affine_field_coeffs(vec, width: int, height: int):
    m00, m01, m10, m11, tx, ty = vec
    coeffs = np.zeros((6, 2), dtype=np.float64)
    coeffs[0, 0] = 0.5 * width * (m00 - 1.0) + 0.5 * height * m01 + tx
    coeffs[1, 0] = width * (m00 - 1.0)
    coeffs[2, 0] = height * m01
    coeffs[0, 1] = 0.5 * width * m10 + 0.5 * height * (m11 - 1.0) + ty
    coeffs[1, 1] = width * m10
    coeffs[2, 1] = height * (m11 - 1.0)
    return coeffs


def plan_field_corrections(
    clip: Path,
    info: VideoInfo,
    windows: list[tuple[int, int, list[GlitchEvent]]],
    transitions: list[Transition],
    replaced_frames: set[int],
) -> list[dict]:
    specs: list[tuple[int, list[int]]] = []
    needed: set[int] = set()
    for i, window in enumerate(windows):
        anomalous = sorted(
            {t for event in window[2] if event.repair == "warp" for t in event.transitions}
        )
        if not anomalous:
            continue
        specs.append((i, anomalous))
        for t in anomalous:
            needed.update((t - 1, t))
    plans: list[dict] = [{} for _ in windows]
    if not specs:
        return plans

    grays: dict[int, "np.ndarray"] = {}
    for idx, frame in enumerate(decode_rgb_frames(clip, info)):
        if idx in needed:
            grays[idx] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if len(grays) == len(needed):
            break

    for i, anomalous in specs:
        w0, w1, _ = windows[i]
        residuals: dict[int, "np.ndarray"] = {}
        usable = True
        for t in anomalous:
            tr = transitions[t - 1]
            if not tr.ok or t - 1 not in grays or t not in grays:
                usable = False
                break
            field = field_between(grays[t - 1], grays[t], info.width, info.height)
            if field is None:
                usable = False
                break
            residuals[t] = field - affine_field_coeffs(tr.vec, info.width, info.height)
        if not usable:
            continue

        end_residual = sum(residuals.values())
        length = w1 - (w0 - 1)
        corrections: dict[int, "np.ndarray"] = {}
        for f in range(w0, w1 + 1):
            if f in replaced_frames:
                continue
            accumulated = sum(
                (residuals[t] for t in anomalous if t <= f),
                np.zeros((6, 2), dtype=np.float64),
            )
            ramp = (f - (w0 - 1)) / length
            coeffs = accumulated - ramp * end_residual
            if field_max_displacement(coeffs, info.width, info.height) < MIN_CORRECTION_PX:
                continue
            corrections[f] = coeffs
        plans[i] = corrections
    return plans


def rolling_median(values, window: int):
    length = len(values)
    half = max(1, window // 2)
    out = np.empty(length, dtype=np.float64)
    for i in range(length):
        chunk = values[max(0, i - half) : min(length, i + half + 1)]
        good = chunk[np.isfinite(chunk)]
        out[i] = float(np.median(good)) if good.size else 0.0
    return out


def robust_z(values, window: int, floor: float, one_sided: bool = False, sigma: float | None = None):
    center = rolling_median(values, window)
    residual = values - center
    if sigma is None:
        finite = residual[np.isfinite(residual)]
        mad = float(np.median(np.abs(finite))) if finite.size else 0.0
        sigma = max(1.4826 * mad, floor)
    z = np.abs(np.nan_to_num(residual)) / sigma
    if one_sided:
        z = np.where(np.nan_to_num(residual) > 0, z, 0.0)
    return z, center, sigma


def motion_statistics(
    transitions: list[Transition],
    info: VideoInfo,
    sigma_overrides: dict | None = None,
) -> dict:
    raw = np.array([tr.raw_diff for tr in transitions], dtype=np.float64)
    motion_scale = float(np.percentile(raw, 75)) if len(raw) else 0.0
    dup = (raw < DUP_ABS_DIFF) & (raw < DUP_REL_DIFF * max(motion_scale, 1e-6))
    cadenced = bool(dup.mean() > DUP_CADENCE_FRACTION)

    components = [
        xf_components(tr.vec, info.width, info.height) if tr.ok else None
        for tr in transitions
    ]

    def series(part: int):
        return np.array(
            [
                components[i][part]
                if tr.ok and not (cadenced and dup[i])
                else math.nan
                for i, tr in enumerate(transitions)
            ],
            dtype=np.float64,
        )

    dx = series(0)
    dy = series(1)
    dlog = series(2)
    drot = series(3)
    daniso = series(4)
    fit = np.array(
        [
            tr.warped_mae if tr.ok and not (cadenced and dup[i]) else math.nan
            for i, tr in enumerate(transitions)
        ],
        dtype=np.float64,
    )
    raw_masked = np.where(cadenced & dup, math.nan, raw)

    overrides = sigma_overrides or {}
    window = max(15, int(round(1.5 * max(info.fps, 1.0))))
    z_dx, med_dx, s_dx = robust_z(dx, window, TRANS_SIGMA_FLOOR, sigma=overrides.get("dx"))
    z_dy, med_dy, s_dy = robust_z(dy, window, TRANS_SIGMA_FLOOR, sigma=overrides.get("dy"))
    z_dlog, med_dlog, s_dlog = robust_z(
        dlog, window, LOG_SCALE_SIGMA_FLOOR, sigma=overrides.get("dlog")
    )
    z_drot, med_drot, s_drot = robust_z(drot, window, ROT_SIGMA_FLOOR, sigma=overrides.get("drot"))
    z_daniso, med_daniso, s_daniso = robust_z(
        daniso, window, LOG_SCALE_SIGMA_FLOOR, sigma=overrides.get("daniso")
    )
    z_raw, med_raw, s_raw = robust_z(
        raw_masked, window, RAW_SIGMA_FLOOR, one_sided=True, sigma=overrides.get("raw")
    )
    z_fit, _, s_fit = robust_z(
        fit, window, FIT_SIGMA_FLOOR, one_sided=True, sigma=overrides.get("fit")
    )

    z_geo = np.max(np.stack([z_dx, z_dy, z_dlog, z_drot, z_daniso]), axis=0)

    return {
        "cadenced": cadenced,
        "dup_fraction": float(dup.mean()),
        "dx": dx,
        "dy": dy,
        "dlog": dlog,
        "drot": drot,
        "daniso": daniso,
        "raw": raw,
        "fit": fit,
        "med_dx": med_dx,
        "med_dy": med_dy,
        "med_dlog": med_dlog,
        "med_drot": med_drot,
        "med_daniso": med_daniso,
        "z_dx": z_dx,
        "z_dy": z_dy,
        "z_dlog": z_dlog,
        "z_drot": z_drot,
        "z_daniso": z_daniso,
        "z_raw": z_raw,
        "z_fit": z_fit,
        "z_geo": z_geo,
        "dup": dup,
        "sigmas": {
            "dx": s_dx,
            "dy": s_dy,
            "dlog": s_dlog,
            "drot": s_drot,
            "daniso": s_daniso,
            "raw": s_raw,
            "fit": s_fit,
        },
    }


def transition_time(index: int, fps: float) -> float:
    return index / max(fps, 1e-9)


def detect_events(
    transitions: list[Transition],
    stats: dict,
    fps: float,
    sensitivity: float,
    ranges: list[tuple[float, float]],
    max_event_transitions: int,
    repair_mode: str,
) -> tuple[list[GlitchEvent], list[str]]:
    dup_flags = (
        np.zeros_like(stats["dup"]) if stats["cadenced"] else stats["dup"]
    )
    flagged = (
        (stats["z_geo"] > sensitivity)
        | (stats["z_raw"] > sensitivity)
        | (stats["z_fit"] > sensitivity)
        | dup_flags
    )
    if ranges:
        for i, tr in enumerate(transitions):
            moment = transition_time(tr.index, fps)
            if not any(lo <= moment <= hi for lo, hi in ranges):
                flagged[i] = False

    groups: list[list[int]] = []
    for pos in np.flatnonzero(flagged):
        if groups and pos - groups[-1][-1] <= GROUP_GAP:
            groups[-1].append(int(pos))
        else:
            groups.append([int(pos)])

    events: list[GlitchEvent] = []
    warnings: list[str] = []
    for group in groups:
        ts = [transitions[pos].index for pos in group]
        t0, t1 = ts[0], ts[-1]
        span = t1 - t0 + 1
        if span > max_event_transitions and not ranges:
            warnings.append(
                f"skipping {span}-transition anomaly at {transition_time(t0, fps):.2f}s-"
                f"{transition_time(t1, fps):.2f}s; too long for a glitch "
                "(pass --range to force repair)"
            )
            continue

        def content_jump(pos: int) -> bool:
            tr = transitions[pos]
            if not tr.ok:
                return True
            if stats["z_fit"][pos] <= sensitivity:
                return False
            explained = tr.warped_mae / max(tr.raw_diff, 1e-6)
            return explained > JUMP_EXPLAINED_RATIO or tr.inlier_ratio < JUMP_MIN_INLIERS

        dup_any = bool(any(dup_flags[pos] for pos in group))
        jump_any = bool(any(content_jump(pos) for pos in group))
        shift_px = max(
            math.hypot(
                np.nan_to_num(stats["dx"][pos] - stats["med_dx"][pos]),
                np.nan_to_num(stats["dy"][pos] - stats["med_dy"][pos]),
            )
            for pos in group
        )
        scale_factor = max(
            math.exp(abs(np.nan_to_num(stats["dlog"][pos] - stats["med_dlog"][pos])))
            for pos in group
        )
        stretch_factor = max(
            math.exp(abs(np.nan_to_num(stats["daniso"][pos] - stats["med_daniso"][pos])))
            for pos in group
        )
        rot_deg = max(
            math.degrees(abs(np.nan_to_num(stats["drot"][pos] - stats["med_drot"][pos])))
            for pos in group
        )
        z_peak = max(
            max(stats["z_geo"][pos], stats["z_raw"][pos], stats["z_fit"][pos])
            for pos in group
        )

        if dup_any:
            kind = "stutter"
        elif jump_any:
            kind = "content-jump"
        else:
            by_kind = {
                "shift": max(max(stats["z_dx"][pos], stats["z_dy"][pos]) for pos in group),
                "scale-pop": max(stats["z_dlog"][pos] for pos in group),
                "stretch-pop": max(stats["z_daniso"][pos] for pos in group),
                "rotation": max(stats["z_drot"][pos] for pos in group),
            }
            kind = max(by_kind, key=by_kind.get)

        if repair_mode == "warp":
            repair = "warp"
        elif repair_mode == "interp":
            repair = "interp"
        else:
            repair = "interp" if kind in ("stutter", "content-jump") else "warp"

        event = GlitchEvent(
            t0=t0,
            t1=t1,
            transitions=ts,
            kind=kind,
            repair=repair,
            z_peak=float(z_peak),
            shift_px=float(shift_px),
            scale_factor=float(scale_factor),
            stretch_factor=float(stretch_factor),
            rot_deg=float(rot_deg),
        )
        if repair == "interp" and kind != "stutter" and not event.interior_frames:
            event.repair = "skipped"
            event.note = "single-transition content change; nothing to rebuild"
            warnings.append(
                f"event at {transition_time(t0, fps):.2f}s looks like a hard content "
                "change across one transition; left untouched"
            )
        events.append(event)
    return events, warnings


def plan_replacements(
    events: list[GlitchEvent],
    stats: dict,
    n_frames: int,
) -> dict[int, tuple[int, int, float]]:
    bad_frames: set[int] = set()
    for event in events:
        if event.repair != "interp":
            continue
        if event.kind == "stutter":
            for t in event.transitions:
                if stats["dup"][t - 1]:
                    bad_frames.add(t)
            for frame in event.interior_frames:
                bad_frames.add(frame)
        else:
            bad_frames.update(event.interior_frames)

    replacements: dict[int, tuple[int, int, float]] = {}
    for frame in sorted(bad_frames):
        lo = frame - 1
        while lo in bad_frames:
            lo -= 1
        hi = frame + 1
        while hi in bad_frames:
            hi += 1
        if lo < 0 or hi > n_frames - 1:
            continue
        alpha = (frame - lo) / (hi - lo)
        replacements[frame] = (lo, hi, alpha)
    return replacements


def plan_windows(
    events: list[GlitchEvent],
    pad: int,
    n_transitions: int,
) -> list[tuple[int, int, list[GlitchEvent]]]:
    spans = [
        (max(1, event.t0 - pad), min(n_transitions, event.t1 + pad), event)
        for event in events
        if event.repair != "skipped"
    ]
    spans.sort(key=lambda item: item[0])
    merged: list[tuple[int, int, list[GlitchEvent]]] = []
    for w0, w1, event in spans:
        if merged and w0 <= merged[-1][1] + 1:
            prev_w0, prev_w1, prev_events = merged.pop()
            merged.append((prev_w0, max(prev_w1, w1), prev_events + [event]))
        else:
            merged.append((w0, w1, [event]))
    return merged


def window_warp_corrections(
    transitions: list[Transition],
    window: tuple[int, int, list[GlitchEvent]],
    replaced_frames: set[int],
    info: VideoInfo,
    excluded_knots: set[int],
) -> dict[int, tuple]:
    w0, w1, events = window
    hat_ts = {t for event in events if event.repair == "warp" for t in event.transitions}
    hat_ts.update(t for t in range(w0, w1 + 1) if not transitions[t - 1].ok)
    if not hat_ts:
        return {}

    knots = [
        t
        for t in range(w0, w1 + 1)
        if transitions[t - 1].ok and t not in excluded_knots
    ]
    if not knots:
        return {}
    knot_vecs = np.array([transitions[t - 1].vec for t in knots], dtype=np.float64)
    all_ts = np.arange(w0, w1 + 1, dtype=np.float64)
    interp = np.stack(
        [np.interp(all_ts, np.array(knots, dtype=np.float64), knot_vecs[:, c]) for c in range(6)],
        axis=1,
    )

    actual_cum = np.eye(3)
    desired_cum = np.eye(3)
    raw_corrections: dict[int, "np.ndarray"] = {}
    for offset, t in enumerate(range(w0, w1 + 1)):
        tr = transitions[t - 1]
        hat_vec = tuple(interp[offset])
        if not tr.ok:
            act_vec = hat_vec
            des_vec = hat_vec
        elif t in hat_ts:
            act_vec = tr.vec
            des_vec = hat_vec
        else:
            act_vec = tr.vec
            des_vec = tr.vec
        actual_cum = xf_matrix(act_vec) @ actual_cum
        desired_cum = xf_matrix(des_vec) @ desired_cum
        correction = desired_cum @ np.linalg.inv(actual_cum)
        raw_corrections[t] = np.array(xf_vec(correction), dtype=np.float64)

    identity = np.array(XF_IDENTITY, dtype=np.float64)
    end_residual = raw_corrections[w1] - identity
    length = w1 - (w0 - 1)
    corrections: dict[int, tuple] = {}
    for t, vec in raw_corrections.items():
        ramp = (t - (w0 - 1)) / length
        adjusted = vec - ramp * end_residual
        if t in replaced_frames:
            continue
        if xf_displacement(tuple(adjusted), info.width, info.height) < MIN_CORRECTION_PX:
            continue
        corrections[t] = tuple(adjusted)
    return corrections


def motion_interp_frame(frame_a, frame_b, alpha: float):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 4, 25, 3, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, 0.5, 4, 25, 3, 7, 1.5, 0)

    height, width = gray_a.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warped_a = cv2.remap(
        frame_a,
        grid_x - alpha * flow_ab[..., 0],
        grid_y - alpha * flow_ab[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_b = cv2.remap(
        frame_b,
        grid_x - (1.0 - alpha) * flow_ba[..., 0],
        grid_y - (1.0 - alpha) * flow_ba[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    blended = (1.0 - alpha) * warped_a.astype(np.float32) + alpha * warped_b.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def buffer_spans(
    windows: list[tuple[int, int, list[GlitchEvent]]],
    warp_map: dict,
    replace_map: dict[int, tuple[int, int, float]],
) -> list[tuple[int, int]]:
    spans = []
    for w0, w1, _ in windows:
        lo, hi = w0, w1
        for frame, (g0, g1, _) in replace_map.items():
            if w0 <= frame <= w1:
                lo = min(lo, g0)
                hi = max(hi, g1)
        spans.append((lo, hi))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def build_encoder_cmd(
    output: Path,
    source: Path,
    info: VideoInfo,
    crf: int,
    preset: str,
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


def render_clip(
    source: Path,
    output: Path,
    info: VideoInfo,
    warp_map: dict,
    field_map: dict,
    replace_map: dict[int, tuple[int, int, float]],
    spans: list[tuple[int, int]],
    crf: int,
    preset: str,
    diag_frames: set[int],
) -> tuple[int, dict]:
    encoder = subprocess.Popen(
        build_encoder_cmd(output, source, info, crf, preset),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    diag_captures: dict[int, tuple] = {}
    span_iter = iter(spans)
    span = next(span_iter, None)
    buffered: dict[int, "np.ndarray"] = {}
    frame_index = 0

    def process_buffered() -> None:
        for f in sorted(buffered):
            frame = buffered[f]
            if f in replace_map:
                g0, g1, alpha = replace_map[f]
                fixed = motion_interp_frame(buffered[g0], buffered[g1], alpha)
            elif f in field_map or f in warp_map:
                inverse = np.linalg.inv(
                    xf_matrix(warp_map[f]) if f in warp_map else np.eye(3)
                ).astype(np.float32)
                xs, ys = np.meshgrid(
                    np.arange(info.width, dtype=np.float32),
                    np.arange(info.height, dtype=np.float32),
                )
                map_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
                map_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
                if f in field_map:
                    disp_x, disp_y = field_disp_maps(field_map[f], info.width, info.height)
                    map_x += disp_x
                    map_y += disp_y
                fixed = cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                fixed = frame
            if f in diag_frames:
                diag_captures[f] = (frame.copy(), fixed.copy())
            encoder.stdin.write(fixed.tobytes())
        buffered.clear()

    try:
        for frame in decode_rgb_frames(source, info):
            while span is not None and frame_index > span[1]:
                span = next(span_iter, None)
            if span is not None and span[0] <= frame_index <= span[1]:
                buffered[frame_index] = frame
                if frame_index == span[1]:
                    process_buffered()
            else:
                encoder.stdin.write(frame.tobytes())
            frame_index += 1
        if buffered:
            process_buffered()
    except BrokenPipeError as exc:
        raise RuntimeError("encoder closed before all frames were written") from exc
    finally:
        encoder.stdin.close()

    encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    if encoder.wait():
        raise RuntimeError(f"encoder failed with exit code {encoder.returncode}\n{encoder_stderr}")
    return frame_index, diag_captures


def write_png(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_motion_csv(path: Path, transitions: list[Transition], stats: dict, info: VideoInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "transition,time_s,dx,dy,scale,rot_deg,stretch,inlier_ratio,n_points,"
        "raw_diff,warped_mae,z_geo,z_raw,z_fit,duplicate"
    ]
    for i, tr in enumerate(transitions):
        if tr.ok:
            dx, dy, dlog, rot, aniso = xf_components(tr.vec, info.width, info.height)
            scale, rot_deg, stretch = math.exp(dlog), math.degrees(rot), math.exp(aniso)
        else:
            dx, dy, scale, rot_deg, stretch = (float("nan"),) * 5
        lines.append(
            f"{tr.index},{transition_time(tr.index, info.fps):.4f},{dx:.4f},{dy:.4f},"
            f"{scale:.6f},{rot_deg:.4f},{stretch:.6f},{tr.inlier_ratio:.3f},{tr.n_points},"
            f"{tr.raw_diff:.4f},{tr.warped_mae:.4f},{stats['z_geo'][i]:.2f},"
            f"{stats['z_raw'][i]:.2f},{stats['z_fit'][i]:.2f},{int(stats['dup'][i])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def event_summary(event: GlitchEvent, fps: float) -> str:
    frames = f"frames {event.t0}-{event.t1 - 1}" if event.interior_frames else f"transition {event.t0}"
    time_lo = transition_time(event.t0, fps)
    time_hi = transition_time(event.t1, fps)
    details = (
        f"peak shift {event.shift_px:.1f} px, scale x{event.scale_factor:.4f}, "
        f"stretch x{event.stretch_factor:.4f}, rot {event.rot_deg:.2f} deg, "
        f"z {event.z_peak:.1f}"
    )
    return f"{frames} @ {time_lo:.2f}-{time_hi:.2f}s  {event.kind}  {details}  repair {event.repair}"


def verify_events(
    events: list[GlitchEvent],
    stats_before: dict,
    stats_after: dict,
    n_transitions_after: int,
) -> list[dict]:
    def window_max(stats: dict, keys: list[str], lo: int, hi: int) -> float:
        hi = min(hi, len(stats["z_geo"]))
        return float(
            max(max(stats[key][pos] for key in keys) for pos in range(lo, hi))
        )

    results = []
    for event in events:
        lo = max(0, event.t0 - 2 - 1)
        hi = min(n_transitions_after, event.t1 + 2)
        geo_before = window_max(stats_before, ["z_geo"], lo, hi)
        geo_after = window_max(stats_after, ["z_geo"], lo, hi)
        content_before = window_max(stats_before, ["z_raw", "z_fit"], lo, hi)
        content_after = window_max(stats_after, ["z_raw", "z_fit"], lo, hi)
        if event.repair == "skipped":
            clean = True
        elif event.repair == "warp":
            clean = geo_after <= 3.0 and content_after <= max(6.0, 0.8 * content_before)
        else:
            clean = max(geo_after, content_after) < max(
                3.0, 0.5 * max(geo_before, content_before)
            )
        results.append(
            {
                "t0": event.t0,
                "t1": event.t1,
                "kind": event.kind,
                "repair": event.repair,
                "z_geometry_before": geo_before,
                "z_geometry_after": geo_after,
                "z_appearance_before": content_before,
                "z_appearance_after": content_after,
                "clean": bool(clean),
            }
        )
    return results


def verify_boundary_fields(
    output: Path,
    info: VideoInfo,
    events: list[GlitchEvent],
    stats: dict,
    transitions: list[Transition],
) -> tuple[dict[int, float], float]:
    targets = sorted({t for e in events if e.repair == "warp" for t in {e.t0, e.t1}})
    if not targets:
        return {}, 0.0

    event_ts = {t for e in events for t in e.transitions}
    span = int(round(2.0 * max(info.fps, 1.0)))
    candidates = [
        tr.index
        for tr in transitions
        if tr.ok
        and not stats["dup"][tr.index - 1]
        and tr.index not in event_ts
        and any(abs(tr.index - t) <= span for t in targets)
    ]
    by_motion = sorted(candidates, key=lambda t: transitions[t - 1].raw_diff, reverse=True)
    moving_half = by_motion[: max(4, len(by_motion) // 2)]
    if len(moving_half) > 8:
        picks = np.linspace(0, len(moving_half) - 1, 8).astype(int)
        controls = [moving_half[i] for i in picks]
    else:
        controls = moving_half

    needed = {f for t in targets + controls for f in (t - 1, t)}
    grays: dict[int, "np.ndarray"] = {}
    for idx, frame in enumerate(decode_rgb_frames(output, info)):
        if idx in needed:
            grays[idx] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if len(grays) == len(needed):
            break

    def measure(t: int) -> float | None:
        if t - 1 not in grays or t not in grays:
            return None
        return static_drift(grays[t - 1], grays[t], info.width, info.height)

    residuals: dict[int, float] = {}
    for t in targets:
        value = measure(t)
        if value is not None:
            residuals[t] = value
    control_values = [value for t in controls if (value := measure(t)) is not None]
    control_level = float(np.median(control_values)) if control_values else 0.0
    return residuals, control_level


def parse_ranges(range_args: list[str], at_args: list[float], duration: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for text in range_args or []:
        sep = ":" if ":" in text else "-"
        parts = text.split(sep)
        if len(parts) != 2:
            raise RuntimeError(f"could not parse range '{text}'; use START:END in seconds")
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise RuntimeError(f"could not parse range '{text}'; use START:END in seconds") from exc
        if hi <= lo:
            raise RuntimeError(f"range '{text}' is empty; END must be greater than START")
        ranges.append((max(0.0, lo), hi))
    for moment in at_args or []:
        ranges.append((max(0.0, moment - AT_WINDOW_SECONDS), moment + AT_WINDOW_SECONDS))
    if duration:
        ranges = [(lo, min(hi, duration)) for lo, hi in ranges if lo < duration]
    return sorted(ranges)


def default_output_path(clip: Path) -> Path:
    return clip.with_name(f"{clip.stem}_stabilized{clip.suffix}")


def default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_report.json")


def default_diagnostics_dir(output: Path) -> Path:
    return output.with_name(f"{output.stem}_diagnostics")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stabilizer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Detect and repair brief geometric glitches (scale pops, position "
            "jumps, stutter frames) inside a single clip."
        ),
        epilog="""\
Typical use:
  ./stabilizer.py clip.mp4

Restrict detection to known trouble spots (seconds):
  ./stabilizer.py clip.mp4 --range 7.5:9.5
  ./stabilizer.py clip.mp4 --at 8.2 --at 8.8

Inspect without rendering:
  ./stabilizer.py clip.mp4 --detect-only

What the tool does:
  1. Estimates global affine motion between every consecutive frame pair.
  2. Flags transitions whose motion spikes away from the local trajectory,
     duplicate/stutter frames, and content jumps.
  3. Re-estimates flagged transitions at full resolution and anchors each
     glitch window with a direct registration across it.
  4. Warps displaced frames back onto the interpolated trajectory, or rebuilds
     broken frames by motion-compensated interpolation of their neighbors.
  5. Re-encodes with the same size, frame rate, and audio; untouched frames
     pass through unmodified. If nothing is detected, the clip is remuxed
     losslessly.

Input requirements:
  - ffmpeg and ffprobe must be available on PATH.
  - Python dependencies from requirements.txt must be installed.
""",
    )
    parser.add_argument("clip", type=Path, help="Clip to stabilize.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path for the corrected clip. Default: the input clip name with "
            "'_stabilized' added before its extension."
        ),
    )
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="START:END",
        help=(
            "Only look for glitches between START and END seconds. May be "
            "given multiple times. Lowers the detection threshold inside the "
            "window (see --sensitivity)."
        ),
    )
    parser.add_argument(
        "--at",
        action="append",
        type=float,
        default=[],
        metavar="SECONDS",
        help=(
            f"Timecode of a known glitch; expands to a +/-{AT_WINDOW_SECONDS:.2f}s "
            "window. May be given multiple times."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        help=(
            "Anomaly threshold in robust z-score units; lower finds more. "
            f"Default: {AUTO_SENSITIVITY} for whole-clip scans, "
            f"{RANGE_SENSITIVITY} inside --range/--at windows."
        ),
    )
    parser.add_argument(
        "--repair",
        choices=("auto", "warp", "interp"),
        default="auto",
        help=(
            "Repair strategy. 'warp' re-aligns displaced frames, 'interp' "
            "rebuilds frames from neighbors, 'auto' picks per glitch. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--pad-frames",
        type=int,
        help=(
            "Clean frames on each side of a glitch included in the correction "
            "window. Default: fps / 3, at least 4."
        ),
    )
    parser.add_argument(
        "--max-event-seconds",
        type=float,
        default=0.75,
        help=(
            "Longest anomaly the whole-clip scan will treat as a glitch; "
            "longer anomalies are reported but left untouched. Ignored inside "
            "--range/--at windows. Default: 0.75."
        ),
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=480,
        help="Maximum working width for motion analysis. Default: 480.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Analyze and report; do not render an output clip.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Path for the JSON report of detections and repairs. Default: "
            "output name with '_report.json'."
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help=(
            "Directory for the motion trace CSVs and before/after PNGs of "
            "repaired frames. Default: output name with '_diagnostics'."
        ),
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not write diagnostic CSVs or PNGs.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip re-analyzing the rendered output to confirm the fix.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=8,
        help=(
            "libx264 CRF for the corrected clip. Lower is higher quality and "
            "larger output. Default: 8."
        ),
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="libx264 preset for the corrected clip. Default: slow.",
    )
    return parser.parse_args(argv)


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def remux_copy(source: Path, output: Path) -> None:
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-i",
            str(source),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def run(args: argparse.Namespace) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")

    load_python_dependencies()

    clip = resolved_path(args.clip)
    if not clip.is_file():
        raise RuntimeError(f"not found: {clip}")
    output = resolved_path(args.output) if args.output else default_output_path(clip)
    if output == clip:
        raise RuntimeError("output must be different from the input path")
    report_path = resolved_path(args.report) if args.report else default_report_path(output)
    diagnostics_dir = (
        resolved_path(args.diagnostics_dir)
        if args.diagnostics_dir
        else default_diagnostics_dir(output)
    )

    info = ffprobe_video(clip)
    ranges = parse_ranges(args.range, args.at, info.duration)
    sensitivity = args.sensitivity if args.sensitivity is not None else (
        RANGE_SENSITIVITY if ranges else AUTO_SENSITIVITY
    )
    pad = args.pad_frames if args.pad_frames is not None else max(4, int(round(info.fps / 3)))
    max_event_transitions = max(2, int(round(args.max_event_seconds * info.fps)))

    print(f"clip:       {clip.name}")
    print(
        f"video:      {info.size_arg} @ {info.rate_arg} fps, "
        f"{info.frames} frames, {info.duration:.2f} s"
    )
    if ranges:
        windows_text = ", ".join(f"{lo:.2f}-{hi:.2f}s" for lo, hi in ranges)
        print(f"mode:       ranged scan ({windows_text}), sensitivity {sensitivity:.1f}, repair {args.repair}")
    else:
        print(f"mode:       whole-clip scan, sensitivity {sensitivity:.1f}, repair {args.repair}")

    transitions, n_frames = analyze_clip(clip, info, args.analysis_width)
    stats = motion_statistics(transitions, info)
    baseline = (
        f"median |dx| {np.nanmedian(np.abs(stats['dx'])):.2f} px, "
        f"|dy| {np.nanmedian(np.abs(stats['dy'])):.2f} px, "
        f"|dscale| {math.expm1(np.nanmedian(np.abs(stats['dlog']))) * 100:.3f}%, "
        f"|drot| {math.degrees(np.nanmedian(np.abs(stats['drot']))):.3f} deg"
    )
    print(f"analysis:   {len(transitions)} transitions; {baseline}")
    if stats["cadenced"]:
        print(
            f"cadence:    {stats['dup_fraction'] * 100:.0f}% near-duplicate transitions; "
            "treating duplicates as the clip's normal frame cadence"
        )

    events, warnings = detect_events(
        transitions,
        stats,
        info.fps,
        sensitivity,
        ranges,
        max_event_transitions,
        args.repair,
    )
    for warning in warnings:
        print(f"warning:    {warning}")
    for i, event in enumerate(events, start=1):
        print(f"event {i}:    {event_summary(event, info.fps)}")

    active_events = [event for event in events if event.repair != "skipped"]
    windows = plan_windows(active_events, pad, len(transitions))
    refine_targets = {
        t for event in active_events if event.repair == "warp" for t in event.transitions
    }
    if refine_targets and not args.detect_only:
        refined, closed = refine_transitions(clip, info, transitions, windows)
        closure_note = f", closure-anchored {closed} window(s)" if closed else ""
        print(
            f"refine:     re-estimated {refined}/{len(refine_targets)} glitch "
            f"transitions at full resolution{closure_note}"
        )
    replace_map = plan_replacements(active_events, stats, n_frames)
    replaced_frames = set(replace_map)

    def below_floor(window, peak: float) -> None:
        for event in window[2]:
            if event.repair == "warp":
                event.repair = "skipped"
                event.note = (
                    f"residual geometry {peak:.2f} px is below the "
                    f"{MIN_EVENT_CORRECTION_PX} px repair floor"
                )
        print(
            f"note:       peak correction {peak:.2f} px at "
            f"{transition_time(window[0], info.fps):.2f}-"
            f"{transition_time(window[1], info.fps):.2f}s is below the repair "
            "floor; leaving those frames untouched"
        )

    field_map: dict[int, "np.ndarray"] = {}
    field_plans = (
        plan_field_corrections(clip, info, windows, transitions, replaced_frames)
        if not args.detect_only
        else [{} for _ in windows]
    )

    warp_map: dict[int, tuple] = {}
    excluded_knots = {t for event in events for t in event.transitions}
    for window, window_fields in zip(windows, field_plans):
        corrections = window_warp_corrections(
            transitions, window, replaced_frames, info, excluded_knots
        )
        peaks = [
            xf_displacement(vec, info.width, info.height) for vec in corrections.values()
        ] + [
            field_max_displacement(c, info.width, info.height)
            for c in window_fields.values()
        ]
        if not peaks:
            continue
        peak = max(peaks)
        if peak < MIN_EVENT_CORRECTION_PX:
            below_floor(window, peak)
            continue
        warp_map.update(corrections)
        field_map.update(window_fields)
    active_events = [event for event in events if event.repair != "skipped"]

    warped_frames = set(warp_map) | set(field_map)
    max_correction = max(
        (
            (xf_displacement(warp_map[f], info.width, info.height) if f in warp_map else 0.0)
            + (
                field_max_displacement(field_map[f], info.width, info.height)
                if f in field_map
                else 0.0
            )
            for f in warped_frames
        ),
        default=0.0,
    )
    if active_events:
        print(
            f"plan:       warp {len(warped_frames)} frames ({len(field_map)} with field "
            f"refinement, max correction {max_correction:.1f} px), "
            f"rebuild {len(replace_map)} frames"
        )
    else:
        print("plan:       no repairs needed")

    report_data: dict = {
        "input": str(clip),
        "output": str(output),
        "video": {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frames": n_frames,
            "duration": info.duration,
        },
        "mode": {
            "ranges": [[lo, hi] for lo, hi in ranges],
            "sensitivity": sensitivity,
            "repair": args.repair,
            "pad_frames": pad,
            "max_event_seconds": args.max_event_seconds,
            "analysis_width": args.analysis_width,
        },
        "baseline": baseline,
        "warnings": warnings,
        "events": [
            {
                "transitions": [event.t0, event.t1],
                "frames": [event.t0, max(event.t0, event.t1 - 1)],
                "time_s": [
                    transition_time(event.t0, info.fps),
                    transition_time(event.t1, info.fps),
                ],
                "kind": event.kind,
                "repair": event.repair,
                "z_peak": event.z_peak,
                "shift_px": event.shift_px,
                "scale_factor": event.scale_factor,
                "stretch_factor": event.stretch_factor,
                "rot_deg": event.rot_deg,
                "note": event.note,
            }
            for event in events
        ],
        "plan": {
            "frames_warped": len(warped_frames),
            "frames_field_refined": len(field_map),
            "frames_rebuilt": len(replace_map),
            "max_correction_px": max_correction,
        },
        "encoder": {"crf": args.crf, "preset": args.preset},
    }

    if not args.no_diagnostics:
        write_motion_csv(diagnostics_dir / "motion_trace.csv", transitions, stats, info)

    if args.detect_only:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        print(f"report:     {report_path}")
        if not args.no_diagnostics:
            print(f"diagnostics: {diagnostics_dir}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    if not warp_map and not field_map and not replace_map:
        remux_copy(clip, output)
        print(f"no repairs; remuxed losslessly -> {output}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        print(f"report:     {report_path}")
        return 0

    diag_frames: set[int] = set()
    if not args.no_diagnostics:

        def correction_size(f: int) -> float:
            if f in replace_map:
                return float("inf")
            if f in field_map:
                return field_max_displacement(field_map[f], info.width, info.height)
            return xf_displacement(warp_map[f], info.width, info.height)

        for event in active_events:
            candidates = [f for f in replace_map if event.t0 <= f < event.t1 + 1]
            candidates += [
                f
                for f in list(field_map) + list(warp_map)
                if event.t0 - pad <= f <= event.t1 + pad
            ]
            if candidates:
                diag_frames.add(max(candidates, key=correction_size))

    spans = buffer_spans(windows, warp_map, replace_map)
    frames_written, diag_captures = render_clip(
        clip,
        output,
        info,
        warp_map,
        field_map,
        replace_map,
        spans,
        args.crf,
        args.preset,
        diag_frames,
    )
    print(f"rendered:   {frames_written} frames -> {output}")

    if not args.no_verify:
        out_info = ffprobe_video(output)
        out_transitions, _ = analyze_clip(output, out_info, args.analysis_width)
        out_stats = motion_statistics(out_transitions, out_info, sigma_overrides=stats["sigmas"])
        checks = verify_events(events, stats, out_stats, len(out_transitions))
        boundary, control_level = verify_boundary_fields(
            output, out_info, events, stats, transitions
        )
        boundary_limit = max(MIN_EVENT_CORRECTION_PX, 2.0 * control_level)
        for check, event in zip(checks, events):
            residuals = [
                boundary[t] for t in (event.t0, event.t1) if boundary.get(t) is not None
            ]
            if event.repair == "warp" and residuals:
                check["static_drift_px"] = max(residuals)
                check["drift_limit_px"] = boundary_limit
                content_ok = check["z_appearance_after"] <= max(
                    6.0, 0.8 * check["z_appearance_before"]
                )
                check["clean"] = bool(
                    check["static_drift_px"] <= boundary_limit and content_ok
                )
        report_data["verify"] = checks
        report_data["verify_control_level_px"] = control_level
        for i, check in enumerate(checks, start=1):
            status = "ok" if check["clean"] else "review"
            boundary_note = (
                f"; boundary static drift {check['static_drift_px']:.2f} px "
                f"(limit {check['drift_limit_px']:.2f})"
                if "static_drift_px" in check
                else ""
            )
            print(
                f"verify:     event {i} geometry z {check['z_geometry_after']:.1f} "
                f"(was {check['z_geometry_before']:.1f}); appearance z "
                f"{check['z_appearance_after']:.1f} (was {check['z_appearance_before']:.1f})"
                f"{boundary_note} {status}"
            )
        if not args.no_diagnostics:
            write_motion_csv(
                diagnostics_dir / "motion_trace_output.csv",
                out_transitions,
                out_stats,
                out_info,
            )

    if not args.no_diagnostics:
        for frame_idx, (before, after) in diag_captures.items():
            write_png(diagnostics_dir / f"frame{frame_idx:04d}_original.png", before)
            write_png(diagnostics_dir / f"frame{frame_idx:04d}_fixed.png", after)
            absdiff = np.abs(before.astype(np.int16) - after.astype(np.int16))
            write_png(
                diagnostics_dir / f"frame{frame_idx:04d}_absdiff_amplified.png",
                np.clip(absdiff * 6, 0, 255).astype(np.uint8),
            )
        print(f"diagnostics: {diagnostics_dir}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"report:     {report_path}")
    return 0


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        return run(args)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {' '.join(map(str, exc.cmd))}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
