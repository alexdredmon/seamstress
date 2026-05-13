#!/usr/bin/env python3
"""
Make the first joiner clip's end line up with the next clip at the timeline cut.

Default use from the project root:

    venv/bin/python joiner/make_end_seamless.py

The script:
  1. Decodes the final video frame of 00-02.mp4 and the first frame of
     01-03.mp4.
  2. Estimates a subpixel affine crop/translation correction with OpenCV ECC.
  3. Solves a robust RGB color matrix on the aligned boundary frame.
  4. Re-renders 00-02.mp4 with that spatial and color correction.
  5. Applies a short residual seam ramp by default, fading into an exact match
     on the final frame.
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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "joiner" / "01-03.mp4"
DEFAULT_SOURCE = ROOT / "joiner" / "00-02.mp4"
DEFAULT_OUTPUT = ROOT / "joiner" / "00-02_seamless.mp4"
DEFAULT_REPORT = ROOT / "joiner" / "end_seamless_report.json"


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


def decode_boundary_frame(path: Path, info: VideoInfo, which: str) -> np.ndarray:
    if which == "first":
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    elif which == "last":
        window = max(1.0, 12.0 / max(info.fps, 1.0))
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
    else:
        raise ValueError(f"unknown boundary side: {which}")

    raw = subprocess.check_output(cmd)
    frame_bytes = info.width * info.height * 3
    count = len(raw) // frame_bytes
    if count < 1:
        raise RuntimeError(f"could not decode {which} frame from {path}")
    frames = np.frombuffer(raw[: count * frame_bytes], np.uint8).reshape(
        count, info.height, info.width, 3
    )
    return frames[-1].copy() if which == "last" else frames[0].copy()


def highpass_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    return np.clip(gray - blur + 0.5, 0.0, 1.0).astype(np.float32)


def make_mask(height: int, width: int, margin: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    margin = max(0, min(margin, min(height, width) // 3))
    mask[margin : height - margin, margin : width - margin] = 255
    return mask


def scale_affine(matrix: np.ndarray, factor: float) -> np.ndarray:
    scaled = matrix.copy()
    scaled[0, 2] *= factor
    scaled[1, 2] *= factor
    return scaled


def downsample_for_ecc(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / width
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale


def estimate_affine(
    reference: np.ndarray,
    source: np.ndarray,
    mask_margin: int,
    ecc_width: int,
    full_refine: bool,
) -> tuple[np.ndarray, float]:
    ref_small, scale = downsample_for_ecc(reference, ecc_width)
    src_small, _ = downsample_for_ecc(source, ecc_width)
    small_mask = make_mask(ref_small.shape[0], ref_small.shape[1], round(mask_margin * scale))

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        1500,
        1e-8,
    )
    initial = np.eye(2, 3, dtype=np.float32)
    cc, matrix = cv2.findTransformECC(
        highpass_gray(ref_small),
        highpass_gray(src_small),
        initial,
        cv2.MOTION_AFFINE,
        criteria,
        small_mask,
        5,
    )

    matrix = scale_affine(matrix, 1.0 / scale).astype(np.float32)
    if not full_refine or scale == 1.0:
        return matrix, float(cc)

    full_mask = make_mask(reference.shape[0], reference.shape[1], mask_margin)
    full_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        500,
        1e-9,
    )
    cc, matrix = cv2.findTransformECC(
        highpass_gray(reference),
        highpass_gray(source),
        matrix,
        cv2.MOTION_AFFINE,
        full_criteria,
        full_mask,
        5,
    )
    return matrix.astype(np.float32), float(cc)


def warp_frame(frame: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def solve_color_matrix(
    reference: np.ndarray,
    warped_source: np.ndarray,
    mask: np.ndarray,
    trim_percentile: float,
) -> np.ndarray:
    src = warped_source.reshape(-1, 3).astype(np.float64)
    ref = reference.reshape(-1, 3).astype(np.float64)
    base = mask.reshape(-1).astype(bool)
    unsaturated = ((src > 3) & (src < 252)).all(axis=1) & ((ref > 3) & (ref < 252)).all(axis=1)
    usable = base & unsaturated
    if usable.sum() < 1000:
        usable = base
    selected = usable.copy()

    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    src_with_bias = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    for _ in range(4):
        matrix = np.linalg.lstsq(src_with_bias[selected], ref[selected], rcond=None)[0]
        predicted = src_with_bias @ matrix
        residual = np.sqrt(np.mean((predicted - ref) ** 2, axis=1))
        threshold = np.percentile(residual[usable], trim_percentile)
        selected = usable & (residual <= threshold)
        if selected.sum() < 1000:
            selected = usable
            break
    return matrix


def apply_color_matrix(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32)
    corrected = rgb @ matrix[:3, :].astype(np.float32) + matrix[3, :].astype(np.float32)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def frame_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))),
    }


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


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


def render_corrected_clip(
    source: Path,
    output: Path,
    info: VideoInfo,
    matrix: np.ndarray,
    color_matrix: np.ndarray,
    seam_residual: np.ndarray,
    anchor_frames: int,
    crf: int,
    preset: str,
) -> int:
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        build_encoder_cmd(output, source, info, crf, preset),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None
    frame_bytes = info.width * info.height * 3
    frame_index = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"partial frame from decoder at frame {frame_index}")
            frame = np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
            corrected = apply_color_matrix(
                warp_frame(frame, matrix, info.width, info.height),
                color_matrix,
            ).astype(np.float32)

            anchor_start = max(0, info.frames - anchor_frames)
            if anchor_frames > 0 and info.frames > 0 and frame_index >= anchor_start:
                if anchor_frames == 1:
                    strength = 1.0
                else:
                    strength = (frame_index - anchor_start) / float(anchor_frames - 1)
                strength = max(0.0, min(1.0, strength))
                corrected = np.clip(corrected + seam_residual * strength, 0, 255)

            encoder.stdin.write(corrected.astype(np.uint8).tobytes())
            frame_index += 1
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Affine-align and color-match 00-02.mp4's final frame to 01-03.mp4's first frame.",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--diagnostics-dir", type=Path, default=ROOT / "joiner" / "diagnostics")
    parser.add_argument("--mask-margin", type=int, default=48)
    parser.add_argument("--ecc-width", type=int, default=640)
    parser.add_argument("--no-full-refine", action="store_true")
    parser.add_argument("--trim-percentile", type=float, default=75.0)
    parser.add_argument(
        "--anchor-frames",
        type=int,
        default=6,
        help="Fade the exact final-frame residual in over this many frames. Use 0 to disable.",
    )
    parser.add_argument("--crf", type=int, default=8)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("error: ffmpeg and ffprobe must be on PATH", file=sys.stderr)
        return 2

    args = parse_args(argv)
    reference = args.reference.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = args.report.expanduser().resolve()
    for path in (reference, source):
        if not path.is_file():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2

    ref_info = ffprobe_video(reference)
    src_info = ffprobe_video(source)
    if (ref_info.width, ref_info.height) != (src_info.width, src_info.height):
        print("error: clips must have the same frame size", file=sys.stderr)
        return 2
    if abs(ref_info.fps - src_info.fps) > 1e-6:
        print("error: clips must have the same frame rate", file=sys.stderr)
        return 2

    print(f"next clip: {reference.name} ({ref_info.size_arg} @ {ref_info.rate_arg} fps)")
    print(f"source:    {source.name} ({src_info.frames} frames)")

    reference_first = decode_boundary_frame(reference, ref_info, "first")
    source_last = decode_boundary_frame(source, src_info, "last")
    raw_metrics = frame_metrics(reference_first, source_last)
    print(f"raw boundary:         MAE {raw_metrics['mae']:.4f}, RMSE {raw_metrics['rmse']:.4f}")

    affine, ecc = estimate_affine(
        reference_first,
        source_last,
        args.mask_margin,
        args.ecc_width,
        full_refine=not args.no_full_refine,
    )
    warped_last = warp_frame(source_last, affine, src_info.width, src_info.height)
    spatial_metrics = frame_metrics(reference_first, warped_last)
    print(f"ECC affine score:     {ecc:.6f}")
    print("affine matrix:")
    print(affine)
    print(f"spatial boundary:     MAE {spatial_metrics['mae']:.4f}, RMSE {spatial_metrics['rmse']:.4f}")

    mask = make_mask(src_info.height, src_info.width, args.mask_margin)
    color_matrix = solve_color_matrix(
        reference_first,
        warped_last,
        mask,
        args.trim_percentile,
    )
    corrected_last = apply_color_matrix(warped_last, color_matrix)
    corrected_metrics = frame_metrics(reference_first, corrected_last)
    seam_residual = reference_first.astype(np.float32) - corrected_last.astype(np.float32)
    anchored_last = np.clip(corrected_last.astype(np.float32) + seam_residual, 0, 255).astype(np.uint8)
    anchored_metrics = frame_metrics(reference_first, anchored_last)
    print("RGB color matrix (input RGB plus bias -> output RGB):")
    print(color_matrix)
    print(f"corrected boundary:   MAE {corrected_metrics['mae']:.4f}, RMSE {corrected_metrics['rmse']:.4f}")
    if args.anchor_frames:
        print(
            f"anchored final frame: MAE {anchored_metrics['mae']:.4f}, "
            f"RMSE {anchored_metrics['rmse']:.4f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    frames_written = render_corrected_clip(
        source,
        output,
        src_info,
        affine,
        color_matrix,
        seam_residual,
        max(0, args.anchor_frames),
        args.crf,
        args.preset,
    )
    print(f"rendered {frames_written} frames -> {output}")

    out_info = ffprobe_video(output)
    output_last = decode_boundary_frame(output, out_info, "last")
    encoded_metrics = frame_metrics(reference_first, output_last)
    print(f"encoded output frame: MAE {encoded_metrics['mae']:.4f}, RMSE {encoded_metrics['rmse']:.4f}")

    if not args.no_diagnostics:
        diagnostics = args.diagnostics_dir.expanduser().resolve()
        write_png(diagnostics / "00_02_01_reference_first.png", reference_first)
        write_png(diagnostics / "00_02_02_source_last.png", source_last)
        write_png(diagnostics / "00_02_03_warped_last.png", warped_last)
        write_png(diagnostics / "00_02_04_corrected_last.png", corrected_last)
        write_png(diagnostics / "00_02_05_output_last_encoded.png", output_last)
        absdiff = np.abs(reference_first.astype(np.int16) - output_last.astype(np.int16)).astype(np.uint8)
        write_png(diagnostics / "00_02_06_encoded_absdiff.png", np.clip(absdiff * 6, 0, 255).astype(np.uint8))

    report_data = {
        "reference": str(reference),
        "source": str(source),
        "output": str(output),
        "frames_written": frames_written,
        "affine_ecc": ecc,
        "affine_matrix": affine.tolist(),
        "color_matrix": color_matrix.tolist(),
        "raw_boundary": raw_metrics,
        "spatial_boundary": spatial_metrics,
        "corrected_boundary": corrected_metrics,
        "anchored_boundary": anchored_metrics,
        "encoded_output_boundary": encoded_metrics,
        "anchor_frames": max(0, args.anchor_frames),
        "crf": args.crf,
        "preset": args.preset,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
