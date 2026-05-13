#!/usr/bin/env python3
"""
Make the cut between two timeline-adjacent clips visually seamless.

The CLI accepts the previous clip and the next clip as positional arguments.
By default it alters the next clip so its first frame matches the previous
clip's final frame. Use --alter previous when you need to preserve the next
clip and adjust the previous clip's ending instead.
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


@dataclass(frozen=True)
class SeamPlan:
    previous: Path
    next_clip: Path
    source: Path
    reference: Path
    source_boundary: str
    reference_boundary: str
    anchor_side: str
    altered_label: str
    reference_label: str


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


def residual_strength(
    frame_index: int,
    total_frames: int,
    anchor_frames: int,
    anchor_side: str,
) -> float:
    if anchor_frames <= 0:
        return 0.0

    span = min(anchor_frames, total_frames) if total_frames > 0 else anchor_frames
    if span <= 0:
        return 0.0

    if anchor_side == "start":
        if frame_index >= span:
            return 0.0
        if span == 1:
            return 1.0
        return 1.0 - frame_index / float(span - 1)

    if anchor_side == "end":
        if total_frames <= 0:
            raise RuntimeError("could not determine frame count needed for --alter previous")
        anchor_start = max(0, total_frames - span)
        if frame_index < anchor_start:
            return 0.0
        if span == 1:
            return 1.0
        return (frame_index - anchor_start) / float(span - 1)

    raise ValueError(f"unknown anchor side: {anchor_side}")


def render_corrected_clip(
    source: Path,
    output: Path,
    info: VideoInfo,
    matrix: np.ndarray,
    color_matrix: np.ndarray,
    seam_residual: np.ndarray,
    anchor_frames: int,
    anchor_side: str,
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

            strength = residual_strength(frame_index, info.frames, anchor_frames, anchor_side)
            if strength:
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


def default_output_path(previous: Path, next_clip: Path, alter: str) -> Path:
    source = next_clip if alter == "next" else previous
    return source.with_name(f"{source.stem}_seamless{source.suffix}")


def default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_report.json")


def default_diagnostics_dir(output: Path) -> Path:
    return output.with_name(f"{output.stem}_diagnostics")


def build_plan(previous: Path, next_clip: Path, alter: str) -> SeamPlan:
    if alter == "next":
        return SeamPlan(
            previous=previous,
            next_clip=next_clip,
            source=next_clip,
            reference=previous,
            source_boundary="first",
            reference_boundary="last",
            anchor_side="start",
            altered_label="next clip start",
            reference_label="previous clip end",
        )

    if alter == "previous":
        return SeamPlan(
            previous=previous,
            next_clip=next_clip,
            source=previous,
            reference=next_clip,
            source_boundary="last",
            reference_boundary="first",
            anchor_side="end",
            altered_label="previous clip end",
            reference_label="next clip start",
        )

    raise ValueError(f"unknown alter mode: {alter}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seamstress.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Make the transition between two separately generated clips of the "
            "same continuous shot visually seamless."
        ),
        epilog="""\
Typical use:
  ./seamstress.py previous.mp4 next.mp4

Preserve the next clip and alter the previous clip's ending instead:
  ./seamstress.py previous.mp4 next.mp4 --alter previous

Write explicit outputs:
  ./seamstress.py previous.mp4 next.mp4 \\
    --output next_fixed.mp4 \\
    --report next_fixed_report.json \\
    --diagnostics-dir next_fixed_diagnostics

What the tool does:
  1. Reads the boundary frames at the cut.
  2. Estimates subpixel affine alignment with OpenCV ECC.
  3. Solves a robust RGB color transform on the aligned boundary frame.
  4. Re-renders the altered clip with the spatial and color correction.
  5. Adds a short residual ramp at the seam so the boundary frame matches exactly
     before final video encoding.

Input requirements:
  - The two clips must have the same width, height, and frame rate.
  - ffmpeg and ffprobe must be available on PATH.
  - Python dependencies from requirements.txt must be installed.
""",
    )
    parser.add_argument(
        "previous",
        type=Path,
        help="Earlier clip in the timeline; its final frame touches the cut.",
    )
    parser.add_argument(
        "next",
        type=Path,
        help="Later clip in the timeline; its first frame touches the cut.",
    )
    parser.add_argument(
        "--alter",
        choices=("next", "previous"),
        default="next",
        help=(
            "Which clip to re-render. 'next' matches the later clip's first frame "
            "to the earlier clip's final frame. 'previous' matches the earlier "
            "clip's final frame to the later clip's first frame. Default: next."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path for the corrected clip. Default: the altered input clip name "
            "with '_seamless' added before its extension."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Path for the JSON report containing alignment matrices and seam "
            "metrics. Default: output name with '_report.json'."
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help=(
            "Directory for diagnostic PNGs of the reference, source, warped, "
            "corrected, encoded, and amplified-difference boundary frames. "
            "Default: output name with '_diagnostics'."
        ),
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not write diagnostic PNGs.",
    )
    parser.add_argument(
        "--mask-margin",
        type=int,
        default=48,
        help=(
            "Pixels to ignore at each frame edge during alignment and color "
            "solving. Increase if edge artifacts influence the match. Default: 48."
        ),
    )
    parser.add_argument(
        "--ecc-width",
        type=int,
        default=640,
        help=(
            "Maximum working width for the initial ECC alignment pass. Lower is "
            "faster; higher can improve difficult matches. Default: 640."
        ),
    )
    parser.add_argument(
        "--no-full-refine",
        action="store_true",
        help="Skip the full-resolution ECC refinement after the downsampled pass.",
    )
    parser.add_argument(
        "--trim-percentile",
        type=float,
        default=75.0,
        help=(
            "Residual percentile kept while solving the robust RGB color matrix. "
            "Lower values reject more outliers. Default: 75.0."
        ),
    )
    parser.add_argument(
        "--anchor-frames",
        type=int,
        default=6,
        help=(
            "Number of frames used for the residual seam ramp. With --alter next, "
            "the exact match fades out from frame 0. With --alter previous, it "
            "fades into the final frame. Use 0 to disable. Default: 6."
        ),
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


def validate_inputs(plan: SeamPlan, output: Path) -> tuple[VideoInfo, VideoInfo]:
    for path in (plan.previous, plan.next_clip):
        if not path.is_file():
            raise RuntimeError(f"not found: {path}")

    if output in (plan.previous, plan.next_clip):
        raise RuntimeError("output must be different from both input paths")

    previous_info = ffprobe_video(plan.previous)
    next_info = ffprobe_video(plan.next_clip)
    if (previous_info.width, previous_info.height) != (next_info.width, next_info.height):
        raise RuntimeError("clips must have the same frame size")
    if abs(previous_info.fps - next_info.fps) > 1e-6:
        raise RuntimeError("clips must have the same frame rate")

    source_info = next_info if plan.source == plan.next_clip else previous_info
    reference_info = previous_info if plan.reference == plan.previous else next_info
    return source_info, reference_info


def write_diagnostics(
    diagnostics: Path,
    reference_boundary: np.ndarray,
    source_boundary: np.ndarray,
    warped_boundary: np.ndarray,
    corrected_boundary: np.ndarray,
    output_boundary: np.ndarray,
) -> None:
    write_png(diagnostics / "01_reference_boundary.png", reference_boundary)
    write_png(diagnostics / "02_source_boundary.png", source_boundary)
    write_png(diagnostics / "03_source_warped.png", warped_boundary)
    write_png(diagnostics / "04_source_corrected.png", corrected_boundary)
    write_png(diagnostics / "05_output_boundary_encoded.png", output_boundary)
    absdiff = np.abs(reference_boundary.astype(np.int16) - output_boundary.astype(np.int16))
    write_png(diagnostics / "06_encoded_absdiff_amplified.png", np.clip(absdiff * 6, 0, 255).astype(np.uint8))


def run(args: argparse.Namespace) -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")

    load_python_dependencies()

    previous = resolved_path(args.previous)
    next_clip = resolved_path(args.next)
    output = resolved_path(args.output) if args.output else default_output_path(previous, next_clip, args.alter)
    report = resolved_path(args.report) if args.report else default_report_path(output)
    diagnostics_dir = (
        resolved_path(args.diagnostics_dir)
        if args.diagnostics_dir
        else default_diagnostics_dir(output)
    )
    plan = build_plan(previous, next_clip, args.alter)

    source_info, reference_info = validate_inputs(plan, output)

    print(f"previous clip: {plan.previous.name}")
    print(f"next clip:     {plan.next_clip.name}")
    print(f"altering:      {plan.source.name} ({plan.altered_label})")
    print(f"reference:     {plan.reference.name} ({plan.reference_label})")
    print(f"video:         {source_info.size_arg} @ {source_info.rate_arg} fps")

    reference_boundary = decode_boundary_frame(
        plan.reference,
        reference_info,
        plan.reference_boundary,
    )
    source_boundary = decode_boundary_frame(
        plan.source,
        source_info,
        plan.source_boundary,
    )
    raw_metrics = frame_metrics(reference_boundary, source_boundary)
    print(f"raw boundary:         MAE {raw_metrics['mae']:.4f}, RMSE {raw_metrics['rmse']:.4f}")

    affine, ecc = estimate_affine(
        reference_boundary,
        source_boundary,
        args.mask_margin,
        args.ecc_width,
        full_refine=not args.no_full_refine,
    )
    warped_boundary = warp_frame(source_boundary, affine, source_info.width, source_info.height)
    spatial_metrics = frame_metrics(reference_boundary, warped_boundary)
    print(f"ECC affine score:     {ecc:.6f}")
    print("affine matrix:")
    print(affine)
    print(f"spatial boundary:     MAE {spatial_metrics['mae']:.4f}, RMSE {spatial_metrics['rmse']:.4f}")

    mask = make_mask(source_info.height, source_info.width, args.mask_margin)
    color_matrix = solve_color_matrix(
        reference_boundary,
        warped_boundary,
        mask,
        args.trim_percentile,
    )
    corrected_boundary = apply_color_matrix(warped_boundary, color_matrix)
    corrected_metrics = frame_metrics(reference_boundary, corrected_boundary)
    seam_residual = reference_boundary.astype(np.float32) - corrected_boundary.astype(np.float32)
    anchored_boundary = np.clip(
        corrected_boundary.astype(np.float32) + seam_residual,
        0,
        255,
    ).astype(np.uint8)
    anchored_metrics = frame_metrics(reference_boundary, anchored_boundary)
    print("RGB color matrix (input RGB plus bias -> output RGB):")
    print(color_matrix)
    print(f"corrected boundary:   MAE {corrected_metrics['mae']:.4f}, RMSE {corrected_metrics['rmse']:.4f}")
    if args.anchor_frames:
        label = "anchored frame 0" if plan.anchor_side == "start" else "anchored final frame"
        print(f"{label}: MAE {anchored_metrics['mae']:.4f}, RMSE {anchored_metrics['rmse']:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    frames_written = render_corrected_clip(
        plan.source,
        output,
        source_info,
        affine,
        color_matrix,
        seam_residual,
        max(0, args.anchor_frames),
        plan.anchor_side,
        args.crf,
        args.preset,
    )
    print(f"rendered {frames_written} frames -> {output}")

    output_info = ffprobe_video(output)
    output_boundary = decode_boundary_frame(output, output_info, plan.source_boundary)
    encoded_metrics = frame_metrics(reference_boundary, output_boundary)
    print(f"encoded output frame: MAE {encoded_metrics['mae']:.4f}, RMSE {encoded_metrics['rmse']:.4f}")

    if not args.no_diagnostics:
        write_diagnostics(
            diagnostics_dir,
            reference_boundary,
            source_boundary,
            warped_boundary,
            corrected_boundary,
            output_boundary,
        )
        print(f"diagnostics: {diagnostics_dir}")

    report_data = {
        "previous": str(plan.previous),
        "next": str(plan.next_clip),
        "alter": args.alter,
        "source": str(plan.source),
        "reference": str(plan.reference),
        "output": str(output),
        "frames_written": frames_written,
        "source_boundary": plan.source_boundary,
        "reference_boundary": plan.reference_boundary,
        "anchor_side": plan.anchor_side,
        "affine_ecc": ecc,
        "affine_matrix": affine.tolist(),
        "color_matrix": color_matrix.tolist(),
        "raw_boundary": raw_metrics,
        "spatial_boundary": spatial_metrics,
        "corrected_boundary": corrected_metrics,
        "anchored_boundary": anchored_metrics,
        "encoded_output_boundary": encoded_metrics,
        "mask_margin": args.mask_margin,
        "ecc_width": args.ecc_width,
        "full_refine": not args.no_full_refine,
        "trim_percentile": args.trim_percentile,
        "anchor_frames": max(0, args.anchor_frames),
        "crf": args.crf,
        "preset": args.preset,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"report: {report}")
    return 0


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        return run(args)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
