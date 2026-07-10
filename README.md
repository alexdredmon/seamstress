# Seamstress

Seamstress makes the cut between two separately generated clips of the same
continuous shot feel seamless. It measures the boundary frames, estimates the
spatial and color mismatch, re-renders one of the two clips, and applies a short
residual ramp so the frame at the cut lands on the reference frame.

The tool has one CLI entrypoint:

```sh
./seamstress.py PREVIOUS_CLIP NEXT_CLIP [options]
```

The repository also ships [Stabilizer](#stabilizer), a companion tool that
repairs brief glitches inside a single clip.

`PREVIOUS_CLIP` is the earlier clip in the timeline. `NEXT_CLIP` is the later
clip in the timeline.

## Requirements

Install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` separately and make sure both commands are on
`PATH`.

The input clips must have the same frame width, frame height, and frame rate.
The corrected output is encoded with `libx264`, `yuv420p`, and copied audio from
the clip being altered when audio is present.

## Quick Start

Adjust the next clip so its first frame matches the previous clip's final frame:

```sh
./seamstress.py previous.mp4 next.mp4
```

This writes:

```text
next_seamless.mp4
next_seamless_report.json
next_seamless_diagnostics/
```

Preserve the next clip and adjust the previous clip's ending instead:

```sh
./seamstress.py previous.mp4 next.mp4 --alter previous
```

This writes:

```text
previous_seamless.mp4
previous_seamless_report.json
previous_seamless_diagnostics/
```

Use explicit output paths when integrating with a larger pipeline:

```sh
./seamstress.py previous.mp4 next.mp4 \
  --output corrected-next.mp4 \
  --report corrected-next-report.json \
  --diagnostics-dir corrected-next-diagnostics
```

## Choosing What To Alter

Use the default `--alter next` when the first clip is already approved and the
later clip can be re-rendered. Seamstress compares:

```text
previous final frame -> next first frame
```

It then applies the correction to the next clip and fades the exact residual
match out from frame 0 over `--anchor-frames`.

Use `--alter previous` when the later clip must remain untouched. Seamstress
compares:

```text
previous final frame -> next first frame
```

It then applies the correction to the previous clip and fades the exact residual
match into the final frame over `--anchor-frames`.

## CLI Reference

```text
usage: seamstress.py [-h] [--alter {next,previous}] [-o OUTPUT]
                     [--report REPORT] [--diagnostics-dir DIAGNOSTICS_DIR]
                     [--no-diagnostics] [--mask-margin MASK_MARGIN]
                     [--ecc-width ECC_WIDTH] [--no-full-refine]
                     [--trim-percentile TRIM_PERCENTILE]
                     [--anchor-frames ANCHOR_FRAMES] [--crf CRF]
                     [--preset PRESET]
                     PREVIOUS_CLIP NEXT_CLIP
```

Arguments:

- `PREVIOUS_CLIP`: earlier clip in the timeline.
- `NEXT_CLIP`: later clip in the timeline.

Options:

- `--alter {next,previous}`: choose which clip gets re-rendered. Default:
  `next`.
- `-o, --output PATH`: corrected clip path. Default: the altered input clip name
  with `_seamless` before the extension.
- `--report PATH`: JSON report path. Default: output name with `_report.json`.
- `--diagnostics-dir PATH`: diagnostic PNG directory. Default: output name with
  `_diagnostics`.
- `--no-diagnostics`: skip diagnostic PNG output.
- `--mask-margin PIXELS`: ignore this many pixels at each frame edge during
  alignment and color solving. Default: `48`.
- `--ecc-width PIXELS`: maximum working width for the initial ECC alignment
  pass. Lower is faster; higher can help difficult matches. Default: `640`.
- `--no-full-refine`: skip the full-resolution ECC refinement pass.
- `--trim-percentile VALUE`: residual percentile kept while solving the robust
  RGB color matrix. Lower values reject more outliers. Default: `75.0`.
- `--anchor-frames N`: number of frames used for the residual seam ramp. Use `0`
  to disable. Default: `6`.
- `--crf N`: `libx264` quality. Lower is higher quality and larger output.
  Default: `8`.
- `--preset NAME`: `libx264` speed/compression preset. Default: `slow`.
- `-h, --help`: show the full CLI help.

## Outputs

The corrected clip is the only video file Seamstress writes by default. It does
not concatenate the two clips; put the untouched clip and corrected clip next to
each other in your editor or downstream pipeline.

The JSON report records:

- input and output paths
- which clip was altered
- which boundary frame was used from each clip
- ECC alignment score
- affine transform matrix
- RGB color matrix
- raw, spatial, corrected, anchored, and encoded seam metrics
- encoder and tuning options used for the run

The diagnostics directory contains:

- `01_reference_boundary.png`: boundary frame that the altered clip must match
- `02_source_boundary.png`: original boundary frame from the altered clip
- `03_source_warped.png`: source boundary after spatial alignment
- `04_source_corrected.png`: source boundary after spatial and color correction
- `05_output_boundary_encoded.png`: decoded boundary frame from the output video
- `06_encoded_absdiff_amplified.png`: amplified absolute difference image

## Tuning

Start with the defaults. For most continuous-shot seams, `--anchor-frames 6` and
`--crf 8` keep the boundary visually stable while limiting visible re-encoding
loss.

Increase `--ecc-width` if alignment is close but still visibly off. Decrease it
for faster test runs.

Increase `--mask-margin` when frame edges contain generation artifacts, black
borders, watermarks, or partial objects that should not drive alignment.

Lower `--trim-percentile` when the clips contain localized changes at the seam
and the color solve is overfitting to those changing regions.

Use `--no-full-refine` for faster previews. Keep full refinement enabled for the
final output.

Use `--anchor-frames 0` to inspect the pure affine and color correction without
the exact residual seam ramp.

## Troubleshooting

If the tool reports that clip sizes or frame rates differ, normalize both clips
with `ffmpeg` before running Seamstress.

If ECC alignment fails, the boundary frames may not be similar enough for a
continuous-shot correction. Try a larger `--mask-margin`, a larger `--ecc-width`,
or a nearby cut point with less motion.

If the seam is exact in diagnostics but not after encoding, lower `--crf`. H.264
encoding and chroma subsampling can introduce small nonzero differences even
when the unencoded anchored frame is an exact match.

If the output path is the same as either input path, Seamstress exits instead of
overwriting source media. Write to a new path and replace files manually only
after reviewing the result.

## Stabilizer

`stabilizer.py` repairs brief glitches inside a single clip: sudden scale,
stretch, or position pops, stutter/duplicate frames, and frame jumps that break
an otherwise smooth shot. It shares Seamstress's requirements (ffmpeg, ffprobe,
and the Python dependencies from `requirements.txt`).

```sh
./stabilizer.py CLIP [options]
```

By default it scans the whole clip and writes `CLIP_stabilized.mp4`, a
`_report.json` with detections and repair metrics, and a `_diagnostics`
directory containing motion-trace CSVs and before/after PNGs of repaired
frames. Point it at known trouble spots (in seconds) to scan only there with a
more sensitive threshold:

```sh
./stabilizer.py clip.mp4 --range 7.5:9.5
./stabilizer.py clip.mp4 --at 8.4
```

How it works:

1. Estimates full-affine global motion between every consecutive frame pair
   and flags transitions whose motion spikes away from the local trajectory,
   plus duplicate frames and content jumps. Clips that animate on held frames
   (a duplicated-frame cadence) are recognized so the cadence itself is not
   flagged.
2. Re-estimates flagged transitions at full resolution, anchors each glitch
   window with a direct registration across it, and fits a robust
   spatially-varying displacement field per flagged transition, so pops that
   are not a single global transform (for example a stretch that varies across
   the frame) are measured correctly.
3. Warps displaced frames back onto the interpolated trajectory (geometric
   pops), or rebuilds broken frames from their neighbors with
   motion-compensated interpolation (stutters and content jumps). A short ramp
   spreads any leftover step so corrections vanish at the window edges, and
   the rendered output is re-measured at each repaired boundary against the
   clip's own clean transitions.
4. Re-encodes with the same size, frame rate, and audio. Frames outside repair
   windows pass through unmodified, editorial hard cuts are recognized and
   left alone, and residual anomalies below a small pixel floor are not
   "repaired" at all. If nothing needs fixing the output is a lossless remux.
5. Re-analyzes the rendered output and reports the before/after anomaly scores
   per event.

Use `--detect-only` to inspect findings without rendering, `--sensitivity` to
tune detection, `--repair {auto,warp,interp}` to force a strategy, and
`--crf`/`--preset` to control encoding. Run `./stabilizer.py --help` for the
full option list.
