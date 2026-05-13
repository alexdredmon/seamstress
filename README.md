# Joiner Seam Scripts

These scripts fix visible one-frame joins by measuring the exact boundary
frames, estimating the crop/translation/color difference, re-rendering the clip
being altered, and applying a short residual ramp at the seam.

## Setup

From the project root:

```sh
venv/bin/python -m pip install -r joiner/requirements.txt
```

Both scripts also require `ffmpeg` and `ffprobe` on `PATH`.

## `make_end_seamless.py`

Use this when the clip being altered comes before the reference clip.

Default boundary:

```text
00-02.mp4 -> 01-03.mp4
```

Run:

```sh
venv/bin/python joiner/make_end_seamless.py
```

Default output:

```text
joiner/00-02_seamless.mp4
joiner/end_seamless_report.json
```

Current verification:

```text
raw boundary:         MAE 10.2315, RMSE 20.4067
corrected boundary:   MAE 3.8621, RMSE 6.8671
anchored final frame: MAE 0.0000, RMSE 0.0000
encoded output frame: MAE 1.5242, RMSE 1.8929
```

Timeline-style concat check:

```text
joiner/diagnostics/seam_test_00-02_to_01-03_copy.mp4
concat boundary: MAE 1.5242, RMSE 1.8929
```

## `make_seamless.py`

Use this when the clip being altered comes after the reference clip.

Default boundary:

```text
01-03.mp4 -> 02-01_matched.mp4
```

Run:

```sh
venv/bin/python joiner/make_seamless.py
```

Default output:

```text
joiner/02-01_seamless.mp4
joiner/seamless_report.json
```

Current verification:

```text
raw boundary:         MAE 14.2157, RMSE 28.4874
corrected boundary:   MAE 4.5452, RMSE 7.8858
anchored frame 0:     MAE 0.0000, RMSE 0.0000
encoded output frame: MAE 1.4705, RMSE 1.7868
```

Timeline-style concat check:

```text
joiner/diagnostics/seam_test_copy.mp4
concat boundary: MAE 1.4705, RMSE 1.7868
```

## Full Sequence Check

The corrected three-clip sequence was also copy-concatenated as:

```text
joiner/diagnostics/seam_test_full_copy.mp4
```

Measured boundaries:

```text
00-02_seamless.mp4 -> 01-03.mp4:        MAE 1.5242, RMSE 1.8929
01-03.mp4 -> 02-01_seamless.mp4:        MAE 1.4705, RMSE 1.7868
```

## Useful Options

Both scripts accept:

```sh
--reference PATH
--source PATH
--output PATH
--report PATH
--anchor-frames N
--crf N
--preset NAME
--no-diagnostics
```

For these clips the default `--anchor-frames 6` and `--crf 8` produced the
cleanest tested seams. The nonzero final MAE is from H.264 re-encoding and
chroma subsampling; the unencoded anchored seam is an exact frame match.
