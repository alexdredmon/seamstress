# Seamstress

Seamstress makes the cuts between separately generated clips of one continuous
shot invisible. Point it at two or more clips (or a whole directory) and it
fixes every seam automatically — no configuration, no tuning:

```sh
./seamstress.py previous.mp4 next.mp4
```

```sh
./seamstress.py clip01.mp4 clip02.mp4 clip03.mp4 --concat final.mp4
```

```sh
./seamstress.py my_take_folder/ --concat final.mp4
```

It is built for the Seedance-style workflow where each clip is generated as a
continuation of the previous clip's final frame and the generations come back
with slightly different color grades, subtle spatial drift, and small changes
to scene elements.

## How it works

For every seam, Seamstress analyzes the boundary frame pair and fits a stack
of corrections, each validated against the seam before it is kept:

1. **Auto-conform.** If the clips differ in resolution or frame rate, the
   altered clip is rescaled and retimed to match before analysis.
2. **Global alignment.** A guarded ECC estimate finds the affine drift between
   the generations (falling back to translation, then identity, and rejecting
   implausible fits).
3. **Dense correspondence.** DIS optical flow with a forward/backward
   consistency check maps each pixel of one boundary frame onto the other.
   Pixels that do not correspond — elements the new generation changed — are
   detected and excluded from every fit, so the correction never tries to
   force content that genuinely differs.
4. **Three-layer color model**, fit only on validated corresponding pixels:
   - a robust, outlier-trimmed RGB matrix solved in linear light,
   - smooth confidence-weighted per-channel refinement curves,
   - a spatially varying low-frequency residual field (masked, diffusion
     inpainted, smoothed) that corrects region-dependent mismatch a global
     transform cannot represent — vignetting, corner tints, uneven exposure.
5. **Seam anchoring that travels with the content.** A decaying dense
   geometric morph absorbs the subtle element shifts between the two
   generations, and the remaining detail residual is advected frame by frame
   with optical flow while it eases out. The boundary frame matches exactly
   and the correction moves with the picture instead of ghosting over it —
   the failure mode of static residual ramps.
6. **Dithered re-render.** The alignment and color model apply to the whole
   clip (so the grade stays consistent for the next seam in a chain); the
   morph, field, and detail layers ease out on smoothstep schedules derived
   from the frame rate. Frames are quantized with triangular-PDF dither to
   prevent banding and encoded with libx264 at CRF 10 by default. Audio is
   copied through untouched.

Every stage measures the seam delta-E before and after itself and reverts to
the safer model if it did not improve, so unusual footage degrades gracefully
to simpler corrections instead of producing artifacts.

## Requirements

```sh
python -m pip install -r requirements.txt
```

`ffmpeg` and `ffprobe` must be on `PATH`.

## Usage

```text
./seamstress.py CLIP [CLIP ...] [options]
```

Give clips in timeline order. Directories expand to their video files in name
order. With more than two clips, Seamstress chains: each seam is measured
against the *corrected* predecessor, so corrections compound properly across
a long take.

Directories that follow the Sora `vid_clips` naming convention —
`SEGMENT-VERSION.mp4`, e.g. `00-01.mp4`, `00-02.mp4`, `01-01.mp4` — are
resolved automatically: one take per segment, chained in segment order. The
highest version of each segment wins unless a lower-numbered take was
modified meaningfully later (a re-render), in which case the fresher file is
used. Every choice is printed, and you can always pass explicit files to
override.

For each corrected clip it writes:

```text
<clip>_seamless.mp4
<clip>_seamless_report.json
<clip>_seamless_diagnostics/
```

The diagnostics directory includes `seam_preview_before_after.mp4` — the
original seam on top, the corrected seam below, slowed 3x — so you can judge
the fix in two seconds.

### Options

- `--concat PATH`: also write the fully stitched timeline (first clip plus
  all corrected clips).
- `--outdir DIR`: put corrected clips, reports, and diagnostics here instead
  of next to each input.
- `-o, --output PATH`: explicit output path (two-clip mode only).
- `--alter {next,previous}`: which side of the seam to re-render (two-clip
  mode only). Default: `next` — the earlier clip is usually already approved.
- `--sort {given,name,mtime}`: ordering for the clip list. Default: `given`.
- `--blend-seconds SECONDS`: overall duration scale for easing the seam
  corrections out. Default: `1.0`.
- `--crf N` / `--preset NAME`: libx264 quality settings. Defaults: `10`,
  `slow`.
- `--no-diagnostics`: skip diagnostic PNGs and the preview video.
- `--mask-margin PIXELS`, `--ecc-width PIXELS`, `--seed N`: analysis
  internals; the defaults are right for essentially all footage.

### Reading the output

Each seam prints a one-line breakdown of how much every layer contributed,
as mean Lab delta-E over the validated seam pixels:

```text
seam dE mean: raw 11.87 -> matrix 4.89 -> curves 4.84 -> field 4.08 -> anchored 0.41
```

and reports the seam error measured back from the encoded file. The JSON
report records the same metrics plus the alignment model, dense-correspondence
statistics, the fraction of the frame detected as changed content, blend
windows, and every setting used.

## Troubleshooting

- **The report says `matrix_reverted`, `curves_reverted`, or
  `field_reverted`.** That layer made the seam worse on this footage and was
  disabled automatically — usually a sign the boundary frames barely
  correspond (a real scene cut rather than a continuation).
- **Low `valid_fraction` in the report.** The two boundary frames share
  little content; Seamstress falls back toward global-only correction. Check
  that the clips really are adjacent generations of the same shot.
- **Banding or blocking near the seam after your NLE re-exports.** Seamstress
  itself dithers; keep your downstream export quality high (CRF <= 16) to
  preserve it.
- **Odd-dimension reference clip.** Re-export it with even width and height;
  yuv420p encoding requires it.
