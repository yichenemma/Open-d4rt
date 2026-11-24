# Long-video tracking and static/dynamic labels

Two scripts built on OpenD4RT's `_infer_tracks` (`infer_track_3d.py`):

| Script | Does | GPU |
|---|---|---|
| `track_recording.py` | Tracks feature points through a video of any length, in overlapping 48-frame windows, and stitches them into full-length 2D trajectories | yes |
| `relabel.py` | Labels each track static or dynamic from the saved 3D tracks, without rerunning D4RT. Two methods: `--thr` (drift) and `--motion-rank` | no |

Run both from the repository root with the `d4rt` environment. A checkpoint must be in
`checkpoints/` (see the main README).

## Quick start

```bash
# 1. Track (frames named frame_000000.png, ...; for OMD use --glob "0*.png")
python tracking/track_recording.py --frames path/to/rgb --out output/my_run

# 2a. Relabel with the drift method
python tracking/relabel.py --run output/my_run --frames path/to/rgb --thr 0.03

# 2b. ... or with the motion-rank method
python tracking/relabel.py --run output/my_run --frames path/to/rgb --motion-rank 80
```

## `track_recording.py`

D4RT processes 48 frames at a time, so the video is cut into windows that overlap by
`--overlap` frames (default: frames 0–47, 24–71, 48–95, ...). Each window is one D4RT run:

1. **Hand-over.** Tracks from the previous window that are still in the image at this window's
   first frame are queried again at their predicted pixel positions, and keep their `global_id`.
2. **New points.** At the window's first frame and every `--reseed` frames, Shi-Tomasi corners
   (`--max-corners`) plus a sparse grid (`--grid-step` px) are added, only where no track is.
3. **Tracking.** D4RT tracks every query through the whole window. 3D positions are in the camera
   coordinates of the window's first frame.
4. **Track end.** A track ends at the last frame (from its seed on) where it is still inside the
   image. Tracks shorter than 3 frames are dropped (`keep = False`).
5. **Round trip.** Each track is re-queried from its end back to its seed frame and the distance
   is saved as `roundtrip_px`. It is not a filter by default (`--rt-max`): on our recordings it
   did not predict tracking error.
6. **First label.** `moving` uses the uncorrected drift with `--move-thr` (default 0.15). Use
   `relabel.py` for better labels.

Windows run in order. `--resume` skips windows already saved and rebuilds the hand-over from
them, so an interrupted run can continue.

**Stitching.** All 3D stays per window, since windows have different coordinates and scale.
Only 2D pixel tracks are stitched: each window supplies frames from its start until the next
window starts, and tracks are joined by `global_id`.

Timestamps come from `timestamps.csv` in the frame folder (OAK recordings, columns `file`,
`device_timestamp_ns`) or `times.txt` in its parent folder (OMD, seconds per line, assumed to match
image order). Otherwise they are `-1`.

### Outputs (`--out OUT`)

| File | Contents |
|---|---|
| `windows/window_KK.npz` | One window. D4RT outputs, point first: `tracks_uv_norm [P,T,2]`, `tracks_xyz_local [P,T,3]`, `tracks_xyz_ref0 [P,T,3]`, `tracks_visibility [P,T]`, `tracks_visibility_logits`, `tracks_confidence [P,T]`. Plus `query_uv_px`, `query_t_src`, `track_t_end`, `keep`, `global_id`, `carried`, `roundtrip_px`, `drift_p90`, `moving`, `scene_depth`, `window_index`, `window_start`, `frame_numbers`, `frame_files`, `timestamps_ns`, `image_width/height` |
| `trajectories_stitched.npz` | `uv_px [G,F,2]` (pixels, NaN where the track does not exist), `visibility [G,F]`, `moving [G]`, `length_frames`, `first_window`, `frame_numbers`, `frame_files`, `timestamps_ns`. Row `g` is `global_id == g` |
| `summary.json` | Per-window counts, hand-over counts, track lengths |
| `trajectories_overlay.mp4/.gif` | Trails on the video: blue = static, red = moving, tilted cross = hidden |

3D units are D4RT's arbitrary scale, not metres; axes are OpenCV (x right, y down, z forward).

### Main options

| Option | Default | Meaning |
|---|---|---|
| `--frames` | required | Folder of frames |
| `--out` | required | Output folder |
| `--glob` | `frame_*.png` | Frame file pattern |
| `--window` / `--overlap` | 48 / 24 | Window length and overlap, in frames |
| `--reseed` | 12 | Add new points every N frames |
| `--max-corners` / `--grid-step` | 150 / 120 | New corners per seed frame / grid spacing in px |
| `--windows` | all | Only these window indices |
| `--resume` | off | Skip windows already saved |
| `--checkpoint` | `OpenD4RT_48CLIP_9Mix_NoCropAUG` | Checkpoint folder, or its name under `checkpoints/` |
| `--chunk` | 2048 | Queries per D4RT call (lower it if you run out of GPU memory) |
| `--fps` | 6 | Overlay video frame rate |

## `relabel.py`

D4RT itself has no static/dynamic output. Both methods turn its 3D tracks into a label, one
window at a time, and write to a new folder so the run's original files are untouched.

Notation for one window: $\mathbf{X}_i(t)$ is track $i$'s 3D position at frame $t$ in the
window's first-frame camera coordinates (`tracks_xyz_ref0`), $s_i$ its seed frame and $e_i$
its last frame. In these coordinates a static point should stay where it was seeded.

### Method 1: drift (`--thr T`)

1. **Remove the error shared by a whole frame.** If D4RT misjudges the camera motion at
   frame $t$, every point in that frame shifts together. For each frame, a similarity transform
   (scale, rotation, translation; Umeyama) is fitted that moves the points back onto their seed
   positions. Since moving points would bias it, the fit is trimmed: fit on all points, keep the
   60% (`--trim`) that fit best, refit, three rounds in total. Most points are static, so the
   final fit describes the shared error only.
2. **Leftover per frame, relative to scene depth.**
   $r_i(t) = \lVert \sigma_t R_t \mathbf{X}_i(t) + \boldsymbol{\tau}_t - \mathbf{X}_i(s_i) \rVert / \bar d$,
   where $\bar d$ is the window's median depth (`scene_depth`). Dividing by it removes D4RT's
   arbitrary, per-window scale: 0.03 means 3% of the typical scene distance in any window.
3. **One score per track.** $\text{drift}_i$ is the 90th percentile of $r_i(t)$ over
   $s_i \le t \le e_i$, so a few bad frames do not decide the label.
4. **Decide.** Dynamic if $\text{drift}_i > \theta_w$. By default $\theta_w$ = `--thr` in every
   window. With `--mad-k k`, a noisy window raises its own threshold:
   $\theta_w = \max(\texttt{thr},\ \text{median}_w + k \cdot 1.4826 \cdot \text{MAD}_w)$,
   so a track must also be a clear outlier against that window's noise (useful for shaky video).

Writes `RUN/relabel/`.

### Method 2: motion rank (`--motion-rank PCT`)

The ranking used by the repository's viewer (`vis/build_like_demo.py`,
`_compute_point_motion_scores`). Over the frames where the track is visible,
$t_1 < \dots < t_n$:

$$
m_i = \mathrm{P}_{90}\,\lVert \mathbf{X}_i(t) - \mathbf{X}_i(t_1) \rVert
+ 0.35\,\mathrm{P}_{75}\,\lVert \mathbf{X}_i(t_{j+1}) - \mathbf{X}_i(t_j) \rVert
+ 0.02\,\overline{c_i}
$$

The first term (distance from where the track was first seen) dominates; the second is the step
between consecutive visible frames; the last is mean confidence. There is no correction of shared
error and no depth normalization. In each window, tracks with $m_i \ge$ the PCT-th percentile are
dynamic, so `--motion-rank 80` always marks the top 20%, whether or not anything moves.

Writes `RUN/relabel_motion_rank/`.

### Which to use

On OMD `swinging_4_unconstrained`, scored against the sequence's object masks at each track's seed
pixel (the masks are probably network-made, so this measures agreement, not exact accuracy):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `--thr 0.03` | 0.78 | 0.68 | 0.72 |
| `--motion-rank 80` | 0.82 | 0.54 | 0.65 |

Drift finds slow-moving objects that motion rank misses. Motion rank marks slightly less
background, but assumes something moves in every window, which fails on mostly static video.

### Outputs

| File | Contents |
|---|---|
| `labels.npz` | Per window: `moving_KK` (bool per track, indexed like that window's tracks) and `drift_comp_KK` or `motion_score_KK`. Stitched: `dynamic_per_frame [G,F]`, the label from the window that supplies each frame. Also `thr`, `motion_rank` |
| `summary.json` | Threshold used in each window and dynamic counts before/after |
| `trajectories_overlay.mp4/.gif` | Trails: red = dynamic, blue = static at that frame (skip with `--no-video`) |

### Options

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Output folder of `track_recording.py` |
| `--frames` / `--glob` | required / `frame_*.png` | Frames, for the overlay video |
| `--thr T` | — | Drift method with threshold T (give this or `--motion-rank`) |
| `--mad-k k` | off | Drift only: per-window threshold |
| `--trim` | 0.6 | Drift only: share of points kept in each fit round |
| `--motion-rank PCT` | — | Motion-rank method: top (100 − PCT)% per window are dynamic |
| `--fps` | 6 | Overlay video frame rate |
| `--no-video` | off | Skip the overlay video |

## Examples

```bash
# OAK recording, shaky handheld camera
python tracking/track_recording.py --frames data/recording/rgb --out output/lab
python tracking/relabel.py --run output/lab --frames data/recording/rgb --thr 0.03 --mad-k 8

# OMD sequence
python tracking/track_recording.py --frames data/omd/swinging_4_unconstrained/image_0 \
    --glob "0*.png" --out output/omd --fps 20
python tracking/relabel.py --run output/omd --frames data/omd/swinging_4_unconstrained/image_0 \
    --glob "0*.png" --thr 0.03 --fps 20
python tracking/relabel.py --run output/omd --frames data/omd/swinging_4_unconstrained/image_0 \
    --glob "0*.png" --motion-rank 80 --fps 20
```
