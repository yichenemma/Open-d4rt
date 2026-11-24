"""Windowed D4RT feature tracking on a long video (a folder of frames).

Per 48-frame window (overlapping by --overlap frames):
  1. Hand over: tracks from the previous window that are still in the image at this
     window's frame 0 are queried again at their predicted positions, keeping their global id.
     Then add Shi-Tomasi corners plus a sparse grid at frame 0, and every --reseed frames,
     only where no existing track is (feature-tracker style coverage).
  2. Track every point through the whole window with D4RT (window frame 0 = 3D reference).
  3. Round-trip check: track from the source frame to the last in-image frame and back,
     and record how far each point comes back. It is not used as a filter by default: on this
     recording it did not predict disagreement with Lucas-Kanade (rank correlation 0.08).
     Tracks are kept when they stay in the image for at least 3 frames (--rt-max filters).
  4. Label kept tracks moving when their 3D drift (in window frame-0 coordinates, divided
     by the median scene depth) exceeds --move-thr at the 90th percentile.
Windows run in order, so each track keeps one global id across windows. Tracks are then
stitched into full-length 2D trajectories (each window supplies frames from its start until
the next window takes over) and drawn into an overlay video.
Matching tracks between windows by distance does not work here: the same point predicted in
two windows is typically 25-40 px apart, more than the spacing between neighbouring points.

Usage: python tracking/track_recording.py --frames RGB_DIR --out OUT_DIR [--windows 0 1 ...]
See tracking/README.md for the outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from infer_track_3d import _infer_tracks, _resize_video, _unwrap_state_dict
from src.core import load_checkpoint, load_yaml_config
from src.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--window", type=int, default=48)
ap.add_argument("--overlap", type=int, default=24)
ap.add_argument("--reseed", type=int, default=12)
ap.add_argument("--windows", type=int, nargs="*", default=None, help="only these window indices")
ap.add_argument("--max-corners", type=int, default=150, help="new corners per seed frame")
ap.add_argument("--grid-step", type=int, default=120, help="sparse grid spacing, full-res px")
ap.add_argument("--rt-max", type=float, default=float("inf"), help="optional round-trip limit, full-res px")
ap.add_argument("--move-thr", type=float, default=0.15,
                help="3D drift / scene depth; windows without people stay below ~0.13 at p99")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--glob", default="frame_*.png", help="frame file pattern (OMD: 0*.png)")
ap.add_argument("--resume", action="store_true", help="skip windows already saved, rebuilding the hand-over")
ap.add_argument("--checkpoint", default="OpenD4RT_48CLIP_9Mix_NoCropAUG",
                help="folder with model.yaml and opend4rt.ckpt, or its name under checkpoints/")
ap.add_argument("--fps", type=float, default=6.0, help="overlay video fps (~real time)")
args = ap.parse_args()

files = sorted(args.frames.glob(args.glob))
N = len(files)
stamps = {}
if (args.frames / "timestamps.csv").exists():  # OAK recordings
    with open(args.frames / "timestamps.csv") as f:
        for row in csv.DictReader(f):
            stamps[row["file"]] = int(row["device_timestamp_ns"])
elif (args.frames.parent / "times.txt").exists():  # OMD: one time in seconds per frame
    secs = np.loadtxt(args.frames.parent / "times.txt", ndmin=1)
    stamps = {f.name: int(round(t * 1e9)) for f, t in zip(files, secs)}
H, W = cv2.imread(str(files[0])).shape[:2]
SZ = np.array([W - 1, H - 1], np.float32)
WIN_DIR = args.out / "windows"
WIN_DIR.mkdir(parents=True, exist_ok=True)

step = args.window - args.overlap
starts = list(range(0, N - args.window + 1, step))
if starts[-1] != N - args.window:
    starts.append(N - args.window)
todo = range(len(starts)) if args.windows is None else sorted(args.windows)
print(f"{N} frames {W}x{H}, {len(starts)} windows of {args.window} (overlap {args.overlap})", flush=True)

ckpt = Path(args.checkpoint) if Path(args.checkpoint).is_dir() else REPO / "checkpoints" / args.checkpoint
cfg = load_yaml_config(str(ckpt / "model.yaml"))
cfg["model"]["encoder"]["pretrained"]["enabled"] = False
model = build_model(cfg["model"]).eval().cuda()
model.load_state_dict(_unwrap_state_dict(load_checkpoint(ckpt / "opend4rt.ckpt", map_location="cpu")), strict=False)
IMG_HW = tuple(int(v) for v in cfg.get_path("model.input.image_size"))


def inside(uv: np.ndarray, margin: float = 2.0) -> np.ndarray:
    return (uv[..., 0] >= margin) & (uv[..., 0] <= W - 1 - margin) & (uv[..., 1] >= margin) & (uv[..., 1] <= H - 1 - margin)


def run(video: np.ndarray, uv_px: np.ndarray, t_src: np.ndarray) -> dict:
    return _infer_tracks(model=model, video_model_rgb=video, native_aspect_ratio=W / H,
                         query_uv_norm=(np.clip(uv_px, 0, SZ) / SZ).astype(np.float32),
                         query_chunk_size=args.chunk, query_src_indices_global=t_src.astype(np.int64))


def new_points(img: np.ndarray, occupied_uv: np.ndarray) -> np.ndarray:
    """Corners plus sparse grid points, away from existing tracks (full-res px)."""
    small = cv2.cvtColor(cv2.resize(img, (W // 2, H // 2)), cv2.COLOR_RGB2GRAY)
    mask = np.full(small.shape, 255, np.uint8)
    for x, y in occupied_uv / 2:
        cv2.circle(mask, (int(x), int(y)), 15, 0, -1)
    c = cv2.goodFeaturesToTrack(small, args.max_corners, 0.01, 12, mask=mask, blockSize=7)
    corners = c.reshape(-1, 2) * 2 if c is not None else np.empty((0, 2))
    for x, y in corners / 2:
        cv2.circle(mask, (int(x), int(y)), 15, 0, -1)
    ys, xs = np.mgrid[args.grid_step // 2:H:args.grid_step, args.grid_step // 2:W:args.grid_step]
    grid = np.stack([xs.ravel(), ys.ravel()], -1)
    grid = grid[mask[grid[:, 1] // 2, grid[:, 0] // 2] > 0]
    return np.concatenate([corners, grid]).astype(np.float32)


def track_window(k: int, carry: dict | None, next_id: int) -> tuple[dict, int]:
    s0 = starts[k]
    idx = list(range(s0, s0 + args.window))
    frames = np.stack([cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB) for i in idx])
    video = _resize_video(frames, IMG_HW)
    T = len(idx)
    t0 = time.time()

    keys = ("tracks_uv_norm", "tracks_xyz_local", "tracks_xyz_ref0", "tracks_visibility",
            "tracks_visibility_logits", "tracks_confidence")
    acc = {k_: [] for k_ in keys}
    q_uv, q_src, gids = [], [], []
    if carry is not None and len(carry["uv"]):
        out = run(video, carry["uv"], np.zeros(len(carry["uv"])))
        for k_ in keys:
            acc[k_].append(out[k_])
        q_uv.append(carry["uv"]); q_src.append(np.zeros(len(carry["uv"]))); gids.append(carry["gid"])
    n_carried = len(carry["uv"]) if carry is not None else 0
    for s in range(0, T, args.reseed):
        if acc["tracks_uv_norm"]:
            uv_s = np.concatenate(acc["tracks_uv_norm"])[:, s] * SZ
            occ = uv_s[inside(uv_s)]
        else:
            occ = np.empty((0, 2))
        pts = new_points(frames[s], occ)
        if len(pts) == 0:
            continue
        out = run(video, pts, np.full(len(pts), s))
        for k_ in keys:
            acc[k_].append(out[k_])
        q_uv.append(pts); q_src.append(np.full(len(pts), s))
        gids.append(np.arange(next_id, next_id + len(pts))); next_id += len(pts)
    arr = {k_: np.concatenate(v) for k_, v in acc.items()}
    q_uv, q_src = np.concatenate(q_uv), np.concatenate(q_src).astype(np.int64)
    gid = np.concatenate(gids).astype(np.int64)
    P = len(q_uv)
    uv = arr["tracks_uv_norm"] * SZ
    t_idx = np.arange(T)[None]

    # last frame (from the source on) where the point is still inside the image
    ins = inside(uv) & (t_idx >= q_src[:, None])
    run_ok = np.cumprod(ins[np.arange(P)[:, None], np.clip(q_src[:, None] + t_idx, 0, T - 1)]
                        * ((q_src[:, None] + t_idx) < T), axis=1)
    t_end = q_src + run_ok.sum(1) - 1

    back = run(video, uv[np.arange(P), t_end], t_end)
    rt = np.linalg.norm(back["tracks_uv_norm"][np.arange(P), q_src] * SZ - q_uv, axis=1)
    long_enough = (t_end - q_src) >= 3
    keep = long_enough & (rt <= args.rt_max)

    # 3D drift in window frame-0 coordinates
    xyz = arr["tracks_xyz_ref0"]
    span = (t_idx >= q_src[:, None]) & (t_idx <= t_end[:, None])
    depth = np.abs(arr["tracks_xyz_local"][np.arange(P), q_src, 2])
    scale = float(np.median(depth[keep])) if keep.any() else float(np.median(depth))
    d = np.linalg.norm(xyz - xyz[np.arange(P), q_src][:, None], axis=-1) / max(scale, 1e-6)
    d[~span] = np.nan
    drift_p90 = np.where(long_enough, np.nanpercentile(np.where(span, d, np.nan), 90, axis=1), np.nan)
    moving = keep & (drift_p90 > args.move_thr)

    np.savez_compressed(
        WIN_DIR / f"window_{k:02d}.npz", **arr,
        query_uv_px=q_uv, query_t_src=q_src, track_t_end=t_end,
        roundtrip_px=rt.astype(np.float32), keep=keep, drift_p90=drift_p90.astype(np.float32),
        moving=moving, scene_depth=np.float32(scale), global_id=gid, carried=np.arange(P) < n_carried,
        window_index=np.int64(k), window_start=np.int64(s0),
        frame_numbers=np.array(idx), frame_files=np.array([files[i].name for i in idx]),
        timestamps_ns=np.array([stamps.get(files[i].name, -1) for i in idx], np.int64),
        image_width=np.int64(W), image_height=np.int64(H),
    )
    print(f"window {k:02d} frames {s0}-{s0 + T - 1}: {P} tracks, kept {keep.sum()} "
          f"({keep.mean():.0%}), moving {moving.sum()}, round trip median {np.median(rt[long_enough]):.1f}px, "
          f"carried in {n_carried}, {time.time() - t0:.0f}s", flush=True)

    # hand over kept tracks still in the image at the next window's first frame
    nxt = k + 1
    if nxt >= len(starts):
        return None, next_id
    off = starts[nxt] - s0
    alive = keep & (q_src <= off) & (t_end >= off)
    return {"uv": uv[alive, off].astype(np.float32), "gid": gid[alive], "from_window": k}, next_id


def carry_from_saved(k: int) -> tuple[dict | None, int]:
    """Rebuild the hand-over out of a saved window file (for --resume)."""
    w = np.load(WIN_DIR / f"window_{k:02d}.npz")
    nid = int(w["global_id"].max()) + 1
    if k + 1 >= len(starts):
        return None, nid
    off = starts[k + 1] - int(w["window_start"])
    alive = w["keep"] & (w["query_t_src"] <= off) & (w["track_t_end"] >= off)
    uv = w["tracks_uv_norm"][alive, off] * SZ
    return {"uv": uv.astype(np.float32), "gid": w["global_id"][alive], "from_window": k}, nid


carry, next_id = None, 0
for k in todo:
    if carry is not None and carry["from_window"] != k - 1:
        carry = None  # windows not consecutive: no hand-over
    if args.resume and (WIN_DIR / f"window_{k:02d}.npz").exists():
        carry, next_id = carry_from_saved(k)
        print(f"window {k:02d}: already saved, resuming after it", flush=True)
        continue
    carry, next_id = track_window(k, carry, next_id)

# ------------------------------------------------------------------ stitching --
done = sorted(WIN_DIR.glob("window_*.npz"))
wins = [dict(np.load(p)) for p in done]
gid_of = [np.where(w["keep"], w["global_id"], -1) for w in wins]
link_stats = [{"window": int(w["window_index"]), "carried_in": int(w["carried"].sum()),
               "carried_kept": int((w["carried"] & w["keep"]).sum()), "kept": int(w["keep"].sum())} for w in wins]
next_id = int(max((g.max() for g in gid_of if g.size), default=-1)) + 1
frame_lo = int(wins[0]["window_start"]) if wins else 0
frame_hi = int(wins[-1]["frame_numbers"][-1]) + 1 if wins else 0
F = frame_hi - frame_lo
G = next_id
nxt_start = [int(w2["window_start"]) for w2 in wins[1:]] + [frame_hi]
uv_all = np.full((G, F, 2), np.nan, np.float32)
vis_all = np.zeros((G, F), bool)
mov_votes = np.zeros((G, 2), np.int64)
first_window = np.full(G, -1, np.int64)
for w, gid, hand in zip(wins, gid_of, nxt_start):
    s0 = int(w["window_start"]) - frame_lo
    stop = hand - int(w["window_start"])  # this window supplies frames until the next one starts
    for j in np.flatnonzero(gid >= 0):
        g = gid[j]
        a, b = int(w["query_t_src"][j]), min(int(w["track_t_end"][j]), stop - 1 if hand < frame_hi else 10**9)
        if b < a:
            continue
        sl = slice(s0 + a, s0 + b + 1)
        uv_all[g, sl] = w["tracks_uv_norm"][j, a:b + 1] * SZ
        vis_all[g, sl] = w["tracks_visibility"][j, a:b + 1]
        mov_votes[g, int(w["moving"][j])] += 1
        if first_window[g] < 0:
            first_window[g] = int(w["window_index"])
moving_all = mov_votes[:, 1] > mov_votes[:, 0]
lengths = np.isfinite(uv_all[..., 0]).sum(1)
np.savez_compressed(
    args.out / "trajectories_stitched.npz", uv_px=uv_all, visibility=vis_all, moving=moving_all,
    length_frames=lengths, first_window=first_window,
    frame_numbers=np.arange(frame_lo, frame_hi), frame_files=np.array([files[i].name for i in range(frame_lo, frame_hi)]),
    timestamps_ns=np.array([stamps.get(files[i].name, -1) for i in range(frame_lo, frame_hi)], np.int64),
    image_width=np.int64(W), image_height=np.int64(H),
)
summary = {
    "frames": [frame_lo, frame_hi - 1], "windows": [int(w["window_index"]) for w in wins],
    "per_window": [{"window": int(w["window_index"]), "start": int(w["window_start"]),
                    "tracks": int(len(w["keep"])), "kept": int(w["keep"].sum()), "moving": int(w["moving"].sum()),
                    "roundtrip_median_px": float(np.median(w["roundtrip_px"][w["track_t_end"] - w["query_t_src"] >= 3]))}
                   for w in wins],
    "handover": link_stats, "global_tracks": int(G), "moving_global_tracks": int(moving_all.sum()),
    "track_length_frames": {"median": float(np.median(lengths)) if G else 0, "p90": float(np.percentile(lengths, 90)) if G else 0,
                            "max": int(lengths.max()) if G else 0},
}
(args.out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_window"}), flush=True)

# -------------------------------------------------------------------- video --
STATIC, MOVING, TRAIL = (60, 210, 255), (255, 70, 40), 15
vid_w, vid_h = 960, 540
s = np.array([vid_w / W, vid_h / H])
writer = cv2.VideoWriter(str(args.out / "trajectories_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (vid_w, vid_h))
gif = []
for t in range(F):
    img = cv2.resize(cv2.cvtColor(cv2.imread(str(files[frame_lo + t])), cv2.COLOR_BGR2RGB), (vid_w, vid_h))
    alive = np.flatnonzero(np.isfinite(uv_all[:, t, 0]))
    for g in alive:
        col = MOVING if moving_all[g] else STATIC
        seg = uv_all[g, max(0, t - TRAIL):t + 1]
        seg = seg[np.isfinite(seg[:, 0])]
        if len(seg) > 1:
            cv2.polylines(img, [np.rint(seg * s).astype(np.int32)], False, col, 2 if moving_all[g] else 1, cv2.LINE_AA)
        p = tuple(np.rint(uv_all[g, t] * s).astype(int))
        cv2.drawMarker(img, p, col, cv2.MARKER_CROSS if vis_all[g, t] else cv2.MARKER_TILTED_CROSS, 8, 1)
    cv2.rectangle(img, (0, 0), (vid_w, 24), (0, 0, 0), -1)
    cv2.putText(img, f"frame {frame_lo + t + 1}  tracks {len(alive)}  moving {int(moving_all[alive].sum())}"
                f"   blue=static  red=moving  x=hidden", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if t % 2 == 0:
        gif.append(cv2.resize(img, (480, 270)))
writer.release()
imageio.mimsave(args.out / "trajectories_overlay.gif", gif, duration=2000 / args.fps, loop=0)
print("wrote", args.out / "trajectories_overlay.mp4", flush=True)
