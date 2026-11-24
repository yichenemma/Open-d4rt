"""Recompute static/dynamic labels from saved D4RT window tracks (no D4RT rerun).

Camera-error compensation: in each window, a static point's 3D position in frame-0
coordinates should stay put, but D4RT's errors often shift whole frames together. For each
frame we fit one similarity transform (trimmed Umeyama) that maps the tracks' positions at that
frame back onto their source positions; most tracks are static, so this removes the shared
error. What is left over (divided by scene depth, 90th percentile over the track's frames) is
`drift_comp`; a track is dynamic when drift_comp > --thr. With --mad-k, the threshold is
per window: max(--thr, median + k * MAD of that window's drift_comp), so a track must also be a
clear outlier against the window's own noise (for shaky video where noise varies by window).

--motion-rank PCT switches to the ranking used by the repo's viewer (vis/build_like_demo.py,
_compute_point_motion_scores): score = p90 displacement from the source position + 0.35 * p75
frame-to-frame step + 0.02 * mean confidence, over visible frames, with no camera-error
compensation; the top (100 - PCT)% of tracks in each window are dynamic (PCT=80 marks 20%).
Results go to RUN/relabel_motion_rank/ so the drift labels are kept.

Labels are kept per window, and the stitched trajectories get a per-frame label from the
window that supplies each frame (no majority vote).

Usage: python tracking/relabel.py --run RUN_DIR --frames RGB_DIR --glob PATTERN (--thr T | --motion-rank PCT) [--no-video]
Writes RUN_DIR/relabel/ (or relabel_motion_rank/): labels.npz, summary.json, trajectories_overlay.mp4/.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.tasks import _umeyama_sim3

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--frames", type=Path, required=True)
ap.add_argument("--glob", default="frame_*.png")
ap.add_argument("--thr", type=float, default=None)
ap.add_argument("--motion-rank", "--motion_rank", type=float, default=None, metavar="PCT",
                help="label the top (100-PCT)%% motion scores per window as dynamic (repo viewer heuristic)")
ap.add_argument("--mad-k", type=float, default=None, help="per-window outlier threshold (lab: shaky camera)")
ap.add_argument("--trim", type=float, default=0.6, help="share of tracks kept when fitting each frame")
ap.add_argument("--fps", type=float, default=6.0)
ap.add_argument("--no-video", action="store_true")
args = ap.parse_args()
if (args.thr is None) == (args.motion_rank is None):
    ap.error("give exactly one of --thr or --motion-rank")

OUT = args.run / ("relabel" if args.motion_rank is None else "relabel_motion_rank")
OUT.mkdir(exist_ok=True)
wins = [dict(np.load(p)) for p in sorted((args.run / "windows").glob("window_*.npz"))]


def compensated_drift(w: dict) -> np.ndarray:
    P, T = w["keep"].shape[0], w["tracks_xyz_ref0"].shape[1]
    xyz, src, end, keep = w["tracks_xyz_ref0"].astype(np.float64), w["query_t_src"], w["track_t_end"], w["keep"]
    x_src = xyz[np.arange(P), src]
    t_idx = np.arange(T)[None]
    span = keep[:, None] & (t_idx >= src[:, None]) & (t_idx <= end[:, None])
    res = np.full((P, T), np.nan)
    for t in range(T):
        idx = np.flatnonzero(span[:, t])
        if len(idx) < 10:
            continue
        a, b = xyz[idx, t], x_src[idx]
        use = np.arange(len(idx))
        for _ in range(3):  # trimmed fit: refit on the best-agreeing share of tracks
            fit = _umeyama_sim3(a[use], b[use])
            if fit is None:
                break
            s, R, tr = fit
            r = np.linalg.norm((s * (R @ a.T)).T + tr - b, axis=1)
            use = np.argsort(r)[: max(10, int(len(idx) * args.trim))]
        else:
            res[idx, t] = r
    res /= max(float(w["scene_depth"]), 1e-6)
    with np.errstate(all="ignore"):
        return np.where(keep, np.nanpercentile(np.where(span, res, np.nan), 90, axis=1), np.nan)


def motion_scores(w: dict) -> np.ndarray:
    """Repo viewer score per track, over visible frames from t_src to track end."""
    P, T = w["keep"].shape[0], w["tracks_xyz_ref0"].shape[1]
    xyz, src, end = w["tracks_xyz_ref0"], w["query_t_src"], w["track_t_end"]
    t_idx = np.arange(T)[None]
    valid = (w["keep"][:, None] & (t_idx >= src[:, None]) & (t_idx <= end[:, None]) & w["tracks_visibility"].astype(bool)
             & np.isfinite(xyz).all(-1))
    out = np.full(P, np.nan)
    for i in np.flatnonzero(valid.sum(1) >= 2):
        pts = xyz[i, valid[i]]
        out[i] = (np.percentile(np.linalg.norm(pts - pts[0], axis=-1), 90)
                  + 0.35 * np.percentile(np.linalg.norm(np.diff(pts, axis=0), axis=-1), 75)
                  + 0.02 * float(np.mean(w["tracks_confidence"][i, valid[i]])))
    return out


per_window = []
lab = {}
for w in wins:
    k = int(w["window_index"])
    if args.motion_rank is not None:
        dc = motion_scores(w)
        v = dc[w["keep"] & np.isfinite(dc)]
        thr_w = float(np.percentile(v, args.motion_rank)) if len(v) else np.inf
        mv = w["keep"] & np.isfinite(dc) & (dc >= thr_w)
        lab[f"motion_score_{k:02d}"], lab[f"moving_{k:02d}"] = dc.astype(np.float32), mv
        per_window.append({"window": k, "kept": int(w["keep"].sum()), "dynamic": int(mv.sum()), "thr": round(thr_w, 4),
                           "dynamic_before": int(w["moving"].sum())})
        continue
    dc = compensated_drift(w)
    thr_w = args.thr
    if args.mad_k is not None:
        v = dc[w["keep"] & np.isfinite(dc)]
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826
        thr_w = max(args.thr, med + args.mad_k * mad)
    mv = w["keep"] & (dc > thr_w)
    lab[f"drift_comp_{k:02d}"], lab[f"moving_{k:02d}"] = dc.astype(np.float32), mv
    per_window.append({"window": k, "kept": int(w["keep"].sum()), "dynamic": int(mv.sum()), "thr": round(thr_w, 4),
                       "dynamic_before": int(w["moving"].sum())})

# per-frame labels on the stitched trajectories (same frame ranges as track_recording.py)
st = dict(np.load(args.run / "trajectories_stitched.npz"))
G, F = st["uv_px"].shape[:2]
frame_lo = int(st["frame_numbers"][0])
dyn_pf = np.zeros((G, F), bool)
nxt = [int(w["window_start"]) for w in wins[1:]] + [frame_lo + F]
for w, hand in zip(wins, nxt):
    mv = lab[f"moving_{int(w['window_index']):02d}"]
    s0 = int(w["window_start"]) - frame_lo
    stop = hand - int(w["window_start"])
    for j in np.flatnonzero(w["keep"]):
        a, b = int(w["query_t_src"][j]), int(w["track_t_end"][j])
        if hand < frame_lo + F:
            b = min(b, stop - 1)
        if b >= a:
            dyn_pf[w["global_id"][j], s0 + a:s0 + b + 1] = mv[j]
dyn_pf &= np.isfinite(st["uv_px"][..., 0])
lab.update(dynamic_per_frame=dyn_pf, global_id_order=np.arange(G),
           thr=np.float32(np.nan if args.thr is None else args.thr),
           motion_rank=np.float32(np.nan if args.motion_rank is None else args.motion_rank))
np.savez_compressed(OUT / "labels.npz", **lab)
summary = {"thr": args.thr, "motion_rank": args.motion_rank, "mad_k": args.mad_k, "trim": args.trim, "per_window": per_window,
           "dynamic_track_frames": int(dyn_pf.sum()), "tracks_ever_dynamic": int(dyn_pf.any(1).sum()),
           "tracks": int(G)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_window"}))
print("per window dynamic (before -> after):", [(p["window"], p["dynamic_before"], p["dynamic"]) for p in per_window])

if not args.no_video:
    files = sorted(args.frames.glob(args.glob))
    uv, vis = st["uv_px"], st["visibility"]
    W, H = int(st["image_width"]), int(st["image_height"])
    vw, vh = 960, int(round(960 * H / W)) // 2 * 2
    sc = np.array([vw / W, vh / H])
    STATIC, DYN, TRAIL = (60, 210, 255), (255, 70, 40), 15
    writer = cv2.VideoWriter(str(OUT / "trajectories_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (vw, vh))
    gif = []
    for t in range(F):
        img = cv2.resize(cv2.cvtColor(cv2.imread(str(files[frame_lo + t])), cv2.COLOR_BGR2RGB), (vw, vh))
        alive = np.flatnonzero(np.isfinite(uv[:, t, 0]))
        for g in alive[np.argsort(dyn_pf[alive, t])]:  # draw dynamic tracks on top
            lo = max(0, t - TRAIL)
            pts, d = uv[g, lo:t + 1], dyn_pf[g, lo:t + 1]
            for i in range(1, len(pts)):
                if np.isfinite(pts[i - 1, 0]) and np.isfinite(pts[i, 0]):
                    cv2.line(img, tuple(np.rint(pts[i - 1] * sc).astype(int)), tuple(np.rint(pts[i] * sc).astype(int)),
                             DYN if d[i] else STATIC, 2 if d[i] else 1, cv2.LINE_AA)
            cv2.drawMarker(img, tuple(np.rint(uv[g, t] * sc).astype(int)), DYN if dyn_pf[g, t] else STATIC,
                           cv2.MARKER_CROSS if vis[g, t] else cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.rectangle(img, (0, 0), (vw, 24), (0, 0, 0), -1)
        cv2.putText(img, f"frame {frame_lo + t + 1}  tracks {len(alive)}  dynamic {int(dyn_pf[alive, t].sum())}"
                    f"   blue=static  red=dynamic", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if t % 2 == 0:
            gif.append(cv2.resize(img, (vw // 2, vh // 2)))
    writer.release()
    imageio.mimsave(OUT / "trajectories_overlay.gif", gif, duration=2000 / args.fps, loop=0)
    print("wrote", OUT / "trajectories_overlay.mp4")
