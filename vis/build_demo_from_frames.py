#!/usr/bin/env python3
"""Build a Viser demo package from an ordered folder of RGB image frames."""

from __future__ import annotations

import argparse
import colorsys
import gc
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _resolve_device, _resize_video, _unwrap_state_dict
from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.model import build_model
from vis.build_like_demo import (
    _build_uv_grid,
    _export_demo_data,
    _export_video_from_frames,
    _jsonable_float_array,
    _predict_camera_branches,
)
from vis.build_like_demo_for_worldtrack import export_predicted_depth_video


def _disable_encoder_pretrain(cfg: Any) -> None:
    encoder_cfg = cfg.get_path("model.encoder", {})
    if isinstance(encoder_cfg, dict):
        pretrained_cfg = encoder_cfg.setdefault("pretrained", {})
        if isinstance(pretrained_cfg, dict):
            pretrained_cfg["enabled"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, help="Folder containing RGB image frames.")
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--ckpt-path", required=True, help="Model checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output Viser package directory.")
    parser.add_argument("--num-frames", type=int, default=16, help="Uniformly sample at most this many frames.")
    parser.add_argument("--sample-mode", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--fps", type=float, default=1.0, help="Playback FPS for uniformly sampled frames.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--point-grid-cols", type=int, default=12)
    parser.add_argument("--point-grid-rows", type=int, default=12)
    parser.add_argument("--point-max-points", type=int, default=144)
    parser.add_argument("--track-max-points", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--point-query-chunk-size", type=int, default=64)
    parser.add_argument("--camera-grid-size", type=int, default=8)
    parser.add_argument("--camera-query-chunk-size", type=int, default=32)
    parser.add_argument("--query-selection", choices=("grid", "features"), default="grid",
                        help="Select D4RT query locations from a regular grid or image corners.")
    parser.add_argument("--feature-min-distance", type=float, default=18.0,
                        help="Minimum spacing in input pixels between detected feature queries.")
    parser.add_argument("--track-min-visible-frames", type=int, default=3)
    return parser.parse_args()


def load_rgb_frames(frames_dir: Path, num_frames: int, sample_mode: str) -> tuple[np.ndarray, list[str]]:
    paths = sorted(
        path for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No PNG/JPG frames found in {frames_dir}")
    if int(num_frames) > 0 and len(paths) > int(num_frames):
        if sample_mode == "uniform":
            indices = np.linspace(0, len(paths) - 1, num=int(num_frames), dtype=np.int64)
            paths = [paths[int(i)] for i in indices]
        else:
            paths = paths[: int(num_frames)]

    frames = []
    expected_size = None
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode frame {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        size = tuple(rgb.shape[:2])
        if expected_size is not None and size != expected_size:
            raise RuntimeError(f"Frame size mismatch at {path}: {size} vs {expected_size}")
        expected_size = size
        frames.append(rgb)
    return np.stack(frames, axis=0), [path.name for path in paths]


def _detect_feature_queries(image_rgb: np.ndarray, *, max_points: int, min_distance: float) -> np.ndarray:
    """Return Shi-Tomasi corner locations, strongest first, in image pixel coordinates."""
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(max_points)),
        qualityLevel=0.01,
        minDistance=max(1.0, float(min_distance)),
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def render_prediction_videos(
    video_rgb: np.ndarray,
    package: dict[str, Any],
    assets_dir: Path,
    fps: float,
) -> dict[str, str]:
    """Render prediction-only 2D and 3D track videos (no GT required)."""
    track_xyz = np.asarray(package["track_xyz_ref0"], dtype=np.float32)
    track_vis = np.asarray(package["track_visibility"], dtype=bool)
    track_uv = np.asarray(package["track_uv_px"], dtype=np.float32)
    frame_count = int(video_rgb.shape[0])
    # _infer_tracks returns point-first arrays [P, T, ...]; no transpose needed (guessing the
    # layout from shapes flips the data when the track count equals the frame count).
    colors = np.asarray([
        [round(v * 255) for v in colorsys.hsv_to_rgb(i / max(track_xyz.shape[0], 1), 0.75, 1.0)]
        for i in range(track_xyz.shape[0])
    ], dtype=np.uint8)

    overlay_frames = []
    for t in range(frame_count):
        frame = np.asarray(video_rgb[t], dtype=np.uint8).copy()
        for q in range(track_xyz.shape[0]):
            color = tuple(int(v) for v in colors[q])
            start = max(0, t - 12)
            for prev in range(start, t):
                if track_vis[q, prev] and track_vis[q, prev + 1]:
                    p0, p1 = track_uv[q, prev], track_uv[q, prev + 1]
                    if np.isfinite(p0).all() and np.isfinite(p1).all():
                        cv2.line(frame, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), color, 2, cv2.LINE_AA)
            point = track_uv[q, t]
            if track_vis[q, t] and np.isfinite(point).all():
                cv2.drawMarker(frame, tuple(np.rint(point).astype(int)), color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        overlay_frames.append(frame)

    overlay_name, _ = _export_video_from_frames(
        video_rgb=np.stack(overlay_frames), fps=fps, dst_video=assets_dir / "tracks_2d_overlay.mp4"
    )
    overlay_poster = assets_dir / "video_poster.jpg"
    overlay_poster_name = "tracks_2d_overlay_poster.jpg"
    overlay_poster.replace(assets_dir / overlay_poster_name)

    valid_xyz = np.isfinite(track_xyz).all(axis=-1) & track_vis
    finite_xyz = track_xyz[valid_xyz]
    if finite_xyz.size:
        lower, upper = np.nanpercentile(finite_xyz, [2, 98], axis=0)
        center = (lower + upper) * 0.5
        half = max(float(np.max(upper - lower) * 0.6), 0.5)
    else:
        center, half = np.zeros(3, dtype=np.float32), 1.0

    track3d_frames = []
    for t in range(frame_count):
        fig = plt.figure(figsize=(8, 6), dpi=90)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("OpenD4RT predicted 3D tracks")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=24.0, azim=45.0)
        for q in range(track_xyz.shape[0]):
            keep = track_vis[q, : t + 1] & np.isfinite(track_xyz[q, : t + 1]).all(axis=-1)
            pts = track_xyz[q, : t + 1][keep]
            rgb = colors[q].astype(np.float32) / 255.0
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, linewidth=1.3)
            if pts.shape[0]:
                ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=rgb, s=18)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        track3d_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    tracks3d_name, _ = _export_video_from_frames(
        video_rgb=np.stack(track3d_frames), fps=fps, dst_video=assets_dir / "tracks_3d.mp4"
    )
    tracks3d_poster = assets_dir / "video_poster.jpg"
    tracks3d_poster_name = "tracks_3d_poster.jpg"
    tracks3d_poster.replace(assets_dir / tracks3d_poster_name)
    return {
        "tracks_2d_overlay": overlay_name,
        "tracks_2d_overlay_poster": overlay_poster_name,
        "tracks_3d": tracks3d_name,
        "tracks_3d_poster": tracks3d_poster_name,
    }


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("vis_like_demo_frames", output_dir)

    video_rgb, selected_frames = load_rgb_frames(frames_dir, args.num_frames, args.sample_mode)
    cfg = load_yaml_config(args.config)
    _disable_encoder_pretrain(cfg)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    device = _resolve_device(args.device)

    model = build_model(cfg["model"]).eval().to(device)
    payload = load_checkpoint(Path(args.ckpt_path), map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {args.ckpt_path}")
    model.load_state_dict(state_dict, strict=False)
    del state_dict, payload
    gc.collect()
    model.eval()

    image_size = cfg.get_path("model.input.image_size", [int(video_rgb.shape[1]), int(video_rgb.shape[2])])
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))
    predicted_camera_data = _predict_camera_branches(
        model=model,
        video_model_rgb=video_model_rgb,
        image_hw=(int(video_rgb.shape[1]), int(video_rgb.shape[2])),
        camera_grid_size=args.camera_grid_size,
        camera_query_chunk_size=args.camera_query_chunk_size,
        predict_intrinsics=True,
        predict_extrinsics=True,
    )
    if args.query_selection == "features":
        max_queries = max(int(args.point_max_points), int(args.track_max_points))
        feature_query_uv_px = _detect_feature_queries(
            video_rgb[0], max_points=max_queries, min_distance=args.feature_min_distance
        )
        if feature_query_uv_px.shape[0] == 0:
            raise RuntimeError("No image features detected in the first selected frame.")
        point_query_uv_px = feature_query_uv_px[: int(args.point_max_points)]
        track_query_uv_px = feature_query_uv_px[: int(args.track_max_points)]
        logger.info("Selected %d Shi-Tomasi feature queries from %s", len(feature_query_uv_px), selected_frames[0])
    else:
        point_query_uv_px = _build_uv_grid(
            width=int(video_rgb.shape[2]), height=int(video_rgb.shape[1]),
            cols=args.point_grid_cols, rows=args.point_grid_rows,
            max_points=args.point_max_points,
        )
        track_query_uv_px = _build_uv_grid(
            width=int(video_rgb.shape[2]), height=int(video_rgb.shape[1]),
            cols=max(2, int(np.sqrt(args.track_max_points))),
            rows=max(2, int(np.sqrt(args.track_max_points))),
            max_points=args.track_max_points,
        )

    package = _export_demo_data(
        model=model,
        video_rgb=video_rgb,
        video_model_rgb=video_model_rgb,
        point_query_uv_px=point_query_uv_px,
        point_query_chunk_size=args.point_query_chunk_size,
        track_query_chunk_size=args.query_chunk_size,
        track_selection="grid",
        track_max_points=int(track_query_uv_px.shape[0]),
        track_min_visible_frames=args.track_min_visible_frames,
        track_query_uv_px=track_query_uv_px,
        track_query_t_src=np.zeros((track_query_uv_px.shape[0],), dtype=np.int64),
        predicted_camera_data=predicted_camera_data,
        suppress_depth_boundary_tracks=False,
    )

    num_frames = int(package["num_frames"])
    track_xyz = np.asarray(package["track_xyz_ref0"], dtype=np.float32)
    track_vis = np.asarray(package["track_visibility"], dtype=bool)
    track_conf = np.asarray(package["track_confidence"], dtype=np.float32)
    # Point-first [P, T, ...] from _infer_tracks (see render_prediction_videos).

    meta: dict[str, Any] = {
        "fps": float(max(args.fps, 1.0)),
        "numFrames": num_frames,
        "videoWidth": int(package["video_width"]),
        "videoHeight": int(package["video_height"]),
        "clipFrames": int(package["clip_frames"]),
        "trackCount": int(track_xyz.shape[0]),
        "trackCountPred": int(track_xyz.shape[0]),
        "trackCountGt": 0,
        "pointCountPerFrame": int(package["point_query_uv_px"].shape[0]),
        "bounds": {
            "min": _jsonable_float_array(package["bounds_min"]),
            "max": _jsonable_float_array(package["bounds_max"]),
            "center": _jsonable_float_array(package["bounds_center"]),
            "radius": float(package["bounds_radius"][0]),
        },
        "ref0K": _jsonable_float_array(package["ref0_K"], ndigits=5),
        "input": {"frameDirectory": str(frames_dir), "sampledFrames": selected_frames},
    }
    if predicted_camera_data is not None:
        meta["cameraPred"] = {
            "K": _jsonable_float_array(predicted_camera_data["K"], ndigits=5),
            "TRef0Cam": _jsonable_float_array(predicted_camera_data["T_ref0_cam"], ndigits=6),
            "validIntrinsics": predicted_camera_data["valid_intrinsics"].astype(np.int32).tolist(),
            "validExtrinsics": predicted_camera_data["valid_extrinsics"].astype(np.int32).tolist(),
            "source": "d4rt_queries",
        }
    data = {
        "meta": meta,
        "tracks": {
            "queryUvPx": _jsonable_float_array(package["track_query_uv_px"], ndigits=3),
            "queryTSrc": package["track_query_t_src"].astype(np.int32).tolist(),
            "xyzRef0": _jsonable_float_array(track_xyz, ndigits=5),
            "uvPx": _jsonable_float_array(package["track_uv_px"], ndigits=3),
            "visibility": track_vis.astype(np.int32).tolist(),
            "confidence": _jsonable_float_array(track_conf, ndigits=4),
        },
        "points": {
            "queryUvPx": _jsonable_float_array(package["point_query_uv_px"], ndigits=3),
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
            "visibility": package["point_visibility"].astype(np.int32).tolist(),
            "rgb": package["point_rgb"].astype(np.int32).tolist(),
            "uvPx": _jsonable_float_array(package["point_uv_px"], ndigits=3),
            "confidence": _jsonable_float_array(package["point_confidence"], ndigits=4),
            "motionScore": _jsonable_float_array(package["point_motion_score"], ndigits=5),
            "isDynamic": package["point_is_dynamic"].astype(np.int32).tolist(),
        },
        "pointsRaw": {"xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5)},
    }
    (assets_dir / "demo_data.json").write_text(json.dumps(data), encoding="utf-8")
    depth_manifest = None
    if predicted_camera_data is not None:
        depth_manifest = export_predicted_depth_video(
            package=package,
            camera_data={"T_ref0_cam": predicted_camera_data["T_ref0_cam"]},
            output_dir=assets_dir,
            grid_rows=args.point_grid_rows,
            grid_cols=args.point_grid_cols,
        )
        if depth_manifest is not None:
            depth_poster = assets_dir / "video_poster.jpg"
            depth_poster_name = "depth_pred_poster.jpg"
            depth_poster.replace(assets_dir / depth_poster_name)
            depth_manifest["poster"] = depth_poster_name

    track_video_manifest = render_prediction_videos(video_rgb, package, assets_dir, float(max(args.fps, 1.0)))
    video_name, poster_name = _export_video_from_frames(
        video_rgb=video_rgb, fps=float(max(args.fps, 1.0)), dst_video=assets_dir / "input_video.mp4"
    )
    manifest = {
        "input_frames": str(frames_dir),
        "sampled_frame_names": selected_frames,
        "video_copy": f"assets/{video_name}",
        "video_poster": f"assets/{poster_name}",
        "data_json": "assets/demo_data.json",
        "viewer": "viser",
        "tracks_2d_overlay": f"assets/{track_video_manifest['tracks_2d_overlay']}",
        "tracks_2d_overlay_poster": f"assets/{track_video_manifest['tracks_2d_overlay_poster']}",
        "tracks_3d": f"assets/{track_video_manifest['tracks_3d']}",
        "tracks_3d_poster": f"assets/{track_video_manifest['tracks_3d_poster']}",
    }
    if depth_manifest is not None:
        manifest["depth_pred_video"] = f"assets/{depth_manifest['video']}"
        manifest["depth_pred_poster"] = f"assets/{depth_manifest['poster']}"
        manifest["depth_pred_raw"] = depth_manifest["raw"]
        meta["depthPred"] = {
            "video": f"assets/{depth_manifest['video']}",
            "poster": f"assets/{depth_manifest['poster']}",
            "raw": depth_manifest["raw"],
            "vmin": float(depth_manifest["vmin"]),
            "vmax": float(depth_manifest["vmax"]),
            "source": "d4rt_dense_point_queries",
        }
        (assets_dir / "demo_data.json").write_text(json.dumps(data), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved frame-folder demo package to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
