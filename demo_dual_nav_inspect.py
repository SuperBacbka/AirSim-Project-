
import argparse
import csv
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import airsim
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env_ppo_inspect_yolo_async import AirSimDroneInspectEnv
from env_ppo_nav import AirSimDroneEnv


NAV_START_POINTS = [[6.438, 18.830, -46.101]]
NAV_ROUTE_JSON = "train_route_ppo_nav.json"

INSPECT_START_POINTS = [[-163.421, 489.102, -72.601]]
INSPECT_YOLO_MODEL_PATH = "YOLO_isolator.pt"
INSPECT_TARGET_CLASS_IDS = {3}
DEFAULT_NAV_MODEL = "models/ppo_nav/train22_last/ppo_nav_final_continued.zip"
DEFAULT_INSPECT_MODEL = "models/ppo_inspect_async_train9.24.17_whis_yolo_seg_spark/ppo_inspect_async_final_continued.zip"
def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {name}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dual-phase NAV+INSPECT PPO demo runner")

    parser.add_argument("--nav-model", default= DEFAULT_NAV_MODEL, type=str)
    parser.add_argument("--inspect-model", default= DEFAULT_INSPECT_MODEL, type=str)
    parser.add_argument("--pairs-json", type=str, default="Lvl1.json")
    parser.add_argument("--vehicle-name", type=str, default="SimpleFlight")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-nav-steps", type=int, default=3000)
    parser.add_argument("--max-inspect-steps", type=int, default=6000)
    parser.add_argument("--nav-final-distance-threshold", type=float, default=2.0)
    parser.add_argument("--switch-route-progress-threshold", type=float, default=0.98)
    parser.add_argument("--nav-max-velocity-override", type=float, default=4.5)
    parser.add_argument("--nav-max-yaw-rate-override", type=float, default=0.0)
    parser.add_argument("--nav-step-duration-override", type=float, default=0.0)
    parser.add_argument("--nav-control-command-duration-override", type=float, default=16.0)
    parser.add_argument("--nav-planner-lookahead-override", type=int, default=2)
    parser.add_argument("--nav-planner-replan-interval-override", type=int, default=0)

    add_bool_arg(parser, "deterministic", True, "Use deterministic actions")
    add_bool_arg(parser, "record-video", True, "Write demo video")
    add_bool_arg(parser, "show-overlay", True, "Draw overlay in video")
    add_bool_arg(parser, "write-light-log", True, "Write compact CSV log")
    add_bool_arg(parser, "auto-report", True, "Auto-build technical report after run")
    add_bool_arg(parser, "save-spark-snapshots", True, "Save 2-3 spark snapshots during INSPECT phase")
    add_bool_arg(parser, "save-bird-snapshots", True, "Save 2-3 bird/safety snapshots during NAV phase")
    add_bool_arg(parser, "video-async", True, "Capture video in background thread")
    add_bool_arg(parser, "nav-disable-visual-for-test", False, "Disable NAV visual detectors for demo-only test")
    add_bool_arg(
        parser,
        "nav-disable-periodic-replan-for-test",
        False,
        "Disable NAV periodic replan by setting very large planner_replan_interval in demo",
    )

    nav_depth_group = parser.add_mutually_exclusive_group(required=False)
    nav_depth_group.add_argument(
        "--nav-use-depth-image",
        dest="nav_use_depth_image",
        action="store_true",
        help="Force NAV depth image usage in demo",
    )
    nav_depth_group.add_argument(
        "--no-nav-use-depth-image",
        dest="nav_use_depth_image",
        action="store_false",
        help="Disable NAV depth image usage in demo",
    )
    parser.set_defaults(nav_use_depth_image=None)

    parser.add_argument("--video-path", type=str, default="demo_dual_nav_inspect_test6.mp4")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-capture-every-n-steps", type=int, default=1)
    parser.add_argument("--video-camera-name", type=str, default="2")
    parser.add_argument("--video-width", type=int, default=1200)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--light-log-path", type=str, default="demo_dual_metrics_test6.csv")
    parser.add_argument("--report-dir", type=str, default="reports/demo_runs")
    parser.add_argument("--report-name", type=str, default="")
    parser.add_argument("--report-plots-dpi", type=int, default=200)
    parser.add_argument("--spark-snapshot-max-count", type=int, default=3)
    parser.add_argument("--spark-snapshot-min-gap-steps", type=int, default=10)
    parser.add_argument("--spark-snapshot-dirname", type=str, default="spark_snapshots")
    parser.add_argument("--bird-snapshot-max-count", type=int, default=3)
    parser.add_argument("--bird-snapshot-min-gap-steps", type=int, default=10)
    parser.add_argument("--bird-snapshot-dirname", type=str, default="bird_snapshots")

    no_heavy_group = parser.add_mutually_exclusive_group(required=False)
    no_heavy_group.add_argument("--no-heavy-info", dest="no_heavy_info", action="store_true")
    no_heavy_group.add_argument("--heavy-info", dest="no_heavy_info", action="store_false")
    parser.set_defaults(no_heavy_info=True)

    return parser.parse_args()


def safe_info(info: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if not isinstance(info, dict):
        return default
    value = info.get(key, default)
    if value is None:
        return default
    return value


def safe_info_deep(info: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if not isinstance(info, dict):
        return default

    if key in info and info[key] is not None:
        return info[key]

    for nested_key in ("reward_info", "planner_info", "control_debug"):
        nested = info.get(nested_key, None)
        if isinstance(nested, dict) and key in nested and nested[key] is not None:
            return nested[key]

    return default


def get_info_path(info: Optional[Dict[str, Any]], path: str, default: Any = np.nan) -> Any:
    """
    Читает вложенные значения из dict по dot-path.
    Пример:
      get_info_path(info, "planner_info.visual_debug.visual_hazard_active", 0.0)
    """
    if not isinstance(info, dict) or not path:
        return default
    cur: Any = info
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur.get(part, default)
    return cur if cur is not None else default


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def unwrap_env(env: Any) -> Any:
    base = env
    for _ in range(12):
        if hasattr(base, "env"):
            base = base.env
        else:
            break
    try:
        return base.unwrapped
    except Exception:
        return base


def load_route_from_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data, dtype=np.float32)


def load_pairs_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs: List[Dict[str, Any]] = []
    for item in data:
        pairs.append(
            {
                "route_point": np.asarray(item["route_point"], dtype=np.float32),
                "target_point": np.asarray(item["target_point"], dtype=np.float32),
                "point_mode": item.get("point_mode", "inspect"),
                "visual_gate_required": bool(item.get("visual_gate_required", False)),
                "visual_gate_visible_threshold": float(item.get("visual_gate_visible_threshold", 0.20)),
                "visual_gate_center_threshold": float(item.get("visual_gate_center_threshold", 0.32)),
                "visual_gate_yaw_threshold": float(item.get("visual_gate_yaw_threshold", 0.65)),
                "visual_gate_hold_steps": int(item.get("visual_gate_hold_steps", 2)),
            }
        )
    return pairs


def infer_world_xy_limit(
    start_points: Optional[List[List[float]]] = None,
    inspection_pairs: Optional[List[Dict[str, Any]]] = None,
    extra_margin: float = 20.0,
    min_limit: float = 80.0,
) -> float:
    pts: List[np.ndarray] = []

    if start_points is not None:
        for p in start_points:
            arr = np.asarray(p, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                pts.append(arr[:2])

    if inspection_pairs is not None:
        for pair in inspection_pairs:
            rp = np.asarray(pair["route_point"], dtype=np.float32).reshape(-1)
            tp = np.asarray(pair["target_point"], dtype=np.float32).reshape(-1)
            if rp.shape[0] >= 2:
                pts.append(rp[:2])
            if tp.shape[0] >= 2:
                pts.append(tp[:2])

    if not pts:
        return float(min_limit)

    pts_arr = np.asarray(pts, dtype=np.float32)
    max_abs_xy = float(np.max(np.abs(pts_arr)))
    return float(max(max_abs_xy + extra_margin, min_limit))

def seg_spark_freshness_provider(env, image, selected):
    if not getattr(env, "enable_visual_inspection", False):
        return np.zeros(16, dtype=np.float32)

    seg = getattr(env, "_last_seg_info", {}) or {}

    if getattr(env, "async_visual_update", False):
        ts = float(getattr(env, "_async_latest_timestamp", 0.0))
        if ts > 0.0:
            age_sec = max(0.0, time.time() - ts)
        else:
            age_sec = 1.0
    else:
        last_step = int(getattr(env, "_last_visual_infer_step", -10**9))
        if last_step < 0:
            age_sec = 1.0
        else:
            age_sec = max(0.0, (env.current_step - last_step) * float(env.step_duration))

    stale_thr = max(float(env.step_duration * env.yolo_inference_interval * 1.5), 0.12)
    age_norm = float(np.clip(age_sec / 0.50, 0.0, 1.0))
    stale_flag = 1.0 if age_sec > stale_thr else 0.0

    seg_visible = float(seg.get("seg_visible", 0.0))
    seg_cable_ratio = float(seg.get("seg_cable_ratio", 0.0))
    seg_tower_ratio = float(seg.get("seg_tower_ratio", 0.0))
    seg_cable_center_dx = float(seg.get("seg_cable_center_dx", 0.0))
    seg_cable_center_dy = float(seg.get("seg_cable_center_dy", 0.0))
    seg_cable_angle_sin = float(seg.get("seg_cable_angle_sin", 0.0))
    seg_cable_angle_cos = float(seg.get("seg_cable_angle_cos", 1.0))
    seg_mask_stability = float(seg.get("seg_mask_stability", 0.0))

    spark = getattr(env, "_last_spark_info", {}) or {}

    spark_soft_gain = 0.15
    spark_conf = float(np.clip(spark.get("spark_max_conf", 0.0), 0.0, 1.0))
    spark_persistent_raw = float(np.clip(spark.get("spark_persistent", 0.0), 0.0, 1.0))
    spark_gate = 1.0 if (spark_conf >= 0.45 or spark_persistent_raw > 0.5) else 0.0

    spark_visible = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_visible", 0.0), 0.0, 1.0))
    spark_max_conf = spark_soft_gain * spark_gate * spark_conf
    spark_max_area = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_max_area", 0.0), 0.0, 1.0))
    spark_center_dx = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_center_dx", 0.0), -1.0, 1.0))
    spark_center_dy = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_center_dy", 0.0), -1.0, 1.0))
    spark_persistent = spark_soft_gain * spark_persistent_raw

    return np.array(
        [
            seg_visible,
            seg_cable_ratio,
            seg_tower_ratio,
            seg_cable_center_dx,
            seg_cable_center_dy,
            seg_cable_angle_sin,
            seg_cable_angle_cos,
            seg_mask_stability,
            spark_visible,
            spark_max_conf,
            spark_max_area,
            spark_center_dx,
            spark_center_dy,
            spark_persistent,
            age_norm,
            stale_flag,
        ],
        dtype=np.float32,
    )


def make_nav_env(args: argparse.Namespace) -> Monitor:
    base_step_duration = 0.08
    base_control_command_duration = 0.65
    base_max_velocity = 6.0
    base_max_yaw_rate = 45.0
    base_planner_replan_interval = 12
    base_planner_lookahead_steps = 1
    base_use_depth_image = True

    step_duration = (
        float(args.nav_step_duration_override)
        if float(args.nav_step_duration_override) > 0.0
        else float(base_step_duration)
    )
    control_command_duration = (
        float(args.nav_control_command_duration_override)
        if float(args.nav_control_command_duration_override) > 0.0
        else float(base_control_command_duration)
    )
    max_velocity = (
        float(args.nav_max_velocity_override)
        if float(args.nav_max_velocity_override) > 0.0
        else float(base_max_velocity)
    )
    max_yaw_rate = (
        float(args.nav_max_yaw_rate_override)
        if float(args.nav_max_yaw_rate_override) > 0.0
        else float(base_max_yaw_rate)
    )
    planner_lookahead_steps = (
        int(args.nav_planner_lookahead_override)
        if int(args.nav_planner_lookahead_override) > 0
        else int(base_planner_lookahead_steps)
    )
    planner_replan_interval = (
        int(args.nav_planner_replan_interval_override)
        if int(args.nav_planner_replan_interval_override) > 0
        else int(base_planner_replan_interval)
    )
    if bool(getattr(args, "nav_disable_periodic_replan_for_test", False)):
        planner_replan_interval = 10 ** 9
    use_depth_image = (
        bool(args.nav_use_depth_image)
        if args.nav_use_depth_image is not None
        else bool(base_use_depth_image)
    )
    nav_disable_visual = bool(args.nav_disable_visual_for_test)
    enable_nav_visual = not nav_disable_visual
    enable_nav_seg_context = not nav_disable_visual
    enable_nav_safety_detector = not nav_disable_visual

    route_sets = load_route_from_json(NAV_ROUTE_JSON)
    env = AirSimDroneEnv(
        vehicle_name=args.vehicle_name,
        start_points=NAV_START_POINTS,
        route_sets=route_sets,
        route_points=None,
        fixed_goal=None,
        random_goal=False,
        max_episode_steps=2200,
        step_duration=step_duration,
        max_velocity=max_velocity,
        max_yaw_rate=max_yaw_rate,
        max_sensor_distance=5.0,
        depth_max_distance=12.0,
        goal_tolerance=1.5,
        waypoint_tolerance=1.2,
        min_altitude=1.0,
        max_altitude=100.0,
        world_xy_limit=520.0,
        min_start_goal_distance=2.0,
        use_route_planner=True,
        planner_grid_resolution=1.5,
        planner_vertical_resolution=1.0,
        planner_grid_half_size=10,
        planner_vertical_half_size=3,
        planner_replan_interval=planner_replan_interval,
        planner_lookahead_steps=planner_lookahead_steps,
        planner_shift_distance=2.5,
        planner_shift_hold_steps=14,
        soft_avoidance_fail_steps=8,
        obstacle_memory_ttl_steps=40,
        use_depth_image=use_depth_image,
        depth_camera_name="0",
        nav_visual_input_size=320,
        depth_image_interval=3,
        reserve_yolo_features=True,
        yolo_feature_dim=16,
        verbose=False,
        min_xy_cmd_ratio=0.03,
        max_vertical_speed_ratio=0.2,
        visual_replan_shift_distance=2.5,
        visual_replan_hold_steps=15,
        visual_hazard_threshold=0.25,
        visual_hazard_clear_steps=8,
        visual_hazard_xy_scale=0.7,
        enable_nav_visual=enable_nav_visual,
        enable_nav_seg_context=enable_nav_seg_context,
        enable_nav_safety_detector=enable_nav_safety_detector,
        visual_hazard_replan_enabled=False,
        debug_print_visual_hazard=False,
        safety_conf_threshold=0.30,
        safety_detector_model_path="Bird.pt",
        safety_target_class_names={"bird"},
        infra_detector_model_path=None,
        nav_visual_obs_mode="safety_only",
        action_smoothing_alpha=0.45,
        nav_use_deep_sort=True,
        cmd_filter_alpha_xy=0.72,
        cmd_filter_alpha_z=0.5,
        cmd_filter_alpha_yaw=0.55,
        max_delta_vxy_cmd=0.45,
        max_delta_vz_cmd=0.18,
        max_delta_yaw_rate_cmd=3.0,
        altitude_assist_mix=0.40,
        inactive_penalty_weight=0.0,
        paired_start_route=True,
        enable_goal_progress_reward=True,
        nav_use_body_frame_control=False,
        nav_fixed_yaw_enabled=False,
        nav_fixed_yaw_deadband_xy_m=1.2,
        nav_fixed_yaw_max_step_deg=10.0,
        nav_apply_controller_gains=False,
        nav_velocity_kp_xy=0.16,
        nav_velocity_kd_xy=0.02,
        nav_velocity_kp_z=2.0,
        nav_velocity_ki_z=0.0,
        nav_velocity_kd_z=0.10,
        nav_angle_level_kp_rp=2.2,
        nav_angle_level_kd_rp=0.20,
        nav_angle_level_kp_yaw=1.2,
        nav_angle_level_kd_yaw=0.10,
        altitude_kp=0.9,
        altitude_kd=0.45,
        altitude_action_z_offset=0.20,
        nav_angle_rate_kp_rp=0.20,
        nav_angle_rate_kd_rp=0.02,
        nav_angle_rate_kp_yaw=0.12,
        nav_angle_rate_kd_yaw=0.01,
        action_repeat=1,
        control_command_duration=control_command_duration,
    )
    return Monitor(env)

def make_inspect_env(args: argparse.Namespace) -> Monitor:
    inspection_pairs = load_pairs_from_json(args.pairs_json)
    world_xy_limit_auto = infer_world_xy_limit(
        start_points=INSPECT_START_POINTS,
        inspection_pairs=inspection_pairs,
        extra_margin=20.0,
        min_limit=80.0,
    )

    env = AirSimDroneInspectEnv(
        vehicle_name=args.vehicle_name,
        start_points=INSPECT_START_POINTS,
        inspection_pairs=inspection_pairs,
        inspection_pair_sets=None,
        random_goal=False,
        max_episode_steps=22000,
        step_duration=0.06,
        cable_soft_ratio=0.0025,
        cable_hard_ratio=0.0080,
        cable_center_soft=0.34,
        cable_center_hard=0.20,
        cable_away_push_speed=0.0,
        cable_soft_xy_scale=0.35,
        cable_hard_xy_scale=0.05,
        cable_yaw_scale=0.35,
        cable_safety_hold_steps=12,
        action_repeat=1,
        min_altitude=1.0,
        max_altitude=60.0,
        control_command_duration=0.30,
        max_velocity=5.0,
        max_yaw_rate=35.0,
        desired_standoff_distance=3.0,
        standoff_tolerance=1.2,
        reserve_yolo_features=True,
        yolo_feature_dim=8,
        aux_visual_feature_dim=16,
        enable_visual_inspection=True,
        enable_nav_visual=False,
        enable_nav_safety_detector=False,
        face_inspection_target=True,
        async_visual_update=True,
        async_camera_poll_interval=0.25,
        async_inference_min_interval=0.75,
        keep_last_async_frame=True,
        scene_camera_name="0",
        yolo_model_path=INSPECT_YOLO_MODEL_PATH,
        yolo_device="cuda",
        yolo_input_size=256,
        yolo_conf_threshold=0.25,
        yolo_iou_threshold=0.5,
        yolo_target_class_ids=INSPECT_TARGET_CLASS_IDS,
        yolo_target_class_names=None,
        yolo_inference_interval=8,
        target_ema_alpha=0.35,
        target_hold_steps=20,
        seg_model_path="Yolo_segemnt.pt",
        seg_inference_interval=10,
        spark_inference_interval=4,
        spark_model_path="spark.pt",
        spark_camera_name="2",
        seg_conf_threshold=0.25,
        spark_conf_threshold=0.25,
        spark_iou_threshold=0.1,
        seg_input_size=192,
        spark_input_size=640,
        debug_draw_detections=False,
        debug_show_window=False,
        debug_save_video=False,
        debug_print_yolo=False,
        debug_print_obs=False,
        debug_print_seg=False,
        debug_video_path=r"logs/inspect_debug/vision_dron.mp4",
        debug_draw_every_n_steps=1,
        enable_heavy_diagnostics=not bool(args.no_heavy_info),
        verbose=False,
        inspect_xy_speed_limit=1.5,
        inspect_xy_delta_limit=0.35,
        inspect_xy_filter_alpha=0.35,
        inspect_manual_yaw_mix=0.2,
        yolo_reward_scale=0.05,
        hide_final_goal_in_obs=True,
        debug_print_reset=True,
        ema_like_pretrain_mode=False,
        max_vertical_speed_ratio=0.35,
        run_external_debug_detectors=True,
        spark_response_enabled=True,
        spark_confirm_conf_thr=0.32,
        spark_confirm_event_thr=0.1,
        spark_confirm_steps_req=2,
        spark_clear_steps_req=50,
        spark_override_duration_steps=100,
        spark_response_backoff_ratio=0.65,
        spark_response_lateral_ratio=0.45,
        spark_hazard_xy_scale=0.72,
        spark_hazard_yaw_scale=0.80,
        no_progress_max_steps=120,
        world_xy_limit=world_xy_limit_auto,
        world_xy_oob_margin=1.0,
        use_depth_image=False,
        inspect_reset_lift_m=0.0,
        nav_use_body_frame_control=False,
        nav_fixed_yaw_enabled=False,
        nav_apply_controller_gains=False,
        altitude_kp=0.75,
        altitude_kd=0.45,
        altitude_action_z_offset=0.20,
        altitude_assist_mix=0.40,
        use_deep_sort=True,
        cmd_filter_alpha_z=0.50,
        max_delta_vz_cmd=0.25,
        require_target_visibility=False,
        control_stream_enabled=True,
        control_stream_hz=25.0,
        control_stream_stale_sec=2.5,
        control_stream_command_duration=0.25,
        cable_desired_standoff_m=6.0,
        cable_soft_distance_m=4.0,
        cable_hard_distance_m=1.0,
        cable_safety_enabled=True,
        waypoint_tolerance=0.4,
        waypoint_pass_tolerance=0.65,
        waypoint_pass_margin=0.08,
        waypoint_pass_increase_steps_req=2,
    )
    env.register_aux_feature_provider(seg_spark_freshness_provider)
    return Monitor(env)


def check_model_env_compat(model: PPO, env: Monitor, name: str) -> None:
    model_shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    env_shape = tuple(getattr(env.observation_space, "shape", ()) or ())
    if model_shape != env_shape:
        raise RuntimeError(
            f"[{name}] observation shape mismatch: model={model_shape}, env={env_shape}. "
            f"Model and environment must be native pair."
        )


def reset_env(env: Monitor) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def step_env(env: Monitor, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info
    raise RuntimeError(f"Unexpected step() return format: {type(out)}")


def _get_route_len(env: Optional[Monitor]) -> Optional[int]:
    if env is None:
        return None
    base = unwrap_env(env)
    for attr in ("current_route_points", "mission_route"):
        points = getattr(base, attr, None)
        if points is not None:
            try:
                n = int(len(points))
            except Exception:
                n = 0
            if n > 0:
                return n
    return None


def _extract_route_progress_proxy(info: Dict[str, Any], env: Optional[Monitor] = None) -> float:
    wp_idx = extract_wp_idx(info)
    if wp_idx < 0:
        return np.nan

    route_len = _get_route_len(env)
    if route_len is None or route_len <= 0:
        return np.nan

    denom = max(1, int(route_len) - 1)
    return float(np.clip(float(wp_idx) / float(denom), 0.0, 1.0))


def extract_route_progress(info: Dict[str, Any], env: Optional[Monitor] = None) -> float:
    route_progress = as_float(safe_info_deep(info, "route_progress_norm", np.nan), np.nan)
    if not np.isnan(route_progress):
        return float(route_progress)
    return _extract_route_progress_proxy(info, env=env)


def extract_wp_idx(info: Dict[str, Any]) -> int:
    for key in [
        "current_waypoint_idx",
        "current_waypoint_idx_dbg",
        "debug_current_waypoint_idx",
        "waypoint_idx",
        "wp_idx_dbg",
    ]:
        val = safe_info_deep(info, key, None)
        if val is not None:
            try:
                return int(float(val))
            except Exception:
                pass
    return -1


def extract_collision_penalty(info: Dict[str, Any]) -> float:
    return as_float(safe_info_deep(info, "collision_penalty", 0.0), 0.0)


def extract_point_mode(info: Dict[str, Any]) -> str:
    is_inspect = as_float(safe_info_deep(info, "point_mode_is_inspect_dbg", 0.0), 0.0)
    is_transit = as_float(safe_info_deep(info, "point_mode_is_transit_dbg", 0.0), 0.0)
    if is_inspect > 0.5:
        return "inspect"
    if is_transit > 0.5:
        return "transit"
    return ""


def _first_finite_from_info(info: Dict[str, Any], keys: List[str], default: float = np.nan) -> float:
    for key in keys:
        value = as_float(safe_info_deep(info, key, np.nan), np.nan)
        if not np.isnan(value):
            return float(value)
    return float(default)


def extract_bird_light_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    eps = 1e-9

    def _prefer_direct_or_fallback(
        direct_keys: List[str],
        fallback_value: float,
        default: float = np.nan,
    ) -> float:
        direct_val = _first_finite_from_info(info, direct_keys, default=np.nan)
        # Приоритет прямых bird_* полей. Если прямое поле отсутствует/NaN/0 -> fallback.
        if not np.isnan(direct_val) and abs(float(direct_val)) > eps:
            return float(direct_val)
        if not np.isnan(fallback_value):
            return float(fallback_value)
        if not np.isnan(direct_val):
            return float(direct_val)
        return float(default)

    vh_active_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.visual_hazard_active", np.nan),
        np.nan,
    )
    vh_score_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.visual_hazard_score", np.nan),
        np.nan,
    )
    vh_confirm_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.visual_hazard_confirm_steps", np.nan),
        np.nan,
    )
    vh_clear_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.visual_hazard_clear_counter", np.nan),
        np.nan,
    )
    safety_visible_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.safety_visible", np.nan),
        np.nan,
    )
    safety_conf_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.safety_conf", np.nan),
        np.nan,
    )
    safety_area_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.safety_area", np.nan),
        np.nan,
    )
    safety_dx_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.safety_dx", np.nan),
        np.nan,
    )
    safety_hazard_nested = as_float(
        get_info_path(info, "planner_info.visual_debug.safety_hazard_score", np.nan),
        np.nan,
    )

    visual_hazard_active_raw = _prefer_direct_or_fallback(
        [
            "visual_hazard_active_dbg",
            "visual_hazard_active",
            "nav_visual_hazard_active",
            "planner_visual_debug_visual_hazard_active",
        ],
        fallback_value=vh_active_nested,
        default=0.0,
    )
    visual_hazard_score_raw = _prefer_direct_or_fallback(
        [
            "visual_hazard_score_dbg",
            "visual_hazard_score",
            "planner_visual_debug_visual_hazard_score",
        ],
        fallback_value=vh_score_nested,
        default=0.0,
    )
    safety_hazard_score_raw = _prefer_direct_or_fallback(
        [
            "safety_hazard_score_dbg",
            "safety_hazard_score",
            "planner_visual_debug_safety_hazard_score",
        ],
        fallback_value=safety_hazard_nested,
        default=0.0,
    )
    visual_hazard_confirm_steps = _prefer_direct_or_fallback(
        ["visual_hazard_confirm_steps_dbg"],
        fallback_value=vh_confirm_nested,
        default=0.0,
    )
    visual_hazard_clear_counter = _prefer_direct_or_fallback(
        ["visual_hazard_clear_counter_dbg"],
        fallback_value=vh_clear_nested,
        default=0.0,
    )
    safety_visible = _prefer_direct_or_fallback(
        [
            "safety_visible_dbg",
            "nav_safety_visible",
            "safety_visible",
            "planner_visual_debug_safety_visible",
        ],
        fallback_value=safety_visible_nested,
        default=0.0,
    )
    safety_conf = _prefer_direct_or_fallback(
        [
            "safety_conf_dbg",
            "nav_safety_conf",
            "safety_conf",
            "planner_visual_debug_selected_conf",
        ],
        fallback_value=safety_conf_nested,
        default=np.nan,
    )
    safety_area = _prefer_direct_or_fallback(
        [
            "safety_area_dbg",
            "planner_visual_debug_selected_area",
        ],
        fallback_value=safety_area_nested,
        default=np.nan,
    )
    safety_dx = _prefer_direct_or_fallback(
        [
            "safety_dx_dbg",
            "planner_visual_debug_selected_center_dx",
        ],
        fallback_value=safety_dx_nested,
        default=np.nan,
    )

    visual_hazard_active_dbg = 1.0 if float(visual_hazard_active_raw) > 0.5 else 0.0

    bird_visible_direct = _first_finite_from_info(
        info,
        ["bird_visible_dbg", "bird_visible"],
        default=np.nan,
    )
    heuristic_visible = any(
        [
            visual_hazard_active_dbg > 0.5,
            float(safety_visible) > 0.5,
            float(visual_hazard_score_raw) > 0.0,
            float(safety_hazard_score_raw) > 0.0,
        ]
    )
    if np.isnan(bird_visible_direct) or float(bird_visible_direct) <= 0.0:
        bird_visible = 1.0 if heuristic_visible else 0.0
    else:
        bird_visible = 1.0 if float(bird_visible_direct) > 0.5 else 0.0

    bird_hazard_direct = _first_finite_from_info(info, ["bird_hazard_active_dbg"], default=np.nan)
    if np.isnan(bird_hazard_direct) or float(bird_hazard_direct) <= 0.0:
        bird_hazard_active = 1.0 if visual_hazard_active_dbg > 0.5 else 0.0
    else:
        bird_hazard_active = 1.0 if float(bird_hazard_direct) > 0.5 else 0.0

    bird_response_direct = _first_finite_from_info(info, ["bird_response_active_dbg"], default=np.nan)
    if np.isnan(bird_response_direct) or float(bird_response_direct) <= 0.0:
        bird_response_active = float(bird_hazard_active)
    else:
        bird_response_active = 1.0 if float(bird_response_direct) > 0.5 else 0.0

    bird_det_count_direct = _first_finite_from_info(
        info,
        ["bird_det_count_dbg", "bird_det_count", "nav_safety_det_count", "safety_det_count"],
        default=np.nan,
    )
    if np.isnan(bird_det_count_direct) or abs(float(bird_det_count_direct)) <= eps:
        bird_det_count = 1.0 if float(safety_visible) > 0.5 else 0.0
    else:
        bird_det_count = float(bird_det_count_direct)

    bird_max_conf = _prefer_direct_or_fallback(
        ["bird_max_conf_dbg", "bird_max_conf", "selected_conf"],
        fallback_value=float(safety_conf),
        default=np.nan,
    )
    bird_area = _prefer_direct_or_fallback(
        ["bird_area_dbg", "bird_area", "selected_area"],
        fallback_value=float(safety_area),
        default=np.nan,
    )
    bird_center_dx = _prefer_direct_or_fallback(
        ["bird_center_dx_dbg", "bird_center_dx", "selected_center_dx"],
        fallback_value=float(safety_dx),
        default=np.nan,
    )
    bird_center_dy_direct = _first_finite_from_info(
        info,
        ["bird_center_dy_dbg", "bird_center_dy", "selected_center_dy"],
        default=np.nan,
    )
    bird_center_dy = float(bird_center_dy_direct) if not np.isnan(bird_center_dy_direct) else np.nan

    return {
        "bird_visible_dbg": float(bird_visible),
        "bird_det_count_dbg": float(bird_det_count),
        "bird_max_conf_dbg": float(bird_max_conf) if not np.isnan(bird_max_conf) else np.nan,
        "bird_center_dx_dbg": float(bird_center_dx) if not np.isnan(bird_center_dx) else np.nan,
        "bird_center_dy_dbg": float(bird_center_dy) if not np.isnan(bird_center_dy) else np.nan,
        "bird_area_dbg": float(bird_area) if not np.isnan(bird_area) else np.nan,
        "bird_hazard_active_dbg": float(bird_hazard_active),
        "bird_response_active_dbg": float(bird_response_active),
        "visual_hazard_active_dbg": float(visual_hazard_active_dbg),
        "visual_hazard_score_dbg": float(visual_hazard_score_raw),
        "visual_hazard_confirm_steps_dbg": float(visual_hazard_confirm_steps),
        "visual_hazard_clear_counter_dbg": float(visual_hazard_clear_counter),
        "safety_visible_dbg": float(safety_visible),
        "safety_conf_dbg": float(safety_conf) if not np.isnan(safety_conf) else np.nan,
        "safety_area_dbg": float(safety_area) if not np.isnan(safety_area) else np.nan,
        "safety_dx_dbg": float(safety_dx) if not np.isnan(safety_dx) else np.nan,
        "safety_hazard_score_dbg": float(safety_hazard_score_raw),
    }


def extract_bird_bbox(info: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x1 = _first_finite_from_info(
        info,
        ["bird_bbox_x1_dbg", "bird_bbox_x1", "selected_x1", "planner_visual_debug_selected_x1"],
        default=0.0,
    )
    y1 = _first_finite_from_info(
        info,
        ["bird_bbox_y1_dbg", "bird_bbox_y1", "selected_y1", "planner_visual_debug_selected_y1"],
        default=0.0,
    )
    x2 = _first_finite_from_info(
        info,
        ["bird_bbox_x2_dbg", "bird_bbox_x2", "selected_x2", "planner_visual_debug_selected_x2"],
        default=0.0,
    )
    y2 = _first_finite_from_info(
        info,
        ["bird_bbox_y2_dbg", "bird_bbox_y2", "selected_y2", "planner_visual_debug_selected_y2"],
        default=0.0,
    )
    return float(x1), float(y1), float(x2), float(y2)


def print_nav_cfg(env: Monitor, args: argparse.Namespace) -> None:
    base_env = unwrap_env(env)
    print(f"[NAV_CFG] step_duration={float(getattr(base_env, 'step_duration', np.nan))}")
    print(f"[NAV_CFG] control_command_duration={float(getattr(base_env, 'control_command_duration', np.nan))}")
    print(f"[NAV_CFG] planner_replan_interval={int(getattr(base_env, 'planner_replan_interval', -1))}")
    print(f"[NAV_CFG] max_velocity={float(getattr(base_env, 'max_velocity', np.nan))}")
    print(f"[NAV_CFG] max_yaw_rate={float(getattr(base_env, 'max_yaw_rate', np.nan))}")
    print(f"[NAV_CFG] planner_lookahead_steps={int(getattr(base_env, 'planner_lookahead_steps', -1))}")
    print(f"[NAV_CFG] use_depth_image={bool(getattr(base_env, 'use_depth_image', False))}")
    print(f"[NAV_CFG] enable_nav_visual={bool(getattr(base_env, 'enable_nav_visual', False))}")
    print(
        f"[NAV_CFG] nav_disable_periodic_replan_for_test="
        f"{bool(getattr(args, 'nav_disable_periodic_replan_for_test', False))}"
    )


def _timing_mean_max(values: List[float]) -> Tuple[float, float]:
    if not values:
        return np.nan, np.nan
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanmax(arr))


def safe_col(df, col: str, default):
    import pandas as pd

    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def to_num(series, default=np.nan):
    s = np.asarray(series, dtype=object)
    out = []
    for v in s:
        if v is None:
            out.append(default)
            continue
        txt = str(v).strip()
        if txt == "" or txt.lower() == "nan":
            out.append(default)
            continue
        try:
            out.append(float(txt))
        except Exception:
            out.append(default)
    import pandas as pd

    return pd.Series(out, index=getattr(series, "index", None), dtype="float64")


def save_plot(fig, path: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def find_events(df, flag_col: str, phase_filter: Optional[str] = None):
    import pandas as pd

    cols = [
        "event_id",
        "flag_col",
        "phase_filter",
        "start_index",
        "end_index",
        "start_global_step",
        "end_global_step",
        "start_step",
        "end_step",
        "duration_rows",
        "start_wp_idx",
        "end_wp_idx",
        "max_flag",
        "mean_flag",
    ]
    if flag_col not in df.columns:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    if phase_filter is not None and "phase" in work.columns:
        phase_u = str(phase_filter).upper()
        work = work[work["phase"].astype(str).str.upper() == phase_u].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    flags = to_num(safe_col(work, flag_col, 0.0), default=0.0).fillna(0.0).to_numpy()
    gstep = to_num(safe_col(work, "global_step", np.nan), default=np.nan).to_numpy()
    step = to_num(safe_col(work, "step", np.nan), default=np.nan).to_numpy()
    wp = to_num(safe_col(work, "wp_idx", np.nan), default=np.nan).to_numpy()
    idx_arr = work.index.to_numpy()

    events: List[Dict[str, Any]] = []
    in_evt = False
    start_i = 0
    evt_id = 0

    def _append_event(si: int, ei: int) -> None:
        nonlocal evt_id
        evt_id += 1
        seg = flags[si : ei + 1]
        events.append(
            {
                "event_id": evt_id,
                "flag_col": flag_col,
                "phase_filter": phase_filter or "",
                "start_index": int(idx_arr[si]),
                "end_index": int(idx_arr[ei]),
                "start_global_step": float(gstep[si]) if not np.isnan(gstep[si]) else np.nan,
                "end_global_step": float(gstep[ei]) if not np.isnan(gstep[ei]) else np.nan,
                "start_step": float(step[si]) if not np.isnan(step[si]) else np.nan,
                "end_step": float(step[ei]) if not np.isnan(step[ei]) else np.nan,
                "duration_rows": int(ei - si + 1),
                "start_wp_idx": float(wp[si]) if not np.isnan(wp[si]) else np.nan,
                "end_wp_idx": float(wp[ei]) if not np.isnan(wp[ei]) else np.nan,
                "max_flag": float(np.nanmax(seg)) if seg.size > 0 else np.nan,
                "mean_flag": float(np.nanmean(seg)) if seg.size > 0 else np.nan,
            }
        )

    for i in range(len(flags)):
        active = bool(flags[i] > 0.5)
        if active and not in_evt:
            in_evt = True
            start_i = i
        elif (not active) and in_evt:
            _append_event(start_i, i - 1)
            in_evt = False

    if in_evt:
        _append_event(start_i, len(flags) - 1)

    return pd.DataFrame(events, columns=cols)


def build_demo_report(csv_path: str, out_dir: str, dpi: int = 200) -> Optional[str]:
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[REPORT][WARN] report dependencies unavailable: {e}")
        return None

    csv_abs = os.path.abspath(csv_path)
    if (not os.path.exists(csv_abs)) or os.path.getsize(csv_abs) <= 0:
        print(f"[REPORT][WARN] light-log not found or empty: {csv_abs}")
        return None

    try:
        df = pd.read_csv(csv_abs)
    except Exception as e:
        print(f"[REPORT][WARN] failed to read light-log: {csv_abs} ({e})")
        return None

    if df.empty:
        print(f"[REPORT][WARN] light-log not found or empty: {csv_abs}")
        return None

    out_abs = os.path.abspath(out_dir)
    os.makedirs(out_abs, exist_ok=True)

    if "global_step" not in df.columns:
        df["global_step"] = np.arange(1, len(df) + 1, dtype=np.float64)
    else:
        df["global_step"] = to_num(df["global_step"], default=np.nan)
        miss = df["global_step"].isna()
        if miss.any():
            df.loc[miss, "global_step"] = np.arange(1, len(df) + 1, dtype=np.float64)[miss.to_numpy()]

    if "phase" not in df.columns:
        df["phase"] = ""
    df["phase"] = df["phase"].astype(str).str.upper()

    numeric_cols = [
        "step",
        "wp_idx",
        "reward",
        "episode_reward",
        "local_distance",
        "final_distance",
        "route_progress_norm",
        "loop_dt_ms",
        "env_step_dt_ms",
        "model_predict_dt_ms",
        "video_dt_ms",
        "planner_replan_step_dbg",
        "local_path_rebuilt_this_step_dbg",
        "local_target_changed_this_step_dbg",
        "visible",
        "conf",
        "center_error",
        "cable_safety_active",
        "bird_visible_dbg",
        "bird_det_count_dbg",
        "bird_max_conf_dbg",
        "bird_center_dx_dbg",
        "bird_center_dy_dbg",
        "bird_area_dbg",
        "bird_hazard_active_dbg",
        "bird_response_active_dbg",
        "visual_hazard_active_dbg",
        "visual_hazard_score_dbg",
        "visual_hazard_confirm_steps_dbg",
        "visual_hazard_clear_counter_dbg",
        "safety_visible_dbg",
        "safety_conf_dbg",
        "safety_area_dbg",
        "safety_dx_dbg",
        "safety_hazard_score_dbg",
        "gate_blocked_dbg",
        "visual_gate_required_dbg",
        "spark_visible_dbg",
        "spark_det_count_dbg",
        "spark_max_conf_dbg",
        "spark_event_score_dbg",
        "spark_response_active_dbg",
        "spark_override_steps_left_dbg",
        "spark_action_override_changed_dbg",
        "spark_camera_id_dbg",
        "spark_iou_threshold_dbg",
        "spark_infer_count_dbg",
        "spark_infer_interval_steps_dbg",
        "spark_infer_time_ms_dbg",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_num(df[c], default=np.nan)

    saved_tables: List[str] = []
    saved_plots: List[str] = []
    skipped_plots: List[Dict[str, str]] = []

    def _series_stat_last(sub_df, col: str) -> float:
        if col not in sub_df.columns:
            return np.nan
        s = to_num(sub_df[col], default=np.nan).dropna()
        if s.empty:
            return np.nan
        return float(s.iloc[-1])

    def _series_stat_max(sub_df, col: str) -> float:
        if col not in sub_df.columns:
            return np.nan
        s = to_num(sub_df[col], default=np.nan).dropna()
        if s.empty:
            return np.nan
        return float(s.max())

    def _series_stat_min(sub_df, col: str) -> float:
        if col not in sub_df.columns:
            return np.nan
        s = to_num(sub_df[col], default=np.nan).dropna()
        if s.empty:
            return np.nan
        return float(s.min())

    # summary_by_phase.csv
    phase_rows: List[Dict[str, Any]] = []
    for phase_name, sub in df.groupby("phase", dropna=False):
        phase_name = str(phase_name)
        term = ""
        if "terminated_reason" in sub.columns:
            term_series = sub["terminated_reason"].astype(str)
            term_nonempty = term_series[
                (~term_series.str.strip().eq("")) & (~term_series.str.lower().eq("nan"))
            ]
            if not term_nonempty.empty:
                term = str(term_nonempty.iloc[-1])
        rp = to_num(safe_col(sub, "route_progress_norm", np.nan), default=np.nan)
        phase_rows.append(
            {
                "phase": phase_name,
                "rows": int(len(sub)),
                "step_min": _series_stat_min(sub, "step"),
                "step_max": _series_stat_max(sub, "step"),
                "max_wp_idx": _series_stat_max(sub, "wp_idx"),
                "route_progress_max": float(rp.dropna().max()) if not rp.dropna().empty else np.nan,
                "route_progress_final": float(rp.dropna().iloc[-1]) if not rp.dropna().empty else np.nan,
                "terminated_reason_last": term,
            }
        )
    summary_by_phase = pd.DataFrame(phase_rows)
    path_summary = os.path.join(out_abs, "summary_by_phase.csv")
    summary_by_phase.to_csv(path_summary, index=False)
    saved_tables.append(os.path.basename(path_summary))

    # event tables
    spark_events = find_events(df, "spark_visible_dbg")
    path_spark = os.path.join(out_abs, "spark_events.csv")
    spark_events.to_csv(path_spark, index=False)
    saved_tables.append(os.path.basename(path_spark))

    spark_response_events = find_events(df, "spark_response_active_dbg")
    path_spark_resp = os.path.join(out_abs, "spark_response_events.csv")
    spark_response_events.to_csv(path_spark_resp, index=False)
    saved_tables.append(os.path.basename(path_spark_resp))

    bird_telemetry_cols = [
        "bird_visible_dbg",
        "bird_det_count_dbg",
        "bird_max_conf_dbg",
        "bird_hazard_active_dbg",
        "bird_response_active_dbg",
        "visual_hazard_active_dbg",
        "visual_hazard_score_dbg",
        "visual_hazard_confirm_steps_dbg",
        "visual_hazard_clear_counter_dbg",
        "safety_visible_dbg",
        "safety_conf_dbg",
        "safety_area_dbg",
        "safety_dx_dbg",
        "safety_hazard_score_dbg",
    ]
    bird_telemetry_present = any(c in df.columns for c in bird_telemetry_cols)

    def _flag_gt(col: str, thr: float) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        vals = to_num(safe_col(df, col, 0.0), default=0.0).fillna(0.0).to_numpy(dtype=np.float64)
        return vals > float(thr)

    bird_visible_series = _flag_gt("bird_visible_dbg", 0.5)
    bird_hazard_series = _flag_gt("bird_hazard_active_dbg", 0.5)
    visual_hazard_series = _flag_gt("visual_hazard_active_dbg", 0.5)
    safety_visible_series = _flag_gt("safety_visible_dbg", 0.5)
    visual_hazard_score_series = _flag_gt("visual_hazard_score_dbg", 0.0)
    safety_hazard_score_series = _flag_gt("safety_hazard_score_dbg", 0.0)

    bird_event_flag = (
        bird_visible_series
        | bird_hazard_series
        | visual_hazard_series
        | safety_visible_series
        | visual_hazard_score_series
        | safety_hazard_score_series
    )
    df["_bird_event_flag_dbg"] = bird_event_flag.astype(np.float64)

    source_cols_used = [
        c
        for c in [
            "bird_visible_dbg",
            "bird_hazard_active_dbg",
            "visual_hazard_active_dbg",
            "safety_visible_dbg",
            "visual_hazard_score_dbg",
            "safety_hazard_score_dbg",
        ]
        if c in df.columns
    ]
    bird_event_source_used = (
        f"combined_or:{','.join(source_cols_used)}"
        if source_cols_used
        else "none"
    )

    if source_cols_used:
        bird_events = find_events(df, "_bird_event_flag_dbg")
        if not bird_events.empty:
            bird_events.insert(1, "source_col", bird_event_source_used)
    else:
        bird_events = pd.DataFrame(
            columns=[
                "event_id",
                "source_col",
                "flag_col",
                "phase_filter",
                "start_index",
                "end_index",
                "start_global_step",
                "end_global_step",
                "start_step",
                "end_step",
                "duration_rows",
                "start_wp_idx",
                "end_wp_idx",
                "max_flag",
                "mean_flag",
            ]
        )
    path_bird = os.path.join(out_abs, "bird_events.csv")
    bird_events.to_csv(path_bird, index=False)
    saved_tables.append(os.path.basename(path_bird))
    bird_visible_rows = int(np.nansum(bird_visible_series.astype(np.float64)))
    bird_hazard_rows = int(np.nansum(bird_hazard_series.astype(np.float64)))
    visual_hazard_rows = int(np.nansum(visual_hazard_series.astype(np.float64)))
    safety_visible_rows = int(np.nansum(safety_visible_series.astype(np.float64)))

    bird_events_count = int(len(bird_events))
    conf_series = pd.Series(dtype=np.float64)
    if "bird_max_conf_dbg" in df.columns:
        conf_series = to_num(safe_col(df, "bird_max_conf_dbg", np.nan), default=np.nan).dropna()
    if conf_series.empty and "safety_conf_dbg" in df.columns:
        conf_series = to_num(safe_col(df, "safety_conf_dbg", np.nan), default=np.nan).dropna()
    bird_max_conf = float(conf_series.max()) if not conf_series.empty else np.nan

    bird_event_wp_indices: List[int] = []
    if not bird_events.empty:
        wp_values: List[int] = []
        for col in ("start_wp_idx", "end_wp_idx"):
            if col in bird_events.columns:
                vals = to_num(safe_col(bird_events, col, np.nan), default=np.nan).dropna().tolist()
                wp_values.extend([int(float(v)) for v in vals if np.isfinite(v)])
        bird_event_wp_indices = sorted(set(wp_values))

    if not bird_telemetry_present:
        bird_telemetry_status = "absent"
    elif (
        bird_events_count > 0
        or bird_visible_rows > 0
        or bird_hazard_rows > 0
        or visual_hazard_rows > 0
        or safety_visible_rows > 0
    ):
        bird_telemetry_status = "present_events"
    else:
        bird_telemetry_status = "present_no_events"

    def group_sum_by_wp(src_df, value_col: str, out_col: str):
        empty = pd.DataFrame(columns=["wp_idx", out_col])
        if src_df is None or src_df.empty:
            return empty
        if "wp_idx" not in src_df.columns or value_col not in src_df.columns:
            return empty

        tmp = src_df[["wp_idx", value_col]].copy()
        tmp["wp_idx"] = to_num(safe_col(tmp, "wp_idx", np.nan), default=np.nan)
        tmp[value_col] = to_num(safe_col(tmp, value_col, np.nan), default=np.nan)
        tmp = tmp[np.isfinite(tmp["wp_idx"])].copy()
        if tmp.empty:
            return empty

        tmp["wp_idx"] = tmp["wp_idx"].astype(int)
        tmp[value_col] = (tmp[value_col].fillna(0.0) > 0.5).astype(float)
        out = (
            tmp.groupby("wp_idx", as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: out_col})
        )
        out[out_col] = to_num(safe_col(out, out_col, 0.0), default=0.0).fillna(0.0)
        return out[["wp_idx", out_col]]

    inspect = df[df["phase"] == "INSPECT"].copy()
    if not inspect.empty and "wp_idx" in inspect.columns:
        inspect_wp = inspect.copy()
        inspect_wp["wp_idx"] = to_num(safe_col(inspect_wp, "wp_idx", np.nan), default=np.nan)
        inspect_wp = inspect_wp[np.isfinite(inspect_wp["wp_idx"])].copy()
        if not inspect_wp.empty:
            inspect_wp["wp_idx"] = inspect_wp["wp_idx"].astype(int)
            grouped = inspect_wp.groupby("wp_idx", as_index=False)
            inspect_by_waypoint = grouped.agg(steps=("global_step", "count"))

            if "route_progress_norm" in inspect_wp.columns:
                rp_max = (
                    inspect_wp.groupby("wp_idx", as_index=False)["route_progress_norm"]
                    .max()
                    .rename(columns={"route_progress_norm": "route_progress_norm_max"})
                )
                if (
                    "wp_idx" in inspect_by_waypoint.columns
                    and "wp_idx" in rp_max.columns
                    and (not rp_max.empty)
                ):
                    inspect_by_waypoint = inspect_by_waypoint.merge(rp_max, on="wp_idx", how="left")
                else:
                    inspect_by_waypoint["route_progress_norm_max"] = np.nan
            else:
                inspect_by_waypoint["route_progress_norm_max"] = np.nan

            for value_col, out_col in [
                ("spark_visible_dbg", "spark_visible_rows"),
                ("spark_response_active_dbg", "spark_response_rows"),
                ("cable_safety_active", "cable_safety_rows"),
                ("gate_blocked_dbg", "gate_blocked_rows"),
                ("bird_visible_dbg", "bird_visible_rows"),
                ("bird_hazard_active_dbg", "bird_hazard_rows"),
                ("visible", "visible_rows"),
            ]:
                tmp = group_sum_by_wp(inspect_wp, value_col=value_col, out_col=out_col)
                if not tmp.empty and "wp_idx" in inspect_by_waypoint.columns:
                    inspect_by_waypoint = inspect_by_waypoint.merge(tmp, on="wp_idx", how="left")
                else:
                    inspect_by_waypoint[out_col] = 0.0

            for out_col in [
                "spark_visible_rows",
                "spark_response_rows",
                "cable_safety_rows",
                "gate_blocked_rows",
                "bird_visible_rows",
                "bird_hazard_rows",
                "visible_rows",
            ]:
                if out_col not in inspect_by_waypoint.columns:
                    inspect_by_waypoint[out_col] = 0.0
                inspect_by_waypoint[out_col] = (
                    to_num(safe_col(inspect_by_waypoint, out_col, 0.0), default=0.0)
                    .fillna(0.0)
                    .astype(float)
                )
        else:
            inspect_by_waypoint = pd.DataFrame(
                columns=[
                    "wp_idx",
                    "steps",
                    "route_progress_norm_max",
                    "spark_visible_rows",
                    "spark_response_rows",
                    "cable_safety_rows",
                    "gate_blocked_rows",
                    "bird_visible_rows",
                    "bird_hazard_rows",
                    "visible_rows",
                ]
            )
    else:
        inspect_by_waypoint = pd.DataFrame(
            columns=[
                "wp_idx",
                "steps",
                "route_progress_norm_max",
                "spark_visible_rows",
                "spark_response_rows",
                "cable_safety_rows",
                "gate_blocked_rows",
                "bird_visible_rows",
                "bird_hazard_rows",
                "visible_rows",
            ]
        )
    path_inspect_wp = os.path.join(out_abs, "inspect_by_waypoint.csv")
    inspect_by_waypoint.to_csv(path_inspect_wp, index=False)
    saved_tables.append(os.path.basename(path_inspect_wp))

    # plots
    def _plot_and_save(
        filename: str,
        build_fn,
        required_all: Optional[List[str]] = None,
        required_any: Optional[List[str]] = None,
    ) -> None:
        missing_all: List[str] = []
        if required_all:
            missing_all = [c for c in required_all if c not in df.columns]
            if missing_all:
                reason = f"missing columns {missing_all}"
                print(f"[REPORT] skipped plot {filename}: {reason}")
                skipped_plots.append({"name": filename, "reason": reason})
                return
        if required_any:
            present_any = [c for c in required_any if c in df.columns]
            if len(present_any) == 0:
                reason = f"missing any of {required_any}"
                print(f"[REPORT] skipped plot {filename}: {reason}")
                skipped_plots.append({"name": filename, "reason": reason})
                return
        try:
            fig = build_fn()
            if fig is None:
                reason = "no_data"
                print(f"[REPORT] skipped plot {filename}: {reason}")
                skipped_plots.append({"name": filename, "reason": reason})
                return
            path = os.path.join(out_abs, filename)
            save_plot(fig, path, dpi=int(dpi))
            saved_plots.append(os.path.basename(path))
            print(f"[REPORT] saved plot: {os.path.abspath(path)}")
        except Exception as e:
            reason = f"error: {e}"
            print(f"[REPORT][WARN] plot skipped ({filename}): {e}")
            skipped_plots.append({"name": filename, "reason": reason})

    def _fig_wp():
        if "wp_idx" not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        phases = sorted(df["phase"].unique().tolist())
        for ph in phases:
            sub = df[df["phase"] == ph]
            ax.plot(sub["global_step"], to_num(safe_col(sub, "wp_idx", np.nan), default=np.nan), linewidth=1.2, label=ph)
        ax.set_title("Mission WP Index")
        ax.set_xlabel("global_step")
        ax.set_ylabel("wp_idx")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_progress():
        if "route_progress_norm" not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["global_step"], to_num(safe_col(df, "route_progress_norm", np.nan), default=np.nan), linewidth=1.2)
        ax.set_title("Route Progress")
        ax.set_xlabel("global_step")
        ax.set_ylabel("route_progress_norm")
        ax.grid(alpha=0.25)
        return fig

    def _fig_spark():
        needed = [c for c in ["spark_visible_dbg", "spark_response_active_dbg", "spark_event_score_dbg"] if c in df.columns]
        if not needed:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        for c in needed:
            ax.plot(df["global_step"], to_num(safe_col(df, c, 0.0), default=0.0), linewidth=1.2, label=c)
        ax.set_title("Spark Detection / Response")
        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_yolo():
        needed = [c for c in ["visible", "conf"] if c in df.columns]
        if not needed:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        for c in needed:
            ax.plot(df["global_step"], to_num(safe_col(df, c, np.nan), default=np.nan), linewidth=1.2, label=c)
        ax.set_title("YOLO Visible/Conf")
        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_cable():
        if "cable_safety_active" not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["global_step"], to_num(safe_col(df, "cable_safety_active", 0.0), default=0.0), linewidth=1.2)
        ax.set_title("Cable Safety Active")
        ax.set_xlabel("global_step")
        ax.set_ylabel("cable_safety_active")
        ax.grid(alpha=0.25)
        return fig

    def _fig_timing():
        needed = [c for c in ["loop_dt_ms", "env_step_dt_ms", "model_predict_dt_ms", "video_dt_ms"] if c in df.columns]
        if not needed:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        for c in needed:
            ax.plot(df["global_step"], to_num(safe_col(df, c, np.nan), default=np.nan), linewidth=1.1, label=c)
        ax.set_title("Timing")
        ax.set_xlabel("global_step")
        ax.set_ylabel("ms")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_nav_shake():
        nav = df[df["phase"] == "NAV"].copy()
        if nav.empty or "local_distance" not in nav.columns:
            return None
        nav = nav.sort_values("global_step").copy()
        nav["abs_delta_local_distance"] = to_num(safe_col(nav, "local_distance", np.nan), default=np.nan).diff().abs()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(nav["global_step"], nav["abs_delta_local_distance"], linewidth=1.2, label="abs_delta_local_distance")
        for c, color in [
            ("planner_replan_step_dbg", "tab:red"),
            ("local_path_rebuilt_this_step_dbg", "tab:orange"),
            ("local_target_changed_this_step_dbg", "tab:green"),
        ]:
            if c in nav.columns:
                f = to_num(safe_col(nav, c, 0.0), default=0.0) > 0.5
                if f.any():
                    ax.scatter(
                        nav.loc[f, "global_step"],
                        nav.loc[f, "abs_delta_local_distance"],
                        s=8,
                        alpha=0.7,
                        label=c,
                        color=color,
                    )
        ax.set_title("NAV Shake Proxy vs Replan")
        ax.set_xlabel("global_step")
        ax.set_ylabel("abs_delta_local_distance")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_inspect_wp():
        if inspect_by_waypoint.empty:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        x = inspect_by_waypoint["wp_idx"].to_numpy()
        ax.bar(x, inspect_by_waypoint["steps"].to_numpy(), alpha=0.75, label="steps")
        if "spark_visible_rows" in inspect_by_waypoint.columns:
            ax.plot(x, inspect_by_waypoint["spark_visible_rows"].to_numpy(), marker="o", linewidth=1.0, label="spark_visible_rows")
        if "spark_response_rows" in inspect_by_waypoint.columns:
            ax.plot(x, inspect_by_waypoint["spark_response_rows"].to_numpy(), marker="o", linewidth=1.0, label="spark_response_rows")
        ax.set_title("INSPECT Events by Waypoint")
        ax.set_xlabel("wp_idx")
        ax.set_ylabel("count")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    def _fig_bird_safety():
        needed = [
            c
            for c in [
                "bird_visible_dbg",
                "bird_hazard_active_dbg",
                "visual_hazard_active_dbg",
                "visual_hazard_score_dbg",
                "safety_visible_dbg",
                "safety_hazard_score_dbg",
            ]
            if c in df.columns
        ]
        if not needed:
            return None
        fig, ax = plt.subplots(figsize=(12, 4))
        for c in needed:
            ax.plot(df["global_step"], to_num(safe_col(df, c, 0.0), default=0.0), linewidth=1.1, label=c)
        ax.set_title("Bird / Safety Events")
        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        return fig

    _plot_and_save("01_mission_wp_idx.png", _fig_wp, required_all=["wp_idx"])
    _plot_and_save("02_route_progress.png", _fig_progress, required_all=["route_progress_norm"])
    _plot_and_save(
        "03_spark_detection_response.png",
        _fig_spark,
        required_any=["spark_visible_dbg", "spark_response_active_dbg"],
    )
    _plot_and_save("04_yolo_visible_conf.png", _fig_yolo, required_any=["visible", "conf"])
    _plot_and_save("05_cable_safety.png", _fig_cable, required_all=["cable_safety_active"])
    _plot_and_save(
        "06_timing.png",
        _fig_timing,
        required_any=["loop_dt_ms", "env_step_dt_ms", "model_predict_dt_ms", "video_dt_ms"],
    )
    _plot_and_save("07_nav_shake_replan.png", _fig_nav_shake, required_all=["local_distance", "planner_replan_step_dbg"])
    _plot_and_save("08_inspect_events_by_waypoint.png", _fig_inspect_wp)
    _plot_and_save(
        "09_bird_safety_events.png",
        _fig_bird_safety,
        required_any=[
            "bird_visible_dbg",
            "bird_hazard_active_dbg",
            "visual_hazard_active_dbg",
            "visual_hazard_score_dbg",
            "safety_visible_dbg",
            "safety_hazard_score_dbg",
        ],
    )

    # markdown + report diagnostics
    term = ""
    if "terminated_reason" in df.columns:
        tr = df["terminated_reason"].astype(str)
        tr2 = tr[(~tr.str.strip().eq("")) & (~tr.str.lower().eq("nan"))]
        if not tr2.empty:
            term = str(tr2.iloc[-1])
    wp_all = to_num(safe_col(df, "wp_idx", np.nan), default=np.nan).dropna()
    max_wp = float(wp_all.max()) if not wp_all.empty else np.nan
    cable_rows = int(np.nansum(to_num(safe_col(df, "cable_safety_active", 0.0), default=0.0).to_numpy() > 0.5))
    phase_counts = (
        df.groupby("phase")["global_step"].count().reset_index().rename(columns={"global_step": "rows"})
        if "phase" in df.columns
        else None
    )
    spark_count = int(len(spark_events))
    spark_response_count = int(len(spark_response_events))
    bird_count = int(bird_events_count)
    spark_visible_rows = int(
        np.nansum(to_num(safe_col(df, "spark_visible_dbg", 0.0), default=0.0).to_numpy(dtype=np.float64) > 0.5)
    )
    spark_response_rows = int(
        np.nansum(to_num(safe_col(df, "spark_response_active_dbg", 0.0), default=0.0).to_numpy(dtype=np.float64) > 0.5)
    )
    snapshots_index_path = os.path.join(out_abs, "spark_snapshots_index.csv")
    snapshots_diag_path = os.path.join(out_abs, "spark_snapshots_diagnostics.json")
    bird_snapshots_index_path = os.path.join(out_abs, "bird_snapshots_index.csv")
    snapshots_df = None
    bird_snapshots_df = None
    snapshots_diag: Dict[str, Any] = {}
    if os.path.exists(snapshots_index_path):
        try:
            snapshots_df = pd.read_csv(snapshots_index_path)
            saved_tables.append(os.path.basename(snapshots_index_path))
        except Exception:
            snapshots_df = None
    if os.path.exists(snapshots_diag_path):
        try:
            with open(snapshots_diag_path, "r", encoding="utf-8") as f:
                snapshots_diag = json.load(f)
            saved_tables.append(os.path.basename(snapshots_diag_path))
        except Exception:
            snapshots_diag = {}
    if os.path.exists(bird_snapshots_index_path):
        try:
            bird_snapshots_df = pd.read_csv(bird_snapshots_index_path)
            saved_tables.append(os.path.basename(bird_snapshots_index_path))
        except Exception:
            bird_snapshots_df = None

    md_lines = [
        "# Demo Video Technical Report",
        "",
        f"- Source CSV: `{csv_abs}`",
        f"- Rows: {int(len(df))}",
        f"- Final terminated_reason: `{term}`",
        f"- Max wp_idx: {max_wp}",
        f"- Spark events count: {spark_count}",
        f"- Spark response events count: {spark_response_count}",
        f"- Spark visible rows: {spark_visible_rows}",
        f"- Spark response rows: {spark_response_rows}",
        f"- Bird events count: {bird_count}",
        f"- Cable safety active rows: {cable_rows}",
        "",
        "## Phase Counts",
    ]
    if phase_counts is not None and not phase_counts.empty:
        for _, r in phase_counts.iterrows():
            md_lines.append(f"- {str(r['phase'])}: {int(r['rows'])}")
    else:
        md_lines.append("- no phase info")

    md_lines.extend(["", "## Spark snapshots"])
    snapshot_count = 0
    if snapshots_df is not None and not snapshots_df.empty:
        snapshot_count = int(len(snapshots_df))
        wp_values = []
        if "wp_idx" in snapshots_df.columns:
            wp_values = sorted(
                {
                    int(float(v))
                    for v in snapshots_df["wp_idx"].tolist()
                    if str(v).strip() != "" and str(v).lower() != "nan"
                }
            )
        snap_dir = ""
        if "snapshot_dir" in snapshots_df.columns:
            first_dir = snapshots_df["snapshot_dir"].astype(str)
            first_dir = first_dir[first_dir.str.strip() != ""]
            if not first_dir.empty:
                snap_dir = str(first_dir.iloc[0])
        md_lines.append(f"- saved snapshots: {snapshot_count}")
        md_lines.append(f"- snapshot dir: `{snap_dir if snap_dir else 'unknown'}`")
        md_lines.append(f"- wp_idx with spark: {wp_values if wp_values else '[]'}")
        md_lines.append("- files:")
        if "file_name" in snapshots_df.columns:
            for fn in snapshots_df["file_name"].astype(str).tolist():
                md_lines.append(f"  - {fn}")
        else:
            md_lines.append("  - file_name column missing")
    else:
        md_lines.append("- saved snapshots: 0")
        md_lines.append("- snapshot dir: not available")
        md_lines.append("- wp_idx with spark: []")

    if spark_visible_rows > int(snapshot_count):
        reasons: List[str] = []
        max_count = snapshots_diag.get("max_count", None)
        min_gap_steps = snapshots_diag.get("min_gap_steps", None)
        skip_max = int(snapshots_diag.get("skipped_due_max_count", 0) or 0)
        skip_gap = int(snapshots_diag.get("skipped_due_min_gap", 0) or 0)
        skip_no_frame = int(snapshots_diag.get("skipped_due_no_frame", 0) or 0)
        if skip_max > 0:
            reasons.append(f"max_count={max_count} (skipped={skip_max})")
        if skip_gap > 0:
            reasons.append(f"min_gap_steps={min_gap_steps} (skipped={skip_gap})")
        if skip_no_frame > 0:
            reasons.append(f"no_frame (skipped={skip_no_frame})")
        if not reasons:
            reasons.append("unknown (check spark snapshots diagnostics)")
        md_lines.append(f"- spark_visible rows > saved snapshots: reasons={'; '.join(reasons)}")

    bird_snapshot_count = 0
    if bird_snapshots_df is not None and not bird_snapshots_df.empty:
        bird_snapshot_count = int(len(bird_snapshots_df))

    md_lines.extend(["", "## Bird / safety events"])
    md_lines.append("- telemetry source: planner_info.visual_debug")
    md_lines.append(f"- bird event source used: {bird_event_source_used}")
    md_lines.append(f"- bird telemetry status: {bird_telemetry_status}")
    md_lines.append(f"- bird events count: {bird_events_count}")
    md_lines.append(f"- bird visible rows: {bird_visible_rows}")
    md_lines.append(f"- bird hazard active rows: {bird_hazard_rows}")
    md_lines.append(f"- visual hazard rows: {visual_hazard_rows}")
    md_lines.append(f"- safety visible rows: {safety_visible_rows}")
    md_lines.append(
        f"- max bird confidence: {bird_max_conf:.4f}"
        if not np.isnan(bird_max_conf)
        else "- max bird confidence: nan"
    )
    md_lines.append(
        f"- wp_idx with bird events: {bird_event_wp_indices if bird_event_wp_indices else '[]'}"
    )
    if bird_telemetry_status == "absent":
        md_lines.append("Bird telemetry absent in light-log.")
    elif bird_telemetry_status == "present_no_events":
        md_lines.append("Bird detector telemetry present, no bird events detected.")
    else:
        md_lines.append("Bird detector telemetry present, bird-related events detected.")
    md_lines.append(f"- bird snapshots saved: {bird_snapshot_count}")

    md_lines.extend(["", "## Saved Tables"])
    for t in saved_tables:
        md_lines.append(f"- {t}")

    md_lines.extend(["", "## Saved Plots"])
    for p in saved_plots:
        md_lines.append(f"- {p}")

    md_lines.extend(
        [
            "",
            "## Report diagnostics",
            f"- created plots count: {len(saved_plots)}",
            f"- skipped plots count: {len(skipped_plots)}",
            f"- spark snapshots saved count: {snapshot_count}",
            f"- bird snapshots saved count: {bird_snapshot_count}",
            f"- bird telemetry status: {bird_telemetry_status}",
        ]
    )
    if skipped_plots:
        md_lines.append("- skipped plots list:")
        for sp in skipped_plots:
            md_lines.append(f"  - {sp.get('name')}: {sp.get('reason')}")

    md_path = os.path.join(out_abs, "demo_video_technical_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    tables_for_index = list(saved_tables) + ["report_index.json"]
    report_index = {
        "created_plots": saved_plots,
        "skipped_plots": skipped_plots,
        "created_tables": tables_for_index,
        "spark_snapshots": int(snapshot_count),
        "spark_visible_rows": int(spark_visible_rows),
        "spark_response_rows": int(spark_response_rows),
        "bird_telemetry_present": bool(bird_telemetry_present),
        "bird_telemetry_status": bird_telemetry_status,
        "bird_events_count": int(bird_events_count),
        "bird_event_source_used": bird_event_source_used,
        "bird_visible_rows": int(bird_visible_rows),
        "bird_hazard_rows": int(bird_hazard_rows),
        "visual_hazard_rows": int(visual_hazard_rows),
        "safety_visible_rows": int(safety_visible_rows),
        "bird_max_conf": float(bird_max_conf) if not np.isnan(bird_max_conf) else np.nan,
        "bird_event_wp_indices": bird_event_wp_indices,
        "bird_snapshots": int(bird_snapshot_count),
        "report_dir": out_abs,
        "csv_path": csv_abs,
    }
    report_index_path = os.path.join(out_abs, "report_index.json")
    with open(report_index_path, "w", encoding="utf-8") as f:
        json.dump(report_index, f, indent=2, ensure_ascii=False)
    saved_tables.append(os.path.basename(report_index_path))

    print(f"[REPORT] plots: created={len(saved_plots)} skipped={len(skipped_plots)}")
    print(f"[REPORT] spark snapshots: saved={snapshot_count}")
    print(
        f"[REPORT] bird telemetry: "
        f"{'present' if bird_telemetry_present else 'absent'}, "
        f"events={bird_events_count}, snapshots={bird_snapshot_count}"
    )

    return out_abs


class SparkSnapshotSaver:
    INDEX_FIELDNAMES = [
        "snapshot_id",
        "spark_event_id",
        "file_name",
        "phase",
        "global_step",
        "env_step",
        "wp_idx",
        "spark_visible_dbg",
        "spark_max_conf_dbg",
        "spark_event_score_dbg",
        "spark_response_active_dbg",
        "spark_bbox_x1_dbg",
        "spark_bbox_y1_dbg",
        "spark_bbox_x2_dbg",
        "spark_bbox_y2_dbg",
        "snapshot_dir",
    ]

    def __init__(
        self,
        enabled: bool,
        report_dir: str,
        snapshot_dirname: str = "spark_snapshots",
        max_count: int = 3,
        min_gap_steps: int = 10,
    ):
        self.enabled = bool(enabled)
        self.max_count = max(1, int(max_count))
        self.min_gap_steps = max(0, int(min_gap_steps))
        self.snapshot_dir = os.path.join(report_dir, snapshot_dirname)
        self.index_path = os.path.join(report_dir, "spark_snapshots_index.csv")
        self._rows: List[Dict[str, Any]] = []
        self._last_saved_env_step: Optional[int] = None
        self._spark_event_id: int = 0
        self._prev_visible: bool = False
        self._last_saved_event_id: int = -1
        self.visible_rows_seen: int = 0
        self.skipped_due_max_count: int = 0
        self.skipped_due_min_gap: int = 0
        self.skipped_due_no_frame: int = 0
        self.saved_with_bbox_unavailable: int = 0
        self.saved_due_new_event: int = 0
        self.saved_event_ids: List[int] = []
        if self.enabled:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    @property
    def count(self) -> int:
        return len(self._rows)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        phase: str,
        env_step: int,
        wp_idx: int,
        spark_conf: float,
        spark_event_score: float,
        spark_response_active: float,
        term_reason: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        has_bbox = bool((x2 > x1) and (y2 > y1) and (x1 >= 0.0) and (y1 >= 0.0))
        if has_bbox:
            xi1 = int(np.clip(round(x1), 0, max(w - 1, 0)))
            yi1 = int(np.clip(round(y1), 0, max(h - 1, 0)))
            xi2 = int(np.clip(round(x2), 0, max(w - 1, 0)))
            yi2 = int(np.clip(round(y2), 0, max(h - 1, 0)))
            if xi2 > xi1 and yi2 > yi1:
                cv2.rectangle(out, (xi1, yi1), (xi2, yi2), (0, 0, 255), 2)
            else:
                has_bbox = False

        lines = [
            "SPARK DETECTED",
            f"phase={phase}",
            f"step={env_step}",
            f"wp_idx={wp_idx}",
            f"conf={spark_conf:.3f}",
            f"event_score={spark_event_score:.3f}",
            f"response_active={int(spark_response_active > 0.5)}",
        ]
        if term_reason:
            lines.append(f"terminated_reason={term_reason}")
        if not has_bbox:
            lines.append("spark bbox unavailable")

        y = 28
        for line in lines:
            cv2.putText(
                out,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26
        return out

    def maybe_save(
        self,
        frame_bgr: Optional[np.ndarray],
        phase: str,
        env_step: int,
        global_step: int,
        info: Dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False
        if str(phase).upper() != "INSPECT":
            return False

        spark_visible = as_float(safe_info_deep(info, "spark_visible_dbg", 0.0), 0.0)
        is_visible = bool(spark_visible > 0.5)

        if is_visible and (not self._prev_visible):
            self._spark_event_id += 1
        self._prev_visible = is_visible

        if not is_visible:
            return False

        self.visible_rows_seen += 1
        current_event_id = int(max(1, self._spark_event_id))

        if self.count >= self.max_count:
            self.skipped_due_max_count += 1
            return False

        if self._last_saved_env_step is not None and int(env_step) < int(self._last_saved_env_step):
            self._last_saved_env_step = None

        allow_due_new_event = bool(current_event_id != self._last_saved_event_id)
        if self._last_saved_env_step is not None:
            if (int(env_step) - int(self._last_saved_env_step) < self.min_gap_steps) and (not allow_due_new_event):
                self.skipped_due_min_gap += 1
                return False

        if frame_bgr is None:
            self.skipped_due_no_frame += 1
            return False

        wp_idx = int(extract_wp_idx(info))
        spark_conf = as_float(safe_info_deep(info, "spark_max_conf_dbg", 0.0), 0.0)
        spark_event_score = as_float(safe_info_deep(info, "spark_event_score_dbg", 0.0), 0.0)
        spark_response_active = as_float(safe_info_deep(info, "spark_response_active_dbg", 0.0), 0.0)
        term_reason = str(safe_info_deep(info, "terminated_reason", ""))

        x1 = as_float(safe_info_deep(info, "spark_bbox_x1_dbg", 0.0), 0.0)
        y1 = as_float(safe_info_deep(info, "spark_bbox_y1_dbg", 0.0), 0.0)
        x2 = as_float(safe_info_deep(info, "spark_bbox_x2_dbg", 0.0), 0.0)
        y2 = as_float(safe_info_deep(info, "spark_bbox_y2_dbg", 0.0), 0.0)
        bbox_available = bool((x2 > x1) and (y2 > y1))

        annotated = self._draw_overlay(
            frame=frame_bgr,
            phase=str(phase),
            env_step=int(env_step),
            wp_idx=int(wp_idx),
            spark_conf=float(spark_conf),
            spark_event_score=float(spark_event_score),
            spark_response_active=float(spark_response_active),
            term_reason=str(term_reason),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        )

        snap_id = self.count + 1
        filename = f"spark_{snap_id:03d}_step_{int(env_step)}_wp_{int(wp_idx)}.jpg"
        out_path = os.path.join(self.snapshot_dir, filename)
        ok = bool(cv2.imwrite(out_path, annotated))
        if not ok:
            return False

        self._rows.append(
            {
                "snapshot_id": int(snap_id),
                "spark_event_id": int(current_event_id),
                "file_name": filename,
                "phase": str(phase),
                "global_step": int(global_step),
                "env_step": int(env_step),
                "wp_idx": int(wp_idx),
                "spark_visible_dbg": float(spark_visible),
                "spark_max_conf_dbg": float(spark_conf),
                "spark_event_score_dbg": float(spark_event_score),
                "spark_response_active_dbg": float(spark_response_active),
                "spark_bbox_x1_dbg": float(x1),
                "spark_bbox_y1_dbg": float(y1),
                "spark_bbox_x2_dbg": float(x2),
                "spark_bbox_y2_dbg": float(y2),
                "snapshot_dir": os.path.abspath(self.snapshot_dir),
            }
        )
        self._last_saved_env_step = int(env_step)
        self._last_saved_event_id = int(current_event_id)
        self.saved_event_ids.append(int(current_event_id))
        if allow_due_new_event:
            self.saved_due_new_event += 1
        if not bbox_available:
            self.saved_with_bbox_unavailable += 1
        return True

    def finalize(self) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.INDEX_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self.INDEX_FIELDNAMES})

        diag = {
            "enabled": bool(self.enabled),
            "snapshot_dir": os.path.abspath(self.snapshot_dir),
            "index_path": os.path.abspath(self.index_path),
            "max_count": int(self.max_count),
            "min_gap_steps": int(self.min_gap_steps),
            "saved_count": int(self.count),
            "saved_event_ids": sorted({int(x) for x in self.saved_event_ids}),
            "visible_rows_seen": int(self.visible_rows_seen),
            "skipped_due_max_count": int(self.skipped_due_max_count),
            "skipped_due_min_gap": int(self.skipped_due_min_gap),
            "skipped_due_no_frame": int(self.skipped_due_no_frame),
            "saved_with_bbox_unavailable": int(self.saved_with_bbox_unavailable),
            "saved_due_new_event": int(self.saved_due_new_event),
        }
        diag_path = os.path.join(os.path.dirname(self.index_path), "spark_snapshots_diagnostics.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)


class BirdSnapshotSaver:
    INDEX_FIELDNAMES = [
        "snapshot_id",
        "file_name",
        "phase",
        "global_step",
        "env_step",
        "wp_idx",
        "bird_visible_dbg",
        "bird_max_conf_dbg",
        "bird_hazard_active_dbg",
        "visual_hazard_score_dbg",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "snapshot_dir",
    ]

    def __init__(
        self,
        enabled: bool,
        report_dir: str,
        snapshot_dirname: str = "bird_snapshots",
        max_count: int = 3,
        min_gap_steps: int = 10,
    ):
        self.enabled = bool(enabled)
        self.max_count = max(1, int(max_count))
        self.min_gap_steps = max(0, int(min_gap_steps))
        self.snapshot_dir = os.path.join(report_dir, snapshot_dirname)
        self.index_path = os.path.join(report_dir, "bird_snapshots_index.csv")
        self._rows: List[Dict[str, Any]] = []
        self._last_saved_env_step: Optional[int] = None
        self._bird_event_id: int = 0
        self._prev_active: bool = False
        self._last_saved_event_id: int = -1
        if self.enabled:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    @property
    def count(self) -> int:
        return len(self._rows)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        phase: str,
        env_step: int,
        wp_idx: int,
        conf: float,
        hazard_score: float,
        visual_hazard_active: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        has_bbox = bool((x2 > x1) and (y2 > y1) and (x1 >= 0.0) and (y1 >= 0.0))
        if has_bbox:
            xi1 = int(np.clip(round(x1), 0, max(w - 1, 0)))
            yi1 = int(np.clip(round(y1), 0, max(h - 1, 0)))
            xi2 = int(np.clip(round(x2), 0, max(w - 1, 0)))
            yi2 = int(np.clip(round(y2), 0, max(h - 1, 0)))
            if xi2 > xi1 and yi2 > yi1:
                cv2.rectangle(out, (xi1, yi1), (xi2, yi2), (0, 255, 255), 2)
            else:
                has_bbox = False

        lines = [
            "BIRD / SAFETY DETECTED",
            f"phase={phase}",
            f"step={env_step}",
            f"wp_idx={wp_idx}",
            f"conf={conf:.3f}" if not np.isnan(conf) else "conf=nan",
            f"hazard_score={hazard_score:.3f}",
            f"visual_hazard_active={int(visual_hazard_active > 0.5)}",
        ]
        if not has_bbox:
            lines.append("bird bbox unavailable")

        y = 28
        for line in lines:
            cv2.putText(
                out,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26
        return out

    def maybe_save(
        self,
        frame_bgr: Optional[np.ndarray],
        phase: str,
        env_step: int,
        global_step: int,
        info: Dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False
        if str(phase).upper() != "NAV":
            return False

        bird = extract_bird_light_metrics(info)
        active = bool(
            (float(bird["bird_visible_dbg"]) > 0.5)
            or (float(bird["bird_hazard_active_dbg"]) > 0.5)
            or (float(bird["visual_hazard_active_dbg"]) > 0.5)
        )
        if active and (not self._prev_active):
            self._bird_event_id += 1
        self._prev_active = active
        if not active:
            return False

        if self.count >= self.max_count:
            return False

        allow_due_new_event = bool(self._bird_event_id != self._last_saved_event_id)
        if self._last_saved_env_step is not None:
            if (int(env_step) - int(self._last_saved_env_step) < self.min_gap_steps) and (not allow_due_new_event):
                return False

        if frame_bgr is None:
            return False

        wp_idx = int(extract_wp_idx(info))
        x1, y1, x2, y2 = extract_bird_bbox(info)

        annotated = self._draw_overlay(
            frame=frame_bgr,
            phase=str(phase),
            env_step=int(env_step),
            wp_idx=int(wp_idx),
            conf=float(bird["bird_max_conf_dbg"]) if not np.isnan(float(bird["bird_max_conf_dbg"])) else np.nan,
            hazard_score=float(bird["visual_hazard_score_dbg"]),
            visual_hazard_active=float(bird["visual_hazard_active_dbg"]),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        )

        snap_id = self.count + 1
        filename = f"bird_{snap_id:03d}_step_{int(env_step)}_wp_{int(wp_idx)}.jpg"
        out_path = os.path.join(self.snapshot_dir, filename)
        if not bool(cv2.imwrite(out_path, annotated)):
            return False

        self._rows.append(
            {
                "snapshot_id": int(snap_id),
                "file_name": filename,
                "phase": str(phase),
                "global_step": int(global_step),
                "env_step": int(env_step),
                "wp_idx": int(wp_idx),
                "bird_visible_dbg": float(bird["bird_visible_dbg"]),
                "bird_max_conf_dbg": float(bird["bird_max_conf_dbg"]),
                "bird_hazard_active_dbg": float(bird["bird_hazard_active_dbg"]),
                "visual_hazard_score_dbg": float(bird["visual_hazard_score_dbg"]),
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "snapshot_dir": os.path.abspath(self.snapshot_dir),
            }
        )
        self._last_saved_env_step = int(env_step)
        self._last_saved_event_id = int(self._bird_event_id)
        return True

    def finalize(self) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.INDEX_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self.INDEX_FIELDNAMES})


class LightLogger:
    FIELDNAMES = [
        "episode",
        "phase",
        "step",
        "loop_dt_ms",
        "env_step_dt_ms",
        "model_predict_dt_ms",
        "video_dt_ms",
        "reward",
        "episode_reward",
        "wp_idx",
        "point_mode",
        "local_distance",
        "final_distance",
        "route_progress_norm",
        "visual_gate_required_dbg",
        "gate_blocked_dbg",
        "target_yaw_error_rad",
        "planner_replan_step_dbg",
        "local_path_rebuilt_this_step_dbg",
        "local_target_changed_this_step_dbg",
        "center_error",
        "visible",
        "conf",
        "terminated_reason",
        "cable_safety_active",
        "bird_visible_dbg",
        "bird_det_count_dbg",
        "bird_max_conf_dbg",
        "bird_center_dx_dbg",
        "bird_center_dy_dbg",
        "bird_area_dbg",
        "bird_hazard_active_dbg",
        "bird_response_active_dbg",
        "visual_hazard_active_dbg",
        "visual_hazard_score_dbg",
        "visual_hazard_confirm_steps_dbg",
        "visual_hazard_clear_counter_dbg",
        "safety_visible_dbg",
        "safety_conf_dbg",
        "safety_area_dbg",
        "safety_dx_dbg",
        "safety_hazard_score_dbg",
        "spark_visible_dbg",
        "spark_det_count_dbg",
        "spark_max_conf_dbg",
        "spark_event_score_dbg",
        "spark_response_active_dbg",
        "spark_override_steps_left_dbg",
        "spark_action_override_changed_dbg",
        "spark_camera_id_dbg",
        "spark_iou_threshold_dbg",
        "spark_infer_count_dbg",
        "spark_infer_interval_steps_dbg",
        "spark_infer_time_ms_dbg",
    ]

    def __init__(self, enabled: bool, path: str):
        self.enabled = bool(enabled)
        self.path = path
        self._header_written = False
        if self.enabled:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def write(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        normalized = {k: row.get(k, "") for k in self.FIELDNAMES}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(normalized)

class VideoRecorder:
    def __init__(
        self,
        enabled: bool,
        path: str,
        fps: int,
        width: int,
        height: int,
        camera_name: str,
        show_overlay: bool,
        capture_every_n_steps: int = 3,
        video_async: bool = True,
    ):
        self.enabled = bool(enabled)
        self.path = path
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.camera_name = str(camera_name)
        self.show_overlay = bool(show_overlay)
        self.capture_every_n_steps = max(1, int(capture_every_n_steps))
        self.video_async = bool(video_async)

        self._writer: Optional[cv2.VideoWriter] = None
        self._sync_client = None
        self._thread_client = None
        self._vehicle_name = ""
        self._ip_address = "127.0.0.1"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_requested_step = -1
        self._last_written_step = -1
        self._last_video_dt_ms = 0.0
        self._last_frame_write_ts = 0.0
        self._latest_phase = ""
        self._latest_step = 0
        self._latest_overlay_info: Dict[str, Any] = {}
        self._last_frame_bgr: Optional[np.ndarray] = None

        if self.enabled:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, float(self.fps), (self.width, self.height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {path}")

    @staticmethod
    def _extract_overlay_info(info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "current_waypoint_idx": safe_info(info, "current_waypoint_idx", -1),
            "current_waypoint_idx_dbg": safe_info(info, "current_waypoint_idx_dbg", -1),
            "debug_current_waypoint_idx": safe_info(info, "debug_current_waypoint_idx", -1),
            "waypoint_idx": safe_info(info, "waypoint_idx", -1),
            "wp_idx_dbg": safe_info(info, "wp_idx_dbg", -1),
            "route_progress_norm": safe_info(info, "route_progress_norm", np.nan),
            "local_distance": safe_info(info, "local_distance", np.nan),
            "final_distance": safe_info(info, "final_distance", np.nan),
            "spark_visible_dbg": safe_info(info, "spark_visible_dbg", 0.0),
            "spark_response_active_dbg": safe_info(info, "spark_response_active_dbg", 0.0),
            "terminated_reason": safe_info(info, "terminated_reason", ""),
        }

    def set_env(self, env: Monitor) -> None:
        if not self.enabled:
            return
        base = unwrap_env(env)
        self._vehicle_name = str(getattr(base, "vehicle_name", ""))
        self._ip_address = str(getattr(base, "ip_address", "127.0.0.1"))
        if self.video_async:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._async_loop,
                    name="demo-video-recorder",
                    daemon=True,
                )
                self._thread.start()
        else:
            self._sync_client = getattr(base, "client", None)

    def _connect_thread_client_if_needed(self) -> bool:
        if self._thread_client is not None:
            return True
        try:
            client = airsim.MultirotorClient(ip=self._ip_address)
            client.confirmConnection()
            self._thread_client = client
            return True
        except Exception:
            self._thread_client = None
            return False

    def _grab_frame(self, client: airsim.MultirotorClient, vehicle_name: str) -> Optional[np.ndarray]:
        if not self.enabled or self._writer is None or client is None:
            return None

        try:
            responses = client.simGetImages(
                [
                    airsim.ImageRequest(
                        self.camera_name,
                        airsim.ImageType.Scene,
                        pixels_as_float=False,
                        compress=False,
                    )
                ],
                vehicle_name=vehicle_name,
            )
        except Exception:
            return None

        if not responses:
            return None
        resp = responses[0]
        if resp.width <= 0 or resp.height <= 0 or len(resp.image_data_uint8) == 0:
            return None

        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img.size != int(resp.width * resp.height * 3):
            return None
        frame = img.reshape(resp.height, resp.width, 3)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def _build_overlay_lines(self, phase: str, step: int, info: Dict[str, Any]) -> List[str]:
        wp_idx = extract_wp_idx(info)
        route_progress = extract_route_progress(info)
        local_distance = as_float(safe_info(info, "local_distance", np.nan), np.nan)
        final_distance = as_float(safe_info(info, "final_distance", np.nan), np.nan)
        spark_visible = as_float(safe_info(info, "spark_visible_dbg", 0.0), 0.0)
        spark_response = as_float(safe_info(info, "spark_response_active_dbg", 0.0), 0.0)
        term_reason = str(safe_info(info, "terminated_reason", ""))

        return [
            f"phase={phase}",
            f"step={step}",
            f"wp_idx={wp_idx}",
            f"route_progress_norm={route_progress:.3f}" if not np.isnan(route_progress) else "route_progress_norm=nan",
            f"local_distance={local_distance:.3f}" if not np.isnan(local_distance) else "local_distance=nan",
            f"final_distance={final_distance:.3f}" if not np.isnan(final_distance) else "final_distance=nan",
            f"spark_visible={spark_visible:.0f} spark_response={spark_response:.0f}",
            f"terminated_reason={term_reason}",
        ]

    def _draw_overlay(self, frame: np.ndarray, phase: str, step: int, info: Dict[str, Any]) -> None:
        if not self.show_overlay:
            return
        y = 28
        for line in self._build_overlay_lines(phase=phase, step=step, info=info):
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 26

    def _async_loop(self) -> None:
        target_period = 1.0 / max(float(self.fps), 1.0)
        while not self._stop_event.is_set():
            with self._lock:
                req_step = int(self._last_requested_step)
                req_phase = self._latest_phase
                req_info = dict(self._latest_overlay_info)
                vehicle_name = self._vehicle_name
                last_written = int(self._last_written_step)
                last_ts = float(self._last_frame_write_ts)

            if req_step < 0 or req_step == last_written:
                time.sleep(0.002)
                continue

            if req_step % self.capture_every_n_steps != 0:
                with self._lock:
                    self._last_written_step = req_step
                continue

            now_ts = time.perf_counter()
            sleep_left = target_period - (now_ts - last_ts)
            if sleep_left > 0.0:
                time.sleep(sleep_left)

            if not self._connect_thread_client_if_needed():
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            frame = self._grab_frame(self._thread_client, vehicle_name)
            if frame is not None:
                self._draw_overlay(frame, req_phase, req_step, req_info)
                self._writer.write(frame)
                with self._lock:
                    self._last_frame_bgr = frame.copy()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._last_written_step = req_step
                self._last_video_dt_ms = float(dt_ms)
                self._last_frame_write_ts = time.perf_counter()



    def write_frame(self, phase: str, step: int, info: Dict[str, Any]) -> float:
        if not self.enabled or self._writer is None:
            return 0.0

        if int(step) % self.capture_every_n_steps != 0:
            return 0.0

        if self.video_async:
            with self._lock:
                self._latest_phase = str(phase)
                self._latest_step = int(step)
                self._latest_overlay_info = self._extract_overlay_info(info)
                self._last_requested_step = int(step)
            return 0.0

        if self._sync_client is None:
            return 0.0

        t0 = time.perf_counter()
        frame = self._grab_frame(self._sync_client, self._vehicle_name)
        if frame is None:
            return 0.0

        self._draw_overlay(frame, phase, step, info)
        self._writer.write(frame)
        with self._lock:
            self._last_frame_bgr = frame.copy()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_video_dt_ms = float(dt_ms)
        return float(dt_ms)

    def get_last_video_dt_ms(self) -> float:
        with self._lock:
            return float(self._last_video_dt_ms)

    def get_last_frame_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last_frame_bgr is None:
                return None
            return self._last_frame_bgr.copy()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        with self._lock:
            self._last_frame_bgr = None


def maybe_log_light_step(
    logger: LightLogger,
    episode: int,
    phase: str,
    step: int,
    reward: float,
    episode_reward: float,
    info: Dict[str, Any],
    env: Optional[Monitor] = None,
    loop_dt_ms: float = np.nan,
    env_step_dt_ms: float = np.nan,
    model_predict_dt_ms: float = np.nan,
    video_dt_ms: float = np.nan,
) -> None:
    bird_metrics = extract_bird_light_metrics(info)
    logger.write(
        {
            "episode": int(episode),
            "phase": str(phase),
            "step": int(step),
            "loop_dt_ms": float(loop_dt_ms),
            "env_step_dt_ms": float(env_step_dt_ms),
            "model_predict_dt_ms": float(model_predict_dt_ms),
            "video_dt_ms": float(video_dt_ms),
            "reward": float(reward),
            "episode_reward": float(episode_reward),
            "wp_idx": int(extract_wp_idx(info)),
            "point_mode": extract_point_mode(info),
            "local_distance": as_float(safe_info_deep(info, "local_distance", np.nan), np.nan),
            "final_distance": as_float(safe_info_deep(info, "final_distance", np.nan), np.nan),
            "route_progress_norm": extract_route_progress(info, env=env),
            "visual_gate_required_dbg": as_float(safe_info_deep(info, "visual_gate_required_dbg", 0.0), 0.0),
            "gate_blocked_dbg": as_float(safe_info_deep(info, "gate_blocked_dbg", 0.0), 0.0),
            "target_yaw_error_rad": as_float(safe_info_deep(info, "target_yaw_error_rad", np.nan), np.nan),
            "planner_replan_step_dbg": as_float(safe_info_deep(info, "planner_replan_step_dbg", 0.0), 0.0),
            "local_path_rebuilt_this_step_dbg": as_float(safe_info_deep(info, "local_path_rebuilt_this_step_dbg", 0.0), 0.0),
            "local_target_changed_this_step_dbg": as_float(safe_info_deep(info, "local_target_changed_this_step_dbg", 0.0), 0.0),
            "center_error": as_float(safe_info_deep(info, "center_error", np.nan), np.nan),
            "visible": as_float(safe_info_deep(info, "visible", np.nan), np.nan),
            "conf": as_float(safe_info_deep(info, "conf", np.nan), np.nan),
            "terminated_reason": str(safe_info_deep(info, "terminated_reason", "")),
            "cable_safety_active": as_float(safe_info_deep(info, "cable_safety_active", 0.0), 0.0),
            "bird_visible_dbg": float(bird_metrics["bird_visible_dbg"]),
            "bird_det_count_dbg": float(bird_metrics["bird_det_count_dbg"]),
            "bird_max_conf_dbg": float(bird_metrics["bird_max_conf_dbg"]),
            "bird_center_dx_dbg": float(bird_metrics["bird_center_dx_dbg"]),
            "bird_center_dy_dbg": float(bird_metrics["bird_center_dy_dbg"]),
            "bird_area_dbg": float(bird_metrics["bird_area_dbg"]),
            "bird_hazard_active_dbg": float(bird_metrics["bird_hazard_active_dbg"]),
            "bird_response_active_dbg": float(bird_metrics["bird_response_active_dbg"]),
            "visual_hazard_active_dbg": float(bird_metrics["visual_hazard_active_dbg"]),
            "visual_hazard_score_dbg": float(bird_metrics["visual_hazard_score_dbg"]),
            "visual_hazard_confirm_steps_dbg": float(bird_metrics["visual_hazard_confirm_steps_dbg"]),
            "visual_hazard_clear_counter_dbg": float(bird_metrics["visual_hazard_clear_counter_dbg"]),
            "safety_visible_dbg": float(bird_metrics["safety_visible_dbg"]),
            "safety_conf_dbg": float(bird_metrics["safety_conf_dbg"]),
            "safety_area_dbg": float(bird_metrics["safety_area_dbg"]),
            "safety_dx_dbg": float(bird_metrics["safety_dx_dbg"]),
            "safety_hazard_score_dbg": float(bird_metrics["safety_hazard_score_dbg"]),
            "spark_visible_dbg": as_float(safe_info_deep(info, "spark_visible_dbg", 0.0), 0.0),
            "spark_det_count_dbg": as_float(safe_info_deep(info, "spark_det_count_dbg", 0.0), 0.0),
            "spark_max_conf_dbg": as_float(safe_info_deep(info, "spark_max_conf_dbg", 0.0), 0.0),
            "spark_event_score_dbg": as_float(safe_info_deep(info, "spark_event_score_dbg", 0.0), 0.0),
            "spark_response_active_dbg": as_float(safe_info_deep(info, "spark_response_active_dbg", 0.0), 0.0),
            "spark_override_steps_left_dbg": as_float(safe_info_deep(info, "spark_override_steps_left_dbg", 0.0), 0.0),
            "spark_action_override_changed_dbg": as_float(safe_info_deep(info, "spark_action_override_changed_dbg", 0.0), 0.0),
            "spark_camera_id_dbg": as_float(safe_info_deep(info, "spark_camera_id_dbg", np.nan), np.nan),
            "spark_iou_threshold_dbg": as_float(safe_info_deep(info, "spark_iou_threshold_dbg", np.nan), np.nan),
            "spark_infer_count_dbg": as_float(safe_info_deep(info, "spark_infer_count_dbg", 0.0), 0.0),
            "spark_infer_interval_steps_dbg": as_float(safe_info_deep(info, "spark_infer_interval_steps_dbg", np.nan), np.nan),
            "spark_infer_time_ms_dbg": as_float(safe_info_deep(info, "spark_infer_time_ms_dbg", np.nan), np.nan),
        }
    )


def maybe_save_spark_snapshot(
    snapshot_saver: Optional[SparkSnapshotSaver],
    recorder: VideoRecorder,
    phase: str,
    env_step: int,
    global_step: int,
    info: Dict[str, Any],
) -> bool:
    if snapshot_saver is None:
        return False
    frame = recorder.get_last_frame_copy()
    return bool(
        snapshot_saver.maybe_save(
            frame_bgr=frame,
            phase=str(phase),
            env_step=int(env_step),
            global_step=int(global_step),
            info=info,
        )
    )


def maybe_save_bird_snapshot(
    snapshot_saver: Optional[BirdSnapshotSaver],
    recorder: VideoRecorder,
    phase: str,
    env_step: int,
    global_step: int,
    info: Dict[str, Any],
) -> bool:
    if snapshot_saver is None:
        return False
    frame = recorder.get_last_frame_copy()
    return bool(
        snapshot_saver.maybe_save(
            frame_bgr=frame,
            phase=str(phase),
            env_step=int(env_step),
            global_step=int(global_step),
            info=info,
        )
    )


def run_nav_phase(
    model: PPO,
    env: Monitor,
    recorder: VideoRecorder,
    logger: LightLogger,
    args: argparse.Namespace,
    episode_idx: int,
    snapshot_saver: Optional[BirdSnapshotSaver] = None,
    global_step_offset: int = 0,
) -> Dict[str, Any]:
    obs, info = reset_env(env)
    base_env = unwrap_env(env)
    step_duration_ms = float(getattr(base_env, "step_duration", 0.0)) * 1000.0
    step_count = 0
    episode_reward = 0.0
    collisions = 0
    nav_success = False
    last_route_progress = extract_route_progress(info, env=env)
    last_final_distance_raw = as_float(safe_info(info, "final_distance_raw", safe_info(info, "final_distance", np.inf)), np.inf)
    term_reason = str(safe_info(info, "terminated_reason", ""))
    loop_dt_samples: List[float] = []
    env_step_dt_samples: List[float] = []
    model_predict_dt_samples: List[float] = []
    video_dt_samples: List[float] = []

    while step_count < int(args.max_nav_steps):
        loop_t0 = time.perf_counter()
        predict_t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        model_predict_dt_ms = (time.perf_counter() - predict_t0) * 1000.0
        env_step_t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = step_env(env, action)
        env_step_dt_ms = (time.perf_counter() - env_step_t0) * 1000.0
        step_count += 1
        episode_reward += float(reward)

        route_progress = extract_route_progress(info, env=env)
        final_distance_raw = as_float(safe_info(info, "final_distance_raw", safe_info(info, "final_distance", np.inf)), np.inf)
        goal_reached = bool(safe_info(info, "goal_reached", False))
        term_reason = str(safe_info(info, "terminated_reason", term_reason))

        collision_penalty = extract_collision_penalty(info)
        if collision_penalty < 0.0:
            collisions += 1

        last_route_progress = route_progress
        last_final_distance_raw = final_distance_raw

        video_dt_ms = float(recorder.write_frame(phase="NAV", step=step_count, info=info))
        maybe_save_bird_snapshot(
            snapshot_saver=snapshot_saver,
            recorder=recorder,
            phase="NAV",
            env_step=step_count,
            global_step=int(global_step_offset) + int(step_count),
            info=info,
        )
        loop_dt_ms = (time.perf_counter() - loop_t0) * 1000.0
        loop_dt_samples.append(float(loop_dt_ms))
        env_step_dt_samples.append(float(env_step_dt_ms))
        model_predict_dt_samples.append(float(model_predict_dt_ms))
        video_dt_samples.append(float(video_dt_ms))
        if video_dt_ms > 50.0 or (step_duration_ms > 0.0 and video_dt_ms > step_duration_ms):
            print(
                f"[VIDEO][WARN][NAV] step={step_count} video_dt_ms={video_dt_ms:.1f} "
                f"env_step_dt_ms={env_step_dt_ms:.1f} loop_dt_ms={loop_dt_ms:.1f}"
            )
        maybe_log_light_step(
            logger,
            episode_idx,
            "NAV",
            step_count,
            reward,
            episode_reward,
            info,
            env=env,
            loop_dt_ms=float(loop_dt_ms),
            env_step_dt_ms=float(env_step_dt_ms),
            model_predict_dt_ms=float(model_predict_dt_ms),
            video_dt_ms=float(video_dt_ms),
        )

        if goal_reached:
            nav_success = True
            if not term_reason:
                term_reason = "goal_reached"
            break

        if final_distance_raw < float(args.nav_final_distance_threshold):
            nav_success = True
            if not term_reason:
                term_reason = "nav_final_distance_threshold"
            break

        if not np.isnan(route_progress) and route_progress >= float(args.switch_route_progress_threshold):
            nav_success = True
            if not term_reason:
                term_reason = "switch_route_progress_threshold"
            break

        if terminated or truncated:
            break

    if step_count >= int(args.max_nav_steps) and not term_reason:
        term_reason = "max_nav_steps"

    loop_mean, loop_max = _timing_mean_max(loop_dt_samples)
    env_step_mean, env_step_max = _timing_mean_max(env_step_dt_samples)
    pred_mean, pred_max = _timing_mean_max(model_predict_dt_samples)
    video_mean, video_max = _timing_mean_max(video_dt_samples)

    return {
        "success": bool(nav_success),
        "steps": int(step_count),
        "episode_reward": float(episode_reward),
        "route_progress_norm": float(last_route_progress) if not np.isnan(last_route_progress) else np.nan,
        "final_distance_raw": float(last_final_distance_raw),
        "terminated_reason": term_reason,
        "collisions": int(collisions),
        "timing": {
            "loop_dt_ms_mean": loop_mean,
            "loop_dt_ms_max": loop_max,
            "env_step_dt_ms_mean": env_step_mean,
            "env_step_dt_ms_max": env_step_max,
            "model_predict_dt_ms_mean": pred_mean,
            "model_predict_dt_ms_max": pred_max,
            "video_dt_ms_mean": video_mean,
            "video_dt_ms_max": video_max,
        },
    }


def run_inspect_phase(
    model: PPO,
    env: Monitor,
    recorder: VideoRecorder,
    logger: LightLogger,
    args: argparse.Namespace,
    episode_idx: int,
    snapshot_saver: Optional[SparkSnapshotSaver] = None,
    global_step_offset: int = 0,
) -> Dict[str, Any]:
    obs, info = reset_env(env)
    base_env = unwrap_env(env)
    step_duration_ms = float(getattr(base_env, "step_duration", 0.0)) * 1000.0
    step_count = 0
    episode_reward = 0.0
    collisions = 0
    inspect_success = False
    term_reason = str(safe_info_deep(info, "terminated_reason", ""))
    last_route_progress = extract_route_progress(info, env=env)
    loop_dt_samples: List[float] = []
    env_step_dt_samples: List[float] = []
    model_predict_dt_samples: List[float] = []
    video_dt_samples: List[float] = []

    while step_count < int(args.max_inspect_steps):
        loop_t0 = time.perf_counter()
        predict_t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        model_predict_dt_ms = (time.perf_counter() - predict_t0) * 1000.0
        env_step_t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = step_env(env, action)
        env_step_dt_ms = (time.perf_counter() - env_step_t0) * 1000.0
        step_count += 1
        episode_reward += float(reward)

        route_progress = extract_route_progress(info, env=env)
        goal_reached = bool(safe_info_deep(info, "goal_reached", False))
        term_reason = str(safe_info_deep(info, "terminated_reason", term_reason))

        collision_penalty = extract_collision_penalty(info)
        if collision_penalty < 0.0:
            collisions += 1

        last_route_progress = route_progress

        video_dt_ms = float(recorder.write_frame(phase="INSPECT", step=step_count, info=info))
        maybe_save_spark_snapshot(
            snapshot_saver=snapshot_saver,
            recorder=recorder,
            phase="INSPECT",
            env_step=step_count,
            global_step=int(global_step_offset) + int(step_count),
            info=info,
        )
        loop_dt_ms = (time.perf_counter() - loop_t0) * 1000.0
        loop_dt_samples.append(float(loop_dt_ms))
        env_step_dt_samples.append(float(env_step_dt_ms))
        model_predict_dt_samples.append(float(model_predict_dt_ms))
        video_dt_samples.append(float(video_dt_ms))
        if video_dt_ms > 50.0 or (step_duration_ms > 0.0 and video_dt_ms > step_duration_ms):
            print(
                f"[VIDEO][WARN][INSPECT] step={step_count} video_dt_ms={video_dt_ms:.1f} "
                f"env_step_dt_ms={env_step_dt_ms:.1f} loop_dt_ms={loop_dt_ms:.1f}"
            )
        maybe_log_light_step(
            logger,
            episode_idx,
            "INSPECT",
            step_count,
            reward,
            episode_reward,
            info,
            env=env,
            loop_dt_ms=float(loop_dt_ms),
            env_step_dt_ms=float(env_step_dt_ms),
            model_predict_dt_ms=float(model_predict_dt_ms),
            video_dt_ms=float(video_dt_ms),
        )

        if terminated or truncated:
            if goal_reached or term_reason in {"goal_reached", "success"}:
                inspect_success = True
            break

    if step_count >= int(args.max_inspect_steps) and not term_reason:
        term_reason = "max_inspect_steps"

    loop_mean, loop_max = _timing_mean_max(loop_dt_samples)
    env_step_mean, env_step_max = _timing_mean_max(env_step_dt_samples)
    pred_mean, pred_max = _timing_mean_max(model_predict_dt_samples)
    video_mean, video_max = _timing_mean_max(video_dt_samples)

    return {
        "success": bool(inspect_success),
        "steps": int(step_count),
        "episode_reward": float(episode_reward),
        "route_progress_norm": float(last_route_progress) if not np.isnan(last_route_progress) else np.nan,
        "terminated_reason": term_reason,
        "collisions": int(collisions),
        "timing": {
            "loop_dt_ms_mean": loop_mean,
            "loop_dt_ms_max": loop_max,
            "env_step_dt_ms_mean": env_step_mean,
            "env_step_dt_ms_max": env_step_max,
            "model_predict_dt_ms_mean": pred_mean,
            "model_predict_dt_ms_max": pred_max,
            "video_dt_ms_mean": video_mean,
            "video_dt_ms_max": video_max,
        },
    }


def _print_phase_timing_summary(all_results: List[Dict[str, Any]], phase_key: str, phase_label: str) -> None:
    metric_keys = (
        ("loop_dt_ms", "loop_dt_ms_mean", "loop_dt_ms_max"),
        ("env_step_dt_ms", "env_step_dt_ms_mean", "env_step_dt_ms_max"),
        ("model_predict_dt_ms", "model_predict_dt_ms_mean", "model_predict_dt_ms_max"),
        ("video_dt_ms", "video_dt_ms_mean", "video_dt_ms_max"),
    )

    for metric_name, mean_key, max_key in metric_keys:
        mean_values: List[float] = []
        max_values: List[float] = []

        for result in all_results:
            phase_result = result.get(phase_key, {})
            timing = phase_result.get("timing", {}) if isinstance(phase_result, dict) else {}
            mean_val = as_float(timing.get(mean_key, np.nan), np.nan)
            max_val = as_float(timing.get(max_key, np.nan), np.nan)
            if not np.isnan(mean_val):
                mean_values.append(float(mean_val))
            if not np.isnan(max_val):
                max_values.append(float(max_val))

        if not mean_values and not max_values:
            continue

        mean_of_means = float(np.nanmean(np.asarray(mean_values, dtype=np.float64))) if mean_values else np.nan
        max_of_max = float(np.nanmax(np.asarray(max_values, dtype=np.float64))) if max_values else np.nan
        print(f"[{phase_label}_TIMING] {metric_name} mean={mean_of_means:.3f} max={max_of_max:.3f}")

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.nav_model):
        raise FileNotFoundError(f"nav model not found: {args.nav_model}")
    if not os.path.exists(args.inspect_model):
        raise FileNotFoundError(f"inspect model not found: {args.inspect_model}")
    if not os.path.exists(args.pairs_json):
        raise FileNotFoundError(f"pairs json not found: {args.pairs_json}")

    nav_model = PPO.load(args.nav_model, device="cpu")
    inspect_model = PPO.load(args.inspect_model, device="cpu")

    report_name = str(args.report_name).strip()
    if report_name == "":
        report_name = f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    run_report_dir = os.path.join(str(args.report_dir), report_name)

    recorder = VideoRecorder(
        enabled=bool(args.record_video),
        path=args.video_path,
        fps=int(args.video_fps),
        width=int(args.video_width),
        height=int(args.video_height),
        camera_name=str(args.video_camera_name),
        show_overlay=bool(args.show_overlay),
        capture_every_n_steps=int(args.video_capture_every_n_steps),
        video_async=bool(args.video_async),
    )
    logger = LightLogger(enabled=bool(args.write_light_log), path=args.light_log_path)
    snapshot_saver = SparkSnapshotSaver(
        enabled=bool(args.save_spark_snapshots),
        report_dir=run_report_dir,
        snapshot_dirname=str(args.spark_snapshot_dirname),
        max_count=int(args.spark_snapshot_max_count),
        min_gap_steps=int(args.spark_snapshot_min_gap_steps),
    )
    bird_snapshot_saver = BirdSnapshotSaver(
        enabled=bool(args.save_bird_snapshots),
        report_dir=run_report_dir,
        snapshot_dirname=str(args.bird_snapshot_dirname),
        max_count=int(args.bird_snapshot_max_count),
        min_gap_steps=int(args.bird_snapshot_min_gap_steps),
    )

    all_results: List[Dict[str, Any]] = []
    global_step_cursor = 0

    try:
        for episode_idx in range(1, int(args.episodes) + 1):
            print(f"\n=== Episode {episode_idx}/{args.episodes}: NAV phase ===")
            nav_env = make_nav_env(args)
            try:
                print_nav_cfg(nav_env, args)
                check_model_env_compat(nav_model, nav_env, "NAV")
                recorder.set_env(nav_env)
                nav_result = run_nav_phase(
                    nav_model,
                    nav_env,
                    recorder,
                    logger,
                    args,
                    episode_idx,
                    snapshot_saver=bird_snapshot_saver,
                    global_step_offset=int(global_step_cursor),
                )
            finally:
                nav_env.close()
            global_step_cursor += int(nav_result["steps"])

            print(
                f"[NAV] success={nav_result['success']} steps={nav_result['steps']} "
                f"route_progress={nav_result['route_progress_norm']:.3f} "
                f"final_distance_raw={nav_result['final_distance_raw']:.3f} "
                f"collisions={nav_result['collisions']} reason={nav_result['terminated_reason']}"
            )

            # TODO: For seamless switch add no-reset inspect start / sync_from_current_drone_state.
            print(f"=== Episode {episode_idx}/{args.episodes}: INSPECT phase ===")
            inspect_env = make_inspect_env(args)
            try:
                check_model_env_compat(inspect_model, inspect_env, "INSPECT")
                recorder.set_env(inspect_env)
                inspect_result = run_inspect_phase(
                    inspect_model,
                    inspect_env,
                    recorder,
                    logger,
                    args,
                    episode_idx,
                    snapshot_saver=snapshot_saver,
                    global_step_offset=int(global_step_cursor),
                )
            finally:
                inspect_env.close()
            global_step_cursor += int(inspect_result["steps"])

            print(
                f"[INSPECT] success={inspect_result['success']} steps={inspect_result['steps']} "
                f"route_progress={inspect_result['route_progress_norm']:.3f} "
                f"collisions={inspect_result['collisions']} reason={inspect_result['terminated_reason']}"
            )

            all_results.append(
                {
                    "episode": episode_idx,
                    "nav": nav_result,
                    "inspect": inspect_result,
                }
            )
    finally:
        recorder.close()
        snapshot_saver.finalize()
        bird_snapshot_saver.finalize()

    nav_success_count = sum(1 for r in all_results if bool(r["nav"]["success"]))
    inspect_success_count = sum(1 for r in all_results if bool(r["inspect"]["success"]))
    nav_steps_total = sum(int(r["nav"]["steps"]) for r in all_results)
    inspect_steps_total = sum(int(r["inspect"]["steps"]) for r in all_results)
    nav_collisions_total = sum(int(r["nav"]["collisions"]) for r in all_results)
    inspect_collisions_total = sum(int(r["inspect"]["collisions"]) for r in all_results)

    final_route_progress = np.nan
    if all_results:
        final_route_progress = all_results[-1]["inspect"]["route_progress_norm"]

    print("\n=== DEMO SUMMARY ===")
    print(f"episodes: {len(all_results)}")
    print(f"nav success: {nav_success_count}/{len(all_results)}")
    print(f"inspect success: {inspect_success_count}/{len(all_results)}")
    print(f"nav steps total: {nav_steps_total}")
    print(f"inspect steps total: {inspect_steps_total}")
    print(f"nav collisions: {nav_collisions_total}")
    print(f"inspect collisions: {inspect_collisions_total}")
    print(f"final route_progress_norm: {final_route_progress}")
    _print_phase_timing_summary(all_results, phase_key="nav", phase_label="NAV")
    _print_phase_timing_summary(all_results, phase_key="inspect", phase_label="INSPECT")
    print(f"video path: {os.path.abspath(args.video_path) if args.record_video else 'disabled'}")
    print(f"light log path: {os.path.abspath(args.light_log_path) if args.write_light_log else 'disabled'}")
    if bool(args.save_spark_snapshots):
        print(f"spark snapshots: {snapshot_saver.count} (index: {os.path.abspath(snapshot_saver.index_path)})")
    if bool(args.save_bird_snapshots):
        print(f"bird snapshots: {bird_snapshot_saver.count} (index: {os.path.abspath(bird_snapshot_saver.index_path)})")

    if bool(args.auto_report):
        if bool(args.write_light_log):
            saved_dir = build_demo_report(
                csv_path=str(args.light_log_path),
                out_dir=run_report_dir,
                dpi=int(args.report_plots_dpi),
            )
            if saved_dir is not None:
                print(f"[REPORT] saved to: {os.path.abspath(saved_dir)}")
        else:
            print("[REPORT] skipped: --write-light-log is disabled")


if __name__ == "__main__":
    main()
