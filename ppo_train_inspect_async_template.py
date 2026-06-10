import os
import time
import numpy as np
import torch as th
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from env_ppo_inspect_yolo_async import AirSimDroneInspectEnv
from ppo_train_nav import (
    EpisodeCsvLoggerCallback,
    PPOWithTrainCsv,
    StepRewardCsvLoggerCallback,
)
def load_pairs_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        pairs.append({
            "route_point": np.asarray(item["route_point"], dtype=np.float32),
            "target_point": np.asarray(item["target_point"], dtype=np.float32),
            "point_mode": item.get("point_mode", "inspect"),
            "visual_gate_required": bool(item.get("visual_gate_required", False)),
            "visual_gate_visible_threshold": float(item.get("visual_gate_visible_threshold", 0.20)),
            "visual_gate_center_threshold": float(item.get("visual_gate_center_threshold", 0.32)),
            "visual_gate_yaw_threshold": float(item.get("visual_gate_yaw_threshold", 0.65)),
            "visual_gate_hold_steps": int(item.get("visual_gate_hold_steps", 2)),
        })
    return pairs
START_POINTS = [
    [-27.861, 14.968, -32.906],
]
INSPECTION_PAIR_SETS = load_pairs_from_json("Lvl1.json")


YOLO_CLASS_MAP = {
    0: "defect_isolator",
    1: "no_detail",
    2: "arc_difect",
    3: "isolator",
}

TARGET_CLASS_IDS = {3}
TARGET_CLASS_NAMES = {"isolator"}

MODEL_DIR = "models/ppo_inspect_train9"
LOG_DIR = "logs/ppo_inspect_train9"
YOLO_MODEL_PATH = r"YOLO_isolator.pt"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
def seg_spark_freshness_provider(env, image, selected):

    if not getattr(env, "enable_visual_inspection", False):
        return np.zeros(16, dtype=np.float32)

    seg = getattr(env, "_last_seg_info", {}) or {}

    # ---------- freshness ----------
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
            age_sec = max(
                0.0,
                (env.current_step - last_step) * float(env.step_duration)
            )

    stale_thr = max(float(env.step_duration * env.yolo_inference_interval * 1.5), 0.12)
    age_norm = float(np.clip(age_sec / 0.50, 0.0, 1.0))
    stale_flag = 1.0 if age_sec > stale_thr else 0.0

    # ---------- seg block: оставляем как есть ----------
    seg_visible = float(seg.get("seg_visible", 0.0))
    seg_cable_ratio = float(seg.get("seg_cable_ratio", 0.0))
    seg_tower_ratio = float(seg.get("seg_tower_ratio", 0.0))
    seg_cable_center_dx = float(seg.get("seg_cable_center_dx", 0.0))
    seg_cable_center_dy = float(seg.get("seg_cable_center_dy", 0.0))
    seg_cable_angle_sin = float(seg.get("seg_cable_angle_sin", 0.0))
    seg_cable_angle_cos = float(seg.get("seg_cable_angle_cos", 1.0))
    seg_mask_stability = float(seg.get("seg_mask_stability", 0.0))

    # ---------- spark block: мягко зануляем ----------
    spark = getattr(env, "_last_spark_info", {}) or {}

    spark_soft_gain = 0.15

    # небольшой антишум: считаем spark значимым только если есть либо хорошая conf,
    # либо уже накопилась персистентность
    spark_conf = float(np.clip(spark.get("spark_max_conf", 0.0), 0.0, 1.0))
    spark_persistent_raw = float(np.clip(spark.get("spark_persistent", 0.0), 0.0, 1.0))
    spark_gate = 1.0 if (spark_conf >= 0.45 or spark_persistent_raw > 0.5) else 0.0

    spark_visible = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_visible", 0.0), 0.0, 1.0))
    spark_max_conf = spark_soft_gain * spark_gate * spark_conf
    spark_max_area = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_max_area", 0.0), 0.0, 1.0))
    spark_center_dx = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_center_dx", 0.0), -1.0, 1.0))
    spark_center_dy = spark_soft_gain * spark_gate * float(np.clip(spark.get("spark_center_dy", 0.0), -1.0, 1.0))
    spark_persistent = spark_soft_gain * spark_persistent_raw

    return np.array([
        # 0..7 seg
        seg_visible,
        seg_cable_ratio,
        seg_tower_ratio,
        seg_cable_center_dx,
        seg_cable_center_dy,
        seg_cable_angle_sin,
        seg_cable_angle_cos,
        seg_mask_stability,

        # 8..13 spark
        spark_visible,
        spark_max_conf,
        spark_max_area,
        spark_center_dx,
        spark_center_dy,
        spark_persistent,

        # 14..15 freshness
        age_norm,
        stale_flag,
    ], dtype=np.float32)


def infer_world_xy_limit(
    start_points=None,
    inspection_pairs=None,
    inspection_pair_sets=None,
    extra_margin: float = 20.0,
    min_limit: float = 80.0,
) -> float:
    pts = []

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

    if inspection_pair_sets is not None:
        for pair_set in inspection_pair_sets:
            for pair in pair_set:
                rp = np.asarray(pair["route_point"], dtype=np.float32).reshape(-1)
                tp = np.asarray(pair["target_point"], dtype=np.float32).reshape(-1)
                if rp.shape[0] >= 2:
                    pts.append(rp[:2])
                if tp.shape[0] >= 2:
                    pts.append(tp[:2])

    if not pts:
        return float(min_limit)

    pts = np.asarray(pts, dtype=np.float32)
    max_abs_xy = float(np.max(np.abs(pts)))
    return float(max(max_abs_xy + extra_margin, min_limit))
WORLD_XY_LIMIT_AUTO = infer_world_xy_limit(
    start_points=START_POINTS,
    inspection_pairs=INSPECTION_PAIR_SETS,
    inspection_pair_sets=None,
    extra_margin=20.0,
    min_limit=80.0,
)

print("AUTO world_xy_limit =", WORLD_XY_LIMIT_AUTO)
def make_env():
    env = AirSimDroneInspectEnv(
        vehicle_name="SimpleFlight",
        start_points=START_POINTS,
        inspection_pairs=INSPECTION_PAIR_SETS,
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
        yolo_model_path=YOLO_MODEL_PATH,
        yolo_device="cuda",
        yolo_input_size=256,
        yolo_conf_threshold=0.25,
        yolo_iou_threshold=0.5,
        yolo_target_class_ids=TARGET_CLASS_IDS,
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
        debug_draw_detections= False,
        debug_show_window=False,
        debug_save_video=False,
        debug_print_yolo=False,
        debug_print_obs=False,
        debug_print_seg=False,
        debug_video_path=r"logs/inspect_debug/vision_dron.mp4",
        debug_draw_every_n_steps=1,
        enable_heavy_diagnostics=False,
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
        spark_override_duration_steps=30,
        spark_response_backoff_ratio=0.65,
        spark_response_lateral_ratio=0.45,
        spark_hazard_xy_scale=0.72,
        spark_hazard_yaw_scale=0.80,
        no_progress_max_steps=120,
        world_xy_limit=WORLD_XY_LIMIT_AUTO,
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
        cable_soft_distance_m = 4.0,
        cable_hard_distance_m = 1.0,
        cable_safety_enabled=True,
        waypoint_tolerance=0.4,
        waypoint_pass_tolerance=0.65,
        waypoint_pass_margin=0.08,
        waypoint_pass_increase_steps_req=2,


    )
    env.register_aux_feature_provider(seg_spark_freshness_provider)
    print("INSPECTION_PAIR_SETS LEN:", len(INSPECTION_PAIR_SETS))
    print("INSPECTION PAIRS:")
    for point_idx, p in enumerate(INSPECTION_PAIR_SETS):
        print(point_idx, p["route_point"], "->", p["target_point"])
    return Monitor(env)



class YoloRewardScaleCallback(BaseCallback):
    def __init__(self, step_freq=25_000, delta=0.1, max_scale=1.0, verbose=1):
        super().__init__(verbose)
        self.step_freq = int(step_freq)
        self.delta = float(delta)
        self.max_scale = float(max_scale)
        self._last_applied_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_applied_step) < self.step_freq:
            return True

        self._last_applied_step = self.num_timesteps

        current_vals = self.training_env.env_method("get_yolo_reward_scale")
        current = float(current_vals[0]) if current_vals else 0.0
        new_value = min(self.max_scale, current + self.delta)

        self.training_env.env_method("set_yolo_reward_scale", new_value)

        if self.verbose:
            print(f"[YOLO SCALE] timesteps={self.num_timesteps} scale: {current:.2f} -> {new_value:.2f}")

        return True

def main():
    env = DummyVecEnv([make_env])

    model = PPOWithTrainCsv(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cpu",
        train_metrics_csv_path=os.path.join(LOG_DIR, "train_metrics.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000 ,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="ppo_inspect_async",
    )

    callback = CallbackList([
        checkpoint_callback,
        EpisodeCsvLoggerCallback(os.path.join(LOG_DIR, "episode_metrics.csv")),
        StepRewardCsvLoggerCallback(os.path.join(LOG_DIR, "reward_trace.csv")),
    ])

    model.learn(
        total_timesteps=1024,
        callback=callback,
        progress_bar=False,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_inspect_async_final"))
    env.close()

if __name__ == "__main__":
    main()
