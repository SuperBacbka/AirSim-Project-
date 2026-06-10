import csv
import os
from typing import Optional
import json
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from env_ppo_nav import AirSimDroneEnv


class EpisodeCsvLoggerCallback(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.header_written = False

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.header_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_summary = info.get("episode_summary")
            if episode_summary is None:
                continue

            row = dict(episode_summary)
            row["num_timesteps"] = int(self.num_timesteps)

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(row)
        return True
def safe_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)

class StepRewardCsvLoggerCallback(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.header_written = False
        self.fieldnames = None
        self._fieldname_set = set()
        self._warned_new_keys = set()

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.header_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        if self.header_written:
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                if header:
                    self.fieldnames = list(header)
                    self._fieldname_set = set(self.fieldnames)
                else:
                    self.header_written = False
            except Exception as exc:
                print(f"[reward_trace schema] failed to read existing header: {exc}")
                self.header_written = False
                self.fieldnames = None
                self._fieldname_set = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            reward_info = info.get("reward_info")
            if reward_info is None:
                continue

            planner_info = info.get("planner_info", {})
            control_debug = info.get("control_debug", {})
            yolo_info = info.get("yolo_info", {})

            row = {
                "num_timesteps": int(self.num_timesteps),
                "step": int(info.get("step", -1)),
                "episode_reward": float(info.get("episode_reward", 0.0)),
                "final_distance": float(info.get("final_distance", np.nan)),
                "local_distance": float(info.get("local_distance", np.nan)),
                "terminated_reason": str(info.get("terminated_reason", "")),
                "done": int(bool(dones[idx])) if idx < len(dones) else 0,
            }

            for key, value in reward_info.items():
                row[key] = safe_csv_value(value)

            for key, value in yolo_info.items():
                if isinstance(value, str):
                    row[key] = value
                elif isinstance(value, (int, np.integer)):
                    row[key] = int(value)
                else:
                    row[key] = safe_csv_value(value)

            for key, value in planner_info.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        col = f"planner_{key}_{sub_key}"
                        if isinstance(sub_value, str):
                            row[col] = sub_value
                        elif isinstance(sub_value, (int, np.integer)):
                            row[col] = int(sub_value)
                        else:
                            row[col] = float(sub_value)
                elif isinstance(value, str):
                    row[f"planner_{key}"] = value
                elif isinstance(value, (int, np.integer)):
                    row[f"planner_{key}"] = int(value)
                else:
                    row[f"planner_{key}"] = float(value)

            for key, value in control_debug.items():
                if value is None:
                    row[key] = ""
                elif isinstance(value, str):
                    row[key] = value
                elif isinstance(value, (bool, np.bool_)):
                    row[key] = int(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    row[key] = safe_csv_value(value)
                else:
                    row[key] = str(value)

            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
                self._fieldname_set = set(self.fieldnames)
            else:
                new_keys = [k for k in row.keys() if k not in self._fieldname_set]
                if new_keys:
                    unseen = [k for k in new_keys if k not in self._warned_new_keys]
                    if unseen:
                        preview = ", ".join(unseen[:10])
                        suffix = " ..." if len(unseen) > 10 else ""
                        print(
                            f"[reward_trace schema] ignored {len(unseen)} new keys after header: "
                            f"{preview}{suffix}"
                        )
                        self._warned_new_keys.update(unseen)

            normalized_row = {field: row.get(field, "") for field in self.fieldnames}
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.fieldnames,
                    extrasaction="ignore",
                )
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(normalized_row)
        return True


class PPOWithTrainCsv(PPO):
    def __init__(self, *args, train_metrics_csv_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics_csv_path = train_metrics_csv_path
        self._train_header_written = False

    def _ensure_train_csv(self):
        if not self.train_metrics_csv_path:
            return
        os.makedirs(os.path.dirname(self.train_metrics_csv_path), exist_ok=True)
        self._train_header_written = (
            os.path.exists(self.train_metrics_csv_path)
            and os.path.getsize(self.train_metrics_csv_path) > 0
        )

    def train(self) -> None:
        super().train()
        if not self.train_metrics_csv_path:
            return

        self._ensure_train_csv()
        values = getattr(self.logger, "name_to_value", {})
        row = {
            "num_timesteps": int(self.num_timesteps),
            "n_updates": int(self._n_updates),
            "train/explained_variance": float(values.get("train/explained_variance", np.nan)),
            "train/value_loss": float(values.get("train/value_loss", np.nan)),
            "train/policy_loss": float(values.get("train/policy_gradient_loss", np.nan)),
            "train/policy_gradient_loss": float(values.get("train/policy_gradient_loss", np.nan)),
            "train/entropy_loss": float(values.get("train/entropy_loss", np.nan)),
            "train/loss": float(values.get("train/loss", np.nan)),
            "train/approx_kl": float(values.get("train/approx_kl", np.nan)),
            "train/clip_fraction": float(values.get("train/clip_fraction", np.nan)),
        }

        with open(self.train_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._train_header_written:
                writer.writeheader()
                self._train_header_written = True
            writer.writerow(row)


START_POINTS = [[-26.402, 37.196, -41.058], [19.358, 24.610, -34.158]]
def load_route_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data, dtype=np.float32)




ROUTE_SETS = load_route_from_json("train_route_ppo_nav.json")

LOG_ROOT = "logs/ppo_nav/train13_last"
MODEL_ROOT = "models/ppo_nav/train13_last"



def make_env():
    env = AirSimDroneEnv(
        vehicle_name="SimpleFlight",
        start_points=START_POINTS,
        route_sets=ROUTE_SETS,
        route_points=None,
        fixed_goal=None,
        random_goal=False,
        max_episode_steps=2200,
        step_duration=0.08,
        max_velocity=6.0,
        max_yaw_rate=45.0,
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
        planner_replan_interval=6,
        planner_lookahead_steps=1,
        planner_shift_distance=2.5,
        planner_shift_hold_steps=14,
        soft_avoidance_fail_steps=8,
        obstacle_memory_ttl_steps=40,
        use_depth_image=True,
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
        enable_nav_visual=True,

        enable_nav_seg_context=True,
        enable_nav_safety_detector= True,
        visual_hazard_replan_enabled=False,
        debug_print_visual_hazard=False,
        safety_conf_threshold=0.30,
        safety_detector_model_path="Bird.pt",
        safety_target_class_names={"bird"},
        infra_detector_model_path=None,
        action_smoothing_alpha=0.45,
        nav_use_deep_sort=True,
        cmd_filter_alpha_xy = 0.72,
        cmd_filter_alpha_z = 0.5,
        cmd_filter_alpha_yaw = 0.55,
        max_delta_vxy_cmd = 0.45,
        max_delta_vz_cmd = 0.18,
        max_delta_yaw_rate_cmd = 3.0,
        altitude_assist_mix = 0.40,
        inactive_penalty_weight = 0.0,
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
        control_command_duration=0.65,
    )
    return Monitor(env)


if __name__ == "__main__":
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    env = make_env()

    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    )

    model = PPOWithTrainCsv(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(LOG_ROOT, "tensorboard"),
        device="cpu",
        train_metrics_csv_path=os.path.join(LOG_ROOT, "train_metrics.csv"),
    )

    sb3_logger = configure(os.path.join(LOG_ROOT, "sb3"), ["stdout", "csv", "tensorboard"])
    model.set_logger(sb3_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path=os.path.join(MODEL_ROOT, "checkpoints"),
        name_prefix="ppo_nav",
    )

    callback = CallbackList([
        checkpoint_callback,
        EpisodeCsvLoggerCallback(os.path.join(LOG_ROOT, "episode_metrics.csv")),
        StepRewardCsvLoggerCallback(os.path.join(LOG_ROOT, "reward_trace.csv")),
    ])

    try:
        model.learn(total_timesteps=5120, callback=callback, progress_bar=False)
        model.save(os.path.join(MODEL_ROOT, "ppo_nav_final"))
    finally:
        env.close()
