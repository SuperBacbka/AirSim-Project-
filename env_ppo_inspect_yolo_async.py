from typing import Any, Callable, Dict, List, Optional
import threading
import time

import numpy as np
import airsim
from gymnasium import spaces

from env_ppo_nav import AirSimDroneEnv
import cv2

class AirSimDroneInspectEnv(AirSimDroneEnv):
    """
    Отдельная ветка под инспекцию.

    База берётся из навигационной среды, но меняются:
    - yaw-логика: дрон смотрит на объект инспекции, а не только на waypoint;
    - observation: добавляются признаки относительного положения к объекту инспекции;
    - reward: добавляются термы за удержание дистанции, видимость и центрирование цели;
    - termination: ослаблен no_progress/inactive, т.к. для инспекции допустимы медленные движения.

    YOLO-признаки ожидаются в том же формате, что и в nav-env:
    [visible, class_id_norm, conf, center_x, center_y, area, dx_from_center, dy_from_center]
    """

    def __init__(
            self,
            *args,
            inspection_target_position=None,
            desired_standoff_distance: float = 8.0,
            standoff_tolerance: float = 2.0,
            desired_bbox_area: float = 0.08,
            require_target_visibility: bool = False,
            lost_target_max_steps: int = 50,
            face_inspection_target: bool = True,
            inspect_xy_speed_limit: float = 3.0,
            inspect_xy_delta_limit: float = 0.40,
            inspect_xy_filter_alpha: float = 0.55,
            inspect_manual_yaw_mix: float = 0.35,

            enable_visual_inspection: bool = False,
            inspection_pair_sets=None,
            inspection_pairs=None,
            scene_camera_name: str = "0",
            spark_camera_name: Optional[str] = None,
            scene_image_type=airsim.ImageType.Scene,
            yolo_model_path: Optional[str] = None,
            yolo_device: Optional[str] = None,
            yolo_input_size: int = 320,
            yolo_conf_threshold: float = 0.25,
            yolo_iou_threshold: float = 0.45,
            spark_iou_threshold: float = 0.45,
            yolo_target_class_ids=None,
            yolo_target_class_names=None,
            yolo_prefer_central_target: bool = True,
            target_ema_alpha: float = 0.35,
            target_hold_steps: int = 6,
            target_min_area: float = 1e-4,
            aux_visual_feature_dim: int = 0,
            debug_draw_detections: bool = False,
            debug_show_window: bool = False,
            debug_save_video: bool = False,
            debug_print_yolo: bool = False,
            debug_print_obs: bool = False,
            enable_heavy_diagnostics: bool = False,
            debug_video_path: str = "debug_inspect.mp4",
            debug_draw_every_n_steps: int = 3,
            yolo_inference_interval: int = 1,
            run_external_debug_detectors: bool = False,
            async_visual_update: bool = True,
            async_camera_poll_interval: float = 0.03,
            async_inference_min_interval: float = 0.05,
            keep_last_async_frame: bool = True,
            yolo_reward_scale: float = 0.1,
            control_command_duration: Optional[float] = None,
            inspect_reset_lift_m: float = 2.5,
            control_stream_enabled: bool = True,
            control_stream_hz: float = 15.0,
            control_stream_stale_sec: float = 0.45,
            control_stream_command_duration: float = 0.18,

            **kwargs,
    ):
        # ===== Dedicated AirSim command streamer =====
        self.control_stream_enabled = bool(control_stream_enabled)
        self.control_stream_hz = float(control_stream_hz)
        self.control_stream_stale_sec = float(control_stream_stale_sec)
        self.control_stream_command_duration = float(control_stream_command_duration)
        self.waypoint_pass_tolerance = float(kwargs.pop("waypoint_pass_tolerance", 0.65))
        self.waypoint_pass_margin = float(kwargs.pop("waypoint_pass_margin", 0.08))
        self.waypoint_pass_increase_steps_req = int(kwargs.pop("waypoint_pass_increase_steps_req", 2))

        self._wp_pass_last_idx = -1
        self._wp_pass_best_dist = 1e9
        self._wp_pass_increase_steps = 0
        # ===== cable / wire visual safety =====
        self.cable_safety_enabled = bool(kwargs.pop("cable_safety_enabled", True))
        # ===== cable safety hysteresis =====
        self.cable_safety_hold_steps = int(kwargs.pop("cable_safety_hold_steps", 35))
        self._cable_hold_left = 0
        self._cable_hold_last_update_step = -1
        self._cable_hold_hard = False
        self._cable_hold_last_dx = 0.0
        self._cable_hold_last_dy = 0.0
        self._cable_hold_last_ratio = 0.0
        self._last_cable_safety_state = None
        self._last_cable_safety_state_step = -1
        # cable_ratio — доля маски кабеля в кадре.
        # Чем больше, тем ближе/опаснее кабель.
        self.cable_soft_ratio = float(kwargs.pop("cable_soft_ratio", 0.0015))
        self.cable_hard_ratio = float(kwargs.pop("cable_hard_ratio", 0.0060))

        # Если кабель близко к центру кадра — считаем, что дрон летит в опасный коридор.
        self.cable_center_soft = float(kwargs.pop("cable_center_soft", 0.42))
        self.cable_center_hard = float(kwargs.pop("cable_center_hard", 0.25))

        # Насколько ограничивать движение при опасном кабеле.
        self.cable_soft_xy_scale = float(kwargs.pop("cable_soft_xy_scale", 0.55))
        self.cable_hard_xy_scale = float(kwargs.pop("cable_hard_xy_scale", 0.18))
        self.cable_yaw_scale = float(kwargs.pop("cable_yaw_scale", 0.50))

        # Небольшой отталкивающий боковой импульс от кабеля.
        self.cable_away_push_speed = float(kwargs.pop("cable_away_push_speed", 0.35))

        self._control_stream_thread = None
        self._control_stream_stop_event = None
        self._control_stream_lock = threading.Lock()
        self._control_stream_client = None

        self._latest_stream_cmd = np.zeros(4, dtype=np.float32)
        self._latest_stream_cmd_time = 0.0
        self._last_policy_raw_action = np.zeros(4, dtype=np.float32)
        self._last_action_apply_diag: Dict[str, float] = {}
        self._last_spark_trace_diag: Dict[str, float] = {
            "spark_override_active_dbg": 0.0,
            "spark_action_override_changed_dbg": 0.0,
            "spark_action_to_apply_0_dbg": 0.0,
            "spark_action_to_apply_1_dbg": 0.0,
            "spark_action_to_apply_2_dbg": 0.0,
            "spark_action_to_apply_3_dbg": 0.0,
            "spark_action_after_override_0_dbg": 0.0,
            "spark_action_after_override_1_dbg": 0.0,
            "spark_action_after_override_2_dbg": 0.0,
            "spark_action_after_override_3_dbg": 0.0,
        }
        self._prev_progress_wp_idx = -1
        self._prev_progress_local_target = None
        self._prev_progress_local_path_index = -1
        self._local_target_changed_this_step = False
        self._corridor_hard_raw_streak = 0
        self._corridor_near_collision_streak = 0
        self.enable_heavy_diagnostics = bool(enable_heavy_diagnostics)

        self.inspect_reset_lift_m = float(inspect_reset_lift_m)
        self.inspection_target_position = None
        self.active_inspection_target = None
        self.desired_standoff_distance = float(desired_standoff_distance)
        self.standoff_tolerance = float(standoff_tolerance)
        self.desired_bbox_area = float(desired_bbox_area)
        self.require_target_visibility = bool(require_target_visibility)
        self.lost_target_max_steps = int(lost_target_max_steps)
        self.face_inspection_target = bool(face_inspection_target)
        self.debug_print_reset = bool(kwargs.pop("debug_print_reset", False))
        self.lost_target_steps = 0
        self.visible_steps = 0
        self.inspect_extra_obs_dim = 5
        self.applied_cmd_obs_dim = 4
        self.run_external_debug_detectors = bool(run_external_debug_detectors)
        self.inspect_xy_speed_limit = float(inspect_xy_speed_limit)
        self.inspect_xy_delta_limit = float(inspect_xy_delta_limit)
        self.inspect_xy_filter_alpha = float(inspect_xy_filter_alpha)
        self.inspect_manual_yaw_mix = float(inspect_manual_yaw_mix)

        # ===== Visual inspection pipeline =====
        self.enable_visual_inspection = bool(enable_visual_inspection)
        self.scene_camera_name = scene_camera_name
        self.spark_camera_name = None if spark_camera_name is None else str(spark_camera_name)
        self.scene_image_type = scene_image_type
        self.yolo_model_path = yolo_model_path
        self.yolo_device = yolo_device
        self.yolo_input_size = int(yolo_input_size)
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_iou_threshold = float(yolo_iou_threshold)
        self.yolo_target_class_ids = None if yolo_target_class_ids is None else {int(x) for x in yolo_target_class_ids}
        self.yolo_target_class_names = None if yolo_target_class_names is None else {str(x).lower() for x in yolo_target_class_names}
        self.class_name_aliases = {"defect_isolator": "defect_isolator", "no_detail": "no_detail", "arc_difect": "arc_difect", "isolator": "isolator"}
        self.yolo_prefer_central_target = bool(yolo_prefer_central_target)
        self.track_max_missed = 12
        self.track_gate_px = 0.12
        self.track_area_alpha = 0.65
        self.track_pos_alpha = 0.75
        self.inspect_xy_far_dist = 2.5
        self.inspect_xy_cruise_speed = 1.15
        self.inspect_xy_far_speed_limit = 2.4
        self.inspect_xy_far_delta_limit = 0.70
        self.inspect_xy_route_assist_gain = 0.22
        self.track_route_gate_cx = 0.35
        self.track_route_gate_area_ratio = 6.0
        self.track_keep_max_missed = 12

        self.target_ema_alpha = float(target_ema_alpha)
        self.target_hold_steps = int(target_hold_steps)
        self.target_min_area = float(target_min_area)
        self.aux_visual_feature_dim = int(aux_visual_feature_dim)

        # ===== HYSTERESIS =====
        self.inspect_lock = False
        self.inspect_lock_steps = 0
        self.inspect_lock_min_steps = 15

        self.inspect_enter_threshold = 0.75
        self.inspect_exit_threshold = 0.45
        # ===== late-stage visual waypoint gate =====
        self.visual_gate_center_threshold = 0.20
        self.visual_gate_yaw_threshold = 0.35   # рад
        self.visual_gate_visible_threshold = 0.5
        self.visual_gate_hold_steps = 3
        self.visual_gate_counter = 0
        self.visual_gate_last_blocked = 0.0
        # ===== hard timeout for hanging near current inspect point =====

        self.point_stall_steps = 0
        self.ema_like_pretrain_mode = bool(kwargs.pop("ema_like_pretrain_mode", False))



        self.primary_detector_fn: Optional[Callable[[np.ndarray, Any], List[Dict[str, float]]]] = None
        self.aux_feature_providers: List[Callable[[Any, Optional[np.ndarray], Optional[Dict[str, float]]], np.ndarray]] = []
        self._yolo_model = None
        self._visual_backend_ready = False
        self._target_memory_steps = 0
        self._last_scene_shape = (0, 0)
        self._last_primary_detection: Optional[Dict[str, float]] = None
        self.raw_target_features = np.zeros(8, dtype=np.float32)
        self.smoothed_target_features = np.zeros(8, dtype=np.float32)
        self.aux_visual_features = np.zeros(self.aux_visual_feature_dim, dtype=np.float32)
        self.debug_draw_detections = bool(debug_draw_detections)
        self.debug_show_window = bool(debug_show_window)
        self.debug_save_video = bool(debug_save_video)
        self.debug_video_path = str(debug_video_path)
        self.debug_draw_every_n_steps = max(1, int(debug_draw_every_n_steps))
        self.debug_video_writer = None
        self.debug_print_yolo = bool(debug_print_yolo)
        self.debug_print_seg = bool(kwargs.pop("debug_print_seg", False))
        self.debug_print_obs = bool(debug_print_obs)
        self.hide_final_goal_in_obs = bool(kwargs.pop("hide_final_goal_in_obs", True))
        self.yolo_inference_interval = max(1, int(yolo_inference_interval))
        self._last_detections: List[Dict[str, float]] = []
        self.async_visual_update = bool(async_visual_update)
        self.async_camera_poll_interval = float(async_camera_poll_interval)
        self.async_inference_min_interval = float(async_inference_min_interval)
        self.keep_last_async_frame = bool(keep_last_async_frame)
        self._visual_thread = None
        self._visual_stop_event = None
        self._visual_lock = None
        self.visual_client = None
        self._async_latest_frame = None
        self._async_latest_detections: List[Dict[str, float]] = []
        self._async_latest_tracks: List[Dict[str, float]] = []
        self._async_latest_selected: Optional[Dict[str, float]] = None
        self._async_latest_raw_features = np.zeros(8, dtype=np.float32)
        self._async_latest_smoothed_features = np.zeros(8, dtype=np.float32)
        self._async_latest_timestamp = 0.0
        self._async_last_infer_time = 0.0
        self._last_visual_infer_step = -10 ** 9
        self._last_deep_sort_tracks: List[Dict[str, float]] = []
        self.use_deep_sort = True
        self.deep_sort_tracker = None
        self.camera_hfov_deg = 84.0  # подставьте ваш реальный HFOV
        self.current_locked_waypoint_idx = -1
        self.current_locked_track_missed = 0
        self._track_active = False
        self._track_missed = 0
        self._track_cx = 0.0
        self._track_cy = 0.0
        self._track_area = 0.0
        self._track_conf = 0.0
        self._track_class_id = -1.0
        self._track_class_name = "none"
        self.yolo_det_count_sum = 0.0
        self.yolo_steps_with_det = 0
        self.yolo_conf_sum = 0.0
        self.yolo_area_sum = 0.0
        self.yolo_center_err_sum = 0.0
        self.yolo_selected_switch_count = 0
        self._prev_selected_signature = None
        self.seg_model_path = kwargs.pop("seg_model_path", None)
        self.spark_model_path = kwargs.pop("spark_model_path", None)
        self.seg_device = kwargs.pop("seg_device", self.yolo_device)
        self.spark_device = kwargs.pop("spark_device", self.yolo_device)
        self.seg_conf_threshold = float(kwargs.pop("seg_conf_threshold", 0.25))
        self.spark_conf_threshold = float(kwargs.pop("spark_conf_threshold", 0.25))
        self.spark_iou_threshold = float(spark_iou_threshold)
        self.seg_input_size = int(kwargs.pop("seg_input_size", 640))
        self.spark_input_size = int(kwargs.pop("spark_input_size", 640))

        self.seg_inference_interval = int(kwargs.pop("seg_inference_interval", 10))
        self.spark_inference_interval = max(1, int(kwargs.pop("spark_inference_interval", 20)))

        self._last_seg_infer_step = -10 ** 9
        self._last_spark_infer_step = -10 ** 9

        self._last_seg_infer_ms = 0.0
        self._last_spark_infer_ms = 0.0
        self._spark_infer_count = 0

        self._seg_model = None
        self._spark_model = None

        self._prev_seg_union_mask = None
        self._spark_persist_steps = 0

        self.seg_steps_with_det = 0
        self.seg_det_count_sum = 0.0
        self.seg_cable_ratio_sum = 0.0
        self.seg_tower_ratio_sum = 0.0
        self.seg_center_err_sum = 0.0
        self.seg_mask_stability_sum = 0.0

        # ===== per-class episode averages for episode_metrics.csv =====
        self.seg_cable_steps_with_det = 0
        self.seg_tower_steps_with_det = 0

        self.seg_cable_center_dx_sum = 0.0
        self.seg_cable_center_dy_sum = 0.0
        self.seg_tower_center_dx_sum = 0.0
        self.seg_tower_center_dy_sum = 0.0

        self.seg_cable_angle_sin_sum = 0.0
        self.seg_cable_angle_cos_sum = 0.0

        self.spark_steps_with_det = 0
        self.spark_event_count = 0
        self.spark_max_conf_sum = 0.0
        self.spark_mean_conf_sum = 0.0
        self.spark_max_area_sum = 0.0
        self.spark_event_score_sum = 0.0
        self.spark_persistent_count = 0

        self._last_yolo_info = {
            "yolo_det_count": 0,
            "yolo_selected_class_name": "none",
            "yolo_selected_class_id": -1,
            "yolo_selected_conf": 0.0,
            "yolo_selected_area": 0.0,
            "yolo_selected_center_error": 0.0,
            "yolo_visible": 0.0,
            "yolo_class_norm": 0.0,
            "yolo_conf": 0.0,
            "yolo_cx": 0.0,
            "yolo_cy": 0.0,
            "yolo_area": 0.0,
            "yolo_dx": 0.0,
            "yolo_dy": 0.0,
        }

        self._last_seg_info = {
            "seg_visible": 0.0,
            "seg_det_count": 0,

            "seg_cable_ratio": 0.0,
            "seg_tower_ratio": 0.0,

            "seg_cable_center_dx": 0.0,
            "seg_cable_center_dy": 0.0,
            "seg_tower_center_dx": 0.0,
            "seg_tower_center_dy": 0.0,

            "seg_cable_angle_sin": 0.0,
            "seg_cable_angle_cos": 1.0,

            "seg_center_dx": 0.0,
            "seg_center_dy": 0.0,
            "seg_line_angle_sin": 0.0,
            "seg_line_angle_cos": 1.0,

            "seg_mask_stability": 0.0,
        }

        self._last_spark_info = {
            "spark_visible": 0.0,
            "spark_det_count": 0,
            "spark_max_conf": 0.0,
            "spark_mean_conf": 0.0,
            "spark_max_area": 0.0,
            "spark_bbox_x1": 0.0,
            "spark_bbox_y1": 0.0,
            "spark_bbox_x2": 0.0,
            "spark_bbox_y2": 0.0,
            "spark_bbox_area": 0.0,
            "spark_center_dx": 0.0,
            "spark_center_dy": 0.0,
            "spark_event_score": 0.0,
            "spark_persistent": 0.0,
        }

        # ===== spark hazard supervisor =====
        self.spark_response_enabled = bool(kwargs.pop("spark_response_enabled", True))

        # подтверждение spark-события
        self.spark_confirm_conf_thr = float(kwargs.pop("spark_confirm_conf_thr", 0.45))
        self.spark_confirm_event_thr = float(kwargs.pop("spark_confirm_event_thr", 0.10))
        self.spark_confirm_steps_req = int(kwargs.pop("spark_confirm_steps_req", 2))

        # сколько шагов без spark нужно, чтобы выйти из hazardous mode
        self.spark_clear_steps_req = int(kwargs.pop("spark_clear_steps_req", 10))

        # сколько шагов длится именно первый evasive override
        self.spark_override_duration_steps = int(kwargs.pop("spark_override_duration_steps", 18))

        # величины override в action-space [-1, 1]
        self.spark_response_backoff_ratio = float(kwargs.pop("spark_response_backoff_ratio", 0.65))
        self.spark_response_lateral_ratio = float(kwargs.pop("spark_response_lateral_ratio", 0.45))

        # насколько осторожно лететь в hazardous mode после разовой реакции
        self.spark_hazard_xy_scale = float(kwargs.pop("spark_hazard_xy_scale", 0.65))
        self.spark_hazard_yaw_scale = float(kwargs.pop("spark_hazard_yaw_scale", 0.70))

        # supervisor state
        self.spark_alarm_latched = False
        self.spark_response_consumed = False
        self.spark_response_active = False
        self.spark_override_steps_left = 0
        self.spark_confirm_steps = 0
        self.spark_clear_steps = 0
        self.spark_last_seen_waypoint = -1
        self.spark_last_response_waypoint = -1
        # inspect-ветка: чуть более живые фильтры, чем nav

        super().__init__(
            *args,
            control_command_duration=control_command_duration,
            **kwargs,
        )


        self.control_command_duration = (
            None if control_command_duration is None else float(control_command_duration)
        )
        self.last_command_duration = 0.0
        self._last_move_task = None

        self.front_avoid_enabled = False
        self.cmd_filter_alpha_z = 0.42
        self.cmd_filter_alpha_yaw = 0.32


        # ===== inspect overrides: убираем лишнюю ватность от внешнего EMA =====
        self.action_smoothing_alpha = 0.45
        self.min_xy_cmd_ratio = 0.03
        # ===== чуть больше authority по vertical / yaw =====
        self.max_vertical_speed = self.max_velocity * 0.30

        self.max_delta_vz_cmd = 0.18
        self.max_delta_yaw_rate_cmd = 3.5
        self.inspection_target_position = self._normalize_point(
            inspection_target_position, "inspection_target_position"
        )

        self.obs_dim = int(self.obs_dim + self.inspect_extra_obs_dim + self.applied_cmd_obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        if self.aux_visual_feature_dim > 0:
            self.obs_dim = int(self.obs_dim + self.aux_visual_feature_dim)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
        # =========================================================
        # Inspect target helpers
        # =========================================================
        self._setup_visual_pipeline()
        if self.use_deep_sort:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self.deep_sort_tracker = DeepSort(max_age=12, n_init=1)
                print("[DEEP_SORT] init ok",
                      "use_deep_sort=", self.use_deep_sort,
                      "tracker_none=", self.deep_sort_tracker is None)
            except Exception as exc:
                self.deep_sort_tracker = None
                print(f"[DEEP_SORT] init failed: {exc}")
        else:
            print("[DEEP_SORT] disabled by config")


        self.yolo_reward_scale = float(np.clip(yolo_reward_scale, 0.0, 1.0))


        # ===== reward/termination cleanup without changing observation =====
        self.point_stall_timeout_sec = 3.5
        self.point_stall_progress_epsilon = max(float(self.no_progress_threshold), 0.0030)
        self.point_stall_penalty_base = -0.02
        self.point_stall_penalty_scale = -0.18

        self.disable_point_stall_when_gate_blocked = True
        self.disable_point_stall_termination_when_gate_blocked = True
        self.disable_target_waypoint_mismatch_for_inspect = True
        self.disable_obstacle_clearance_reward = True

        self.gate_blocked_camping_penalty = -0.06
        self.free_camping_penalty = -0.03
        self.visual_gate_penalty_base = -0.05

        self.caps_w_xy *= 0.55
        self.caps_w_z *= 0.70
        self.caps_w_yaw *= 0.60

        self.asap_w_xy *= 0.40
        self.asap_w_z *= 0.50
        self.asap_w_yaw *= 0.45

        self.active_progress_reward_weight = min(float(self.active_progress_reward_weight), 0.004)
        self.inspection_pairs = None

        if inspection_pairs is not None:
            self.inspection_pairs = []
            for i, pair in enumerate(inspection_pairs):
                route_point = np.asarray(pair["route_point"], dtype=np.float32)
                target_point = np.asarray(pair["target_point"], dtype=np.float32)

                if route_point.shape != (3,):
                    raise ValueError(
                        f"inspection_pairs[{i}]['route_point'] must be shape (3,), got {route_point.shape}"
                    )
                if target_point.shape != (3,):
                    raise ValueError(
                        f"inspection_pairs[{i}]['target_point'] must be shape (3,), got {target_point.shape}"
                    )

                point_mode = str(pair.get("point_mode", "inspect")).lower()
                visual_gate_required = bool(pair.get("visual_gate_required", point_mode == "inspect"))

                self.inspection_pairs.append({
                    "route_point": route_point.copy(),
                    "target_point": target_point.copy(),
                    "point_mode": point_mode,
                    "visual_gate_required": visual_gate_required,
                    "visual_gate_visible_threshold": float(pair.get("visual_gate_visible_threshold", 0.50)),
                    "visual_gate_center_threshold": float(pair.get("visual_gate_center_threshold", 0.20)),
                    "visual_gate_yaw_threshold": float(pair.get("visual_gate_yaw_threshold", 0.35)),
                    "visual_gate_hold_steps": int(pair.get("visual_gate_hold_steps", 3)),
                    "desired_standoff_distance": pair.get("desired_standoff_distance", None),
                })

        self.inspection_pair_sets = None
        if inspection_pair_sets is not None:
            self.inspection_pair_sets = []
            for set_idx, pair_set in enumerate(inspection_pair_sets):
                parsed_set = []
                for i, pair in enumerate(pair_set):
                    route_point = np.asarray(pair["route_point"], dtype=np.float32)
                    target_point = np.asarray(pair["target_point"], dtype=np.float32)

                    if route_point.shape != (3,):
                        raise ValueError(
                            f"inspection_pair_sets[{set_idx}][{i}]['route_point'] must be shape (3,), got {route_point.shape}"
                        )
                    if target_point.shape != (3,):
                        raise ValueError(
                            f"inspection_pair_sets[{set_idx}][{i}]['target_point'] must be shape (3,), got {target_point.shape}"
                        )

                    point_mode = str(pair.get("point_mode", "inspect")).lower()
                    visual_gate_required = bool(pair.get("visual_gate_required", point_mode == "inspect"))

                    parsed_set.append({
                        "route_point": route_point.copy(),
                        "target_point": target_point.copy(),
                        "point_mode": point_mode,
                        "visual_gate_required": visual_gate_required,
                        "visual_gate_visible_threshold": float(pair.get("visual_gate_visible_threshold", 0.50)),
                        "visual_gate_center_threshold": float(pair.get("visual_gate_center_threshold", 0.20)),
                        "visual_gate_yaw_threshold": float(pair.get("visual_gate_yaw_threshold", 0.35)),
                        "visual_gate_hold_steps": int(pair.get("visual_gate_hold_steps", 3)),
                        "desired_standoff_distance": pair.get("desired_standoff_distance", None),
                    })
                self.inspection_pair_sets.append(parsed_set)

    # =========================================================
    # Visual inspection pipeline
    # =========================================================

    def _setup_visual_pipeline(self):
        if not self.enable_visual_inspection:
            return
        if self.yolo_model_path is None and self.primary_detector_fn is None:
            print("[InspectEnv] Visual pipeline enabled, but no detector configured.")
            return
        if self.yolo_model_path is not None:
            try:
                from ultralytics import YOLO  # type: ignore
                self._yolo_model = YOLO(self.yolo_model_path)
                self._visual_backend_ready = True
                print(f"[InspectEnv] Loaded YOLO model: {self.yolo_model_path}")
            except Exception as exc:
                self._yolo_model = None
                self._visual_backend_ready = False
                print(f"[InspectEnv] YOLO load failed: {exc}")
        else:
            self._visual_backend_ready = True

    def register_primary_detector(self, detector_fn: Callable[[np.ndarray, Any], List[Dict[str, float]]]):
        self.primary_detector_fn = detector_fn
        self.enable_visual_inspection = True
        self._visual_backend_ready = True

    def register_aux_feature_provider(self, provider_fn: Callable[[Any, Optional[np.ndarray], Optional[Dict[str, float]]], np.ndarray]):
        self.aux_feature_providers.append(provider_fn)

    def reset_visual_target_state(self):
        self._target_memory_steps = 0
        self._last_scene_shape = (0, 0)
        self._last_primary_detection = None
        self._last_detections = []
        self._last_deep_sort_tracks = []
        self._async_latest_tracks = []
        self.raw_target_features[:] = 0.0
        self.smoothed_target_features[:] = 0.0
        if self.aux_visual_feature_dim > 0:
            self.aux_visual_features[:] = 0.0
        self._track_active = False
        self._track_missed = 0
        self._track_cx = 0.0
        self._track_cy = 0.0
        self._track_area = 0.0
        self._track_conf = 0.0
        self._track_class_id = -1.0
        self._track_class_name = "none"
        self._prev_selected_signature = None
        self._last_yolo_info = {
            "yolo_det_count": 0,
            "yolo_selected_class_name": "none",
            "yolo_selected_class_id": -1,
            "yolo_selected_conf": 0.0,
            "yolo_selected_area": 0.0,
            "yolo_selected_center_error": 0.0,
            "yolo_visible": 0.0,
            "yolo_class_norm": 0.0,
            "yolo_conf": 0.0,
            "yolo_cx": 0.0,
            "yolo_cy": 0.0,
            "yolo_area": 0.0,
            "yolo_dx": 0.0,
            "yolo_dy": 0.0,
        }
        self._last_seg_info = {
            "seg_visible": 0.0,
            "seg_det_count": 0,

            "seg_cable_ratio": 0.0,
            "seg_tower_ratio": 0.0,

            "seg_cable_center_dx": 0.0,
            "seg_cable_center_dy": 0.0,
            "seg_tower_center_dx": 0.0,
            "seg_tower_center_dy": 0.0,

            "seg_cable_angle_sin": 0.0,
            "seg_cable_angle_cos": 1.0,

            "seg_center_dx": 0.0,
            "seg_center_dy": 0.0,
            "seg_line_angle_sin": 0.0,
            "seg_line_angle_cos": 1.0,

            "seg_mask_stability": 0.0,
        }
        self._last_spark_info = {
            "spark_visible": 0.0,
            "spark_det_count": 0,
            "spark_max_conf": 0.0,
            "spark_mean_conf": 0.0,
            "spark_max_area": 0.0,
            "spark_bbox_x1": 0.0,
            "spark_bbox_y1": 0.0,
            "spark_bbox_x2": 0.0,
            "spark_bbox_y2": 0.0,
            "spark_bbox_area": 0.0,
            "spark_center_dx": 0.0,
            "spark_center_dy": 0.0,
            "spark_event_score": 0.0,
            "spark_persistent": 0.0,
        }
        self._prev_seg_union_mask = None
        self._spark_persist_steps = 0

        self.seg_steps_with_det = 0
        self.seg_det_count_sum = 0.0
        self.seg_cable_ratio_sum = 0.0
        self.seg_tower_ratio_sum = 0.0
        self.seg_center_err_sum = 0.0
        self.seg_mask_stability_sum = 0.0

        self.seg_cable_steps_with_det = 0
        self.seg_tower_steps_with_det = 0

        self.seg_cable_center_dx_sum = 0.0
        self.seg_cable_center_dy_sum = 0.0
        self.seg_tower_center_dx_sum = 0.0
        self.seg_tower_center_dy_sum = 0.0

        self.seg_cable_angle_sin_sum = 0.0
        self.seg_cable_angle_cos_sum = 0.0

        self.spark_steps_with_det = 0
        self.spark_event_count = 0
        self.spark_max_conf_sum = 0.0
        self.spark_mean_conf_sum = 0.0
        self.spark_max_area_sum = 0.0
        self.spark_event_score_sum = 0.0
        self.spark_persistent_count = 0
        self._spark_infer_count = 0
        self._last_spark_infer_step = -10 ** 9
        self._last_spark_infer_ms = 0.0
        self._last_spark_trace_diag = {
            "spark_override_active_dbg": 0.0,
            "spark_action_override_changed_dbg": 0.0,
            "spark_action_to_apply_0_dbg": 0.0,
            "spark_action_to_apply_1_dbg": 0.0,
            "spark_action_to_apply_2_dbg": 0.0,
            "spark_action_to_apply_3_dbg": 0.0,
            "spark_action_after_override_0_dbg": 0.0,
            "spark_action_after_override_1_dbg": 0.0,
            "spark_action_after_override_2_dbg": 0.0,
            "spark_action_after_override_3_dbg": 0.0,
        }
        self.yolo_det_count_sum = 0.0
        self.yolo_steps_with_det = 0
        self.yolo_conf_sum = 0.0
        self.yolo_area_sum = 0.0
        self.yolo_center_err_sum = 0.0
        self.yolo_selected_switch_count = 0
        self.current_locked_track_id = None
        self.current_locked_waypoint_idx = -1
        self.current_locked_track_missed = 0

    def _debug_primary_yolo_state(
            self,
            stage: str,
            image_bgr: Optional[np.ndarray],
            detections: Optional[List[Dict[str, float]]],
            selected: Optional[Dict[str, float]],
            raw_features: Optional[np.ndarray],
            smoothed: Optional[np.ndarray],
    ):
        if not self.debug_print_yolo:
            return

        h = w = 0
        if image_bgr is not None:
            h, w = image_bgr.shape[:2]

        meta = None
        point_mode = "unknown"
        visual_gate_required = False
        gate_visible_thr = self.visual_gate_visible_threshold
        gate_center_thr = self.visual_gate_center_threshold
        gate_yaw_thr = self.visual_gate_yaw_threshold

        try:
            meta = self._get_current_inspection_pair_meta()
            point_mode = str(meta.get("point_mode", "inspect")).lower()
            visual_gate_required = bool(meta.get("visual_gate_required", False))
            gate_visible_thr = float(meta.get("visual_gate_visible_threshold", self.visual_gate_visible_threshold))
            gate_center_thr = float(meta.get("visual_gate_center_threshold", self.visual_gate_center_threshold))
            gate_yaw_thr = float(meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))
        except Exception:
            pass

        async_age = -1.0
        if self.async_visual_update and float(getattr(self, "_async_latest_timestamp", 0.0)) > 0.0:
            async_age = float(time.time() - float(self._async_latest_timestamp))



        for i, det in enumerate((detections or [])[:10]):
            cls_id = int(det.get("class_id", -1))
            cls_name_raw = str(det.get("class_name", "none")).lower()
            cls_name = self.class_name_aliases.get(cls_name_raw, cls_name_raw)

            target_ok_id = (
                    self.yolo_target_class_ids is None
                    or cls_id in self.yolo_target_class_ids
            )
            target_ok_name = (
                    self.yolo_target_class_names is None
                    or cls_name in self.yolo_target_class_names
            )
            area_ok = float(det.get("area", 0.0)) >= float(self.target_min_area)
            conf_ok = float(det.get("conf", 0.0)) >= float(self.yolo_conf_threshold)

            center_dist = float(np.sqrt(
                float(det.get("dx_from_center", 0.0)) ** 2 +
                float(det.get("dy_from_center", 0.0)) ** 2
            ))

            track_dist = -1.0
            if self._track_active:
                track_dist = float(np.sqrt(
                    (float(det.get("center_x", 0.0)) - float(self._track_cx)) ** 2 +
                    (float(det.get("center_y", 0.0)) - float(self._track_cy)) ** 2
                ))



        if selected is None:
            print("[DEBUG YOLO SELECT] none")
        else:
            print(
                f"[DEBUG YOLO SELECT] "
                f"cls_id={int(selected.get('class_id', -1))} "
                f"cls_name='{str(selected.get('class_name', 'none'))}' "
                f"conf={float(selected.get('conf', 0.0)):.3f} "
                f"area={float(selected.get('area', 0.0)):.6f} "
                f"cx={float(selected.get('center_x', 0.0)):.3f} "
                f"cy={float(selected.get('center_y', 0.0)):.3f} "
                f"track_id={selected.get('track_id', 'none')}"
            )



        if raw_features is None:
            raw_features = np.zeros(8, dtype=np.float32)
        if smoothed is None:
            smoothed = np.zeros(8, dtype=np.float32)



    def _reset_seg_info_for_empty_frame(self):
        self._last_seg_info = {
            "seg_visible": 0.0,
            "seg_det_count": 0,
            "seg_cable_ratio": 0.0,
            "seg_tower_ratio": 0.0,
            "seg_cable_center_dx": 0.0,
            "seg_cable_center_dy": 0.0,
            "seg_tower_center_dx": 0.0,
            "seg_tower_center_dy": 0.0,
            "seg_cable_angle_sin": 0.0,
            "seg_cable_angle_cos": 1.0,
            "seg_center_dx": 0.0,
            "seg_center_dy": 0.0,
            "seg_line_angle_sin": 0.0,
            "seg_line_angle_cos": 1.0,
            "seg_mask_stability": 0.0,
        }

    def _update_seg_debug_info(self, image_bgr: Optional[np.ndarray]):
        if image_bgr is None:
            self._reset_seg_info_for_empty_frame()
            return
        step = int(getattr(self, "current_step", 0))

        # Не запускаем segmentation каждый env-step.
        # Между inference оставляем последнее значение _last_seg_info.
        if (step - int(getattr(self, "_last_seg_infer_step", -10 ** 9))) < int(
                getattr(self, "seg_inference_interval", 10)):
            return

        self._last_seg_infer_step = step
        seg_t0 = time.perf_counter()

        self._lazy_load_external_debug_models()
        if self._seg_model is None:
            self._reset_seg_info_for_empty_frame()
            return

        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            self._reset_seg_info_for_empty_frame()
            return

        try:
            results = self._seg_model.predict(
                source=image_bgr,
                conf=self.seg_conf_threshold,
                imgsz=self.seg_input_size,
                verbose=False,
                device=self.seg_device,
            )
        except Exception:
            self._reset_seg_info_for_empty_frame()
            return

        if not results:
            self._reset_seg_info_for_empty_frame()
            return

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        names = getattr(result, "names", {}) or {}

        if (
                boxes is None
                or masks is None
                or getattr(masks, "data", None) is None
                or getattr(boxes, "cls", None) is None
                or getattr(boxes, "conf", None) is None
        ):
            if self.debug_print_seg:
                print(
                    f"[DEBUG SEG EMPTY] step={int(self.current_step)} "
                    f"boxes_none={boxes is None} "
                    f"masks_none={masks is None} "
                    f"masks_data_none={getattr(masks, 'data', None) is None if masks is not None else True}"
                )
            self._reset_seg_info_for_empty_frame()
            return

        cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        mask_arr = masks.data.detach().cpu().numpy()

        if self.debug_print_seg:
            try:
                print(
                    f"[DEBUG SEG RAW] step={int(self.current_step)} "
                    f"results={len(results)} "
                    f"boxes={0 if boxes is None else len(cls_ids)} "
                    f"masks_shape={None if getattr(masks, 'data', None) is None else tuple(masks.data.shape)} "
                    f"names={dict(names)}"
                )
            except Exception as exc:
                print(f"[DEBUG SEG RAW] print-failed: {exc}")

        mask_arr = np.asarray(mask_arr)
        if mask_arr.ndim == 0 or mask_arr.size == 0:
            self._reset_seg_info_for_empty_frame()
            return

        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]

        if mask_arr.shape[0] == 0:
            self._reset_seg_info_for_empty_frame()
            return

        mask_arr = (mask_arr > 0.5).astype(np.uint8)

        n = min(len(cls_ids), len(confs), len(mask_arr))
        if n <= 0:
            self._reset_seg_info_for_empty_frame()
            return

        cls_ids = cls_ids[:n]
        confs = confs[:n]
        mask_arr = mask_arr[:n]

        cable_union = np.zeros_like(mask_arr[0], dtype=np.uint8)
        tower_union = np.zeros_like(mask_arr[0], dtype=np.uint8)
        det_count = 0

        best_cable_mask = None
        best_cable_score = -1.0
        best_tower_mask = None
        best_tower_score = -1.0

        for det_idx, (cls_id, conf, m) in enumerate(zip(cls_ids, confs, mask_arr)):
            cls_id_int = int(cls_id)
            class_name = str(names.get(cls_id_int, str(cls_id_int))).lower()

            mask_pixels = int(m.sum())
            mask_ratio = float(mask_pixels) / max(float(h * w), 1.0)

            is_cable = class_name == "cable" or cls_id_int in (0, 1)
            is_tower = class_name in ("tower_lattice", "tower_tucohy", "tower_wooden") or cls_id_int in (2, 3, 4)
            accepted = bool(is_cable or is_tower)

            if self.debug_print_seg:
                print(
                    f"[DEBUG SEG DET] step={int(self.current_step)} "
                    f"idx={det_idx} cls_id={cls_id_int} class_name='{class_name}' "
                    f"conf={float(conf):.3f} mask_pixels={mask_pixels} mask_ratio={mask_ratio:.6f} "
                    f"is_cable={int(is_cable)} is_tower={int(is_tower)} accepted={int(accepted)}"
                )

            if not accepted:
                continue

            det_count += 1
            score = float(conf) * float(mask_pixels)

            if is_cable:
                cable_union = np.maximum(cable_union, m)
                if score > best_cable_score:
                    best_cable_score = score
                    best_cable_mask = m.copy()

            if is_tower:
                tower_union = np.maximum(tower_union, m)
                if score > best_tower_score:
                    best_tower_score = score
                    best_tower_mask = m.copy()

        total_union = np.maximum(cable_union, tower_union)

        cable_ratio = float(cable_union.sum()) / float(max(h * w, 1))
        tower_ratio = float(tower_union.sum()) / float(max(h * w, 1))
        seg_visible = 1.0 if det_count > 0 and int(total_union.sum()) > 0 else 0.0

        def _mask_center(mask: Optional[np.ndarray]):
            if mask is None or int(mask.sum()) <= 0:
                return 0.0, 0.0
            ys, xs = np.where(mask > 0)
            cx = float(xs.mean())
            cy = float(ys.mean())
            dx = float(np.clip((cx / max(w, 1)) - 0.5, -1.0, 1.0))
            dy = float(np.clip((cy / max(h, 1)) - 0.5, -1.0, 1.0))
            return dx, dy

        def _mask_angle(mask: Optional[np.ndarray]):
            if mask is None or int(mask.sum()) <= 0:
                return 0.0, 1.0
            ys, xs = np.where(mask > 0)
            if xs.shape[0] < 2:
                return 0.0, 1.0
            pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            mean = pts.mean(axis=0, keepdims=True)
            centered = pts - mean
            cov = centered.T @ centered / max(float(len(pts)), 1.0)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            angle = float(np.arctan2(principal[1], principal[0]))
            return float(np.sin(angle)), float(np.cos(angle))

        cable_center_dx, cable_center_dy = _mask_center(best_cable_mask)
        tower_center_dx, tower_center_dy = _mask_center(best_tower_mask)
        cable_angle_sin, cable_angle_cos = _mask_angle(best_cable_mask)

        if best_cable_mask is not None and int(best_cable_mask.sum()) > 0:
            seg_center_dx = cable_center_dx
            seg_center_dy = cable_center_dy
            angle_sin = cable_angle_sin
            angle_cos = cable_angle_cos
        else:
            seg_center_dx = tower_center_dx
            seg_center_dy = tower_center_dy
            angle_sin = 0.0
            angle_cos = 1.0

        stability = 0.0
        if self._prev_seg_union_mask is not None:
            prev = self._prev_seg_union_mask.astype(bool)
            curr = total_union.astype(bool)
            inter = float(np.logical_and(prev, curr).sum())
            union = float(np.logical_or(prev, curr).sum())
            stability = inter / max(union, 1.0)

        self._prev_seg_union_mask = total_union.copy()

        center_err = float(np.sqrt(seg_center_dx ** 2 + seg_center_dy ** 2))

        self._last_seg_info = {
            "seg_visible": float(seg_visible),
            "seg_det_count": int(det_count),
            "seg_cable_ratio": float(cable_ratio),
            "seg_tower_ratio": float(tower_ratio),
            "seg_cable_center_dx": float(cable_center_dx),
            "seg_cable_center_dy": float(cable_center_dy),
            "seg_tower_center_dx": float(tower_center_dx),
            "seg_tower_center_dy": float(tower_center_dy),
            "seg_cable_angle_sin": float(cable_angle_sin),
            "seg_cable_angle_cos": float(cable_angle_cos),
            "seg_center_dx": float(seg_center_dx),
            "seg_center_dy": float(seg_center_dy),
            "seg_line_angle_sin": float(angle_sin),
            "seg_line_angle_cos": float(angle_cos),
            "seg_mask_stability": float(stability),
        }
        self._last_seg_infer_ms = 1000.0 * (time.perf_counter() - seg_t0)



        if self.debug_print_seg:
            print(
                f"[DEBUG SEG OUT] step={int(self.current_step)} "
                f"det_count={int(det_count)} "
                f"seg_visible={float(seg_visible):.0f} "
                f"cable_ratio={float(cable_ratio):.6f} "
                f"tower_ratio={float(tower_ratio):.6f} "
                f"cable_dx={float(cable_center_dx):.4f} "
                f"cable_dy={float(cable_center_dy):.4f} "
                f"tower_dx={float(tower_center_dx):.4f} "
                f"tower_dy={float(tower_center_dy):.4f} "
                f"stability={float(stability):.4f}"
            )

        if seg_visible > 0.5:
            self.seg_steps_with_det += 1
        self.seg_det_count_sum += float(det_count)
        self.seg_cable_ratio_sum += float(cable_ratio)
        self.seg_tower_ratio_sum += float(tower_ratio)
        self.seg_center_err_sum += float(center_err)
        self.seg_mask_stability_sum += float(stability)

        if best_cable_mask is not None and int(best_cable_mask.sum()) > 0:
            self.seg_cable_steps_with_det += 1
            self.seg_cable_center_dx_sum += float(self._last_seg_info["seg_cable_center_dx"])
            self.seg_cable_center_dy_sum += float(self._last_seg_info["seg_cable_center_dy"])
            self.seg_cable_angle_sin_sum += float(self._last_seg_info["seg_cable_angle_sin"])
            self.seg_cable_angle_cos_sum += float(self._last_seg_info["seg_cable_angle_cos"])

        if best_tower_mask is not None and int(best_tower_mask.sum()) > 0:
            self.seg_tower_steps_with_det += 1
            self.seg_tower_center_dx_sum += float(self._last_seg_info["seg_tower_center_dx"])
            self.seg_tower_center_dy_sum += float(self._last_seg_info["seg_tower_center_dy"])

    def update_aux_visual_features(self, values):
        if self.aux_visual_feature_dim <= 0:
            return
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        n = min(arr.shape[0], self.aux_visual_feature_dim)
        self.aux_visual_features[:] = 0.0
        self.aux_visual_features[:n] = arr[:n]

    def set_yolo_reward_scale(self, value: float):
        self.yolo_reward_scale = float(np.clip(value, 0.0, 1.0))

    def get_yolo_reward_scale(self) -> float:
        return float(self.yolo_reward_scale)

    def _lazy_load_external_debug_models(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            return

        if self.seg_model_path is not None and self._seg_model is None:
            try:
                self._seg_model = YOLO(self.seg_model_path)
            except Exception:
                self._seg_model = None

        if self.spark_model_path is not None and self._spark_model is None:
            try:
                self._spark_model = YOLO(self.spark_model_path)
            except Exception:
                self._spark_model = None

    def _start_visual_worker(self):
        if not self.enable_visual_inspection or not self.async_visual_update:
            return
        if not self._visual_backend_ready:
            return
        if self._visual_thread is not None and self._visual_thread.is_alive():
            return

        self._visual_stop_event = threading.Event()
        self._visual_lock = threading.Lock()

        self.visual_client = airsim.MultirotorClient(ip=self.ip_address)
        self.visual_client.confirmConnection()

        self._async_latest_frame = None
        self._async_latest_detections = []
        self._async_latest_tracks = []
        self._async_latest_selected = None
        self._async_latest_raw_features = None
        self._async_latest_smoothed_features = None
        self._async_latest_timestamp = 0.0
        self._async_last_infer_time = 0.0

        self._visual_thread = threading.Thread(
            target=self._visual_worker_loop,
            name=f"visual_worker_{self.vehicle_name}",
            daemon=True,
        )
        self._visual_thread.start()

    def _stop_visual_worker(self):
        if self._visual_stop_event is not None:
            self._visual_stop_event.set()
        if self._visual_thread is not None:
            self._visual_thread.join(timeout=1.0)

        self._visual_thread = None
        self._visual_stop_event = None
        self._visual_lock = None
        self.visual_client = None

    def _visual_worker_loop(self):
        last_frame = None
        last_detections = []

        while self._visual_stop_event is not None and not self._visual_stop_event.is_set():
            try:
                frame = self._get_scene_image_bgr(client=self.visual_client)
                now = time.time()

                if frame is None:
                    time.sleep(self.async_camera_poll_interval)
                    continue

                detections = last_detections
                should_infer = (now - self._async_last_infer_time) >= self.async_inference_min_interval

                if should_infer:
                    detections = self._run_primary_detector(frame)
                    self._async_last_infer_time = now
                    last_detections = [dict(d) for d in detections]

                with self._visual_lock:
                    self._async_latest_frame = frame.copy() if self.keep_last_async_frame else frame
                    self._async_latest_detections = [dict(d) for d in detections]
                    self._async_latest_tracks = []
                    self._async_latest_selected = None
                    self._async_latest_raw_features = None
                    self._async_latest_smoothed_features = None
                    self._async_latest_timestamp = now

                last_frame = frame
                time.sleep(self.async_camera_poll_interval)

            except Exception:
                time.sleep(max(self.async_camera_poll_interval, 0.02))

    def _select_with_tracking(self, detections):
        """
        Выбор детекции с учётом предыдущего трека (anti-flicker)
        """

        if not detections:
            return None

        # если нет активного трека → обычный выбор
        if not self._track_active:
            return self._select_primary_detection(detections)

        best = None
        best_score = -1e9

        for d in detections:
            cx = d["center_x"]
            cy = d["center_y"]

            dx = cx - self._track_cx
            dy = cy - self._track_cy
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > self.track_gate_px:
                continue  # отбрасываем далекие bbox

            score = d["conf"] - dist * 2.0

            if score > best_score:
                best = d
                best_score = score

        # fallback если ничего не прошло гейтинг
        if best is None:
            best = self._select_primary_detection(detections)

        return best

    def _read_async_visual_state(self):
        if self._visual_lock is None:
            return None, [], None, None, None

        with self._visual_lock:
            frame = None if self._async_latest_frame is None else self._async_latest_frame.copy()
            detections = [dict(d) for d in self._async_latest_detections]

        return frame, detections, None, None, None

    def _get_scene_image_bgr(
            self,
            client=None,
            camera_name: Optional[str] = None,
            image_type=None,
    ) -> Optional[np.ndarray]:
        if not self.enable_visual_inspection:
            return None

        rpc_client = self.client if client is None else client
        camera_name_to_use = self.scene_camera_name if camera_name is None else camera_name
        image_type_to_use = self.scene_image_type if image_type is None else image_type

        try:
            responses = rpc_client.simGetImages([
                airsim.ImageRequest(
                    camera_name_to_use,
                    image_type_to_use,
                    pixels_as_float=False,
                    compress=False,
                )
            ], vehicle_name=self.vehicle_name)

            if not responses:
                return None

            resp = responses[0]
            if resp.width <= 0 or resp.height <= 0 or len(resp.image_data_uint8) == 0:
                return None

            img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            if img.size != resp.height * resp.width * 3:
                return None

            img = img.reshape(resp.height, resp.width, 3)
            if camera_name is None and image_type is None:
                self._last_scene_shape = (resp.height, resp.width)
            return img
        except Exception:
            return None

    def _run_primary_detector(self, image_bgr: np.ndarray) -> List[Dict[str, float]]:
        if self.primary_detector_fn is not None:
            try:
                detections = self.primary_detector_fn(image_bgr, self)
                return detections if detections is not None else []
            except Exception:
                return []

        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model.predict(
                source=image_bgr,
                conf=self.yolo_conf_threshold,
                iou=self.yolo_iou_threshold,
                imgsz=self.yolo_input_size,
                verbose=False,
                device=self.yolo_device,
            )
        except Exception:
            return []

        detections: List[Dict[str, float]] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None) or {}
        if boxes is None:
            return detections

        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else []
        clss = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else []

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            if bw <= 1e-6 or bh <= 1e-6:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area_norm = float((bw * bh) / max(float(w * h), 1e-6))
            cls_id_int = int(cls_id)
            class_name = str(names.get(cls_id_int, cls_id_int)).lower()
            cls_name_alias = self.class_name_aliases.get(class_name, class_name)

            target_ok_id = (
                    self.yolo_target_class_ids is None
                    or cls_id_int in self.yolo_target_class_ids
            )
            target_ok_name = (
                    self.yolo_target_class_names is None
                    or cls_name_alias in self.yolo_target_class_names
            )
            conf_ok = float(conf) >= float(self.yolo_conf_threshold)
            area_ok = float(area_norm) >= float(self.target_min_area)

            if self.debug_print_yolo:
                print(
                    f"[DEBUG YOLO RAW DET] step={int(self.current_step)} "
                    f"cls_id={cls_id_int} raw_cls='{class_name}' alias_cls='{cls_name_alias}' "
                    f"conf={float(conf):.3f} area={float(area_norm):.6f} "
                    f"cx={float(cx / max(w, 1)):.3f} cy={float(cy / max(h, 1)):.3f} "
                    f"ok_id={int(target_ok_id)} ok_name={int(target_ok_name)} "
                    f"ok_conf={int(conf_ok)} ok_area={int(area_ok)}"
                )

            if self.yolo_target_class_ids is not None and cls_id_int not in self.yolo_target_class_ids:
                continue
            if self.yolo_target_class_names is not None and class_name not in self.yolo_target_class_names:
                continue
            if float(conf) < self.yolo_conf_threshold:
                continue
            if area_norm < self.target_min_area:
                continue

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "class_id": float(cls_id_int),
                "class_name": class_name,
                "conf": float(conf),
                "center_x": float(np.clip(cx / max(w, 1), 0.0, 1.0)),
                "center_y": float(np.clip(cy / max(h, 1), 0.0, 1.0)),
                "area": float(np.clip(area_norm, 0.0, 1.0)),
                "dx_from_center": float(np.clip((cx / max(w, 1)) - 0.5, -1.0, 1.0)),
                "dy_from_center": float(np.clip((cy / max(h, 1)) - 0.5, -1.0, 1.0)),
            })
        return detections

    def _select_primary_detection(self, detections: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not detections:
            return None

        prev = self._last_primary_detection

        def score(det: Dict[str, float]) -> float:
            center_dist = float(np.sqrt(det["dx_from_center"] ** 2 + det["dy_from_center"] ** 2))
            score_val = 2.0 * float(det["conf"]) + 1.5 * float(det["area"])
            if self.yolo_prefer_central_target:
                score_val -= 0.75 * center_dist
            if prev is not None:
                prev_dx = float(det["center_x"] - prev.get("center_x", 0.5))
                prev_dy = float(det["center_y"] - prev.get("center_y", 0.5))
                temporal_dist = float(np.sqrt(prev_dx ** 2 + prev_dy ** 2))
                score_val -= 0.50 * temporal_dist
            return score_val

        return max(detections, key=score)

    def _detection_to_feature_vector(self, detection: Optional[Dict[str, float]], visible_override: Optional[float] = None) -> np.ndarray:
        vals = np.zeros(8, dtype=np.float32)
        if detection is None:
            if visible_override is not None:
                vals[0] = float(visible_override)
            return vals

        class_id_norm = float(detection.get("class_id", 0.0))
        class_id_norm = float(np.clip(class_id_norm / 10.0, 0.0, 1.0))
        vals[:] = np.array([
            1.0 if visible_override is None else float(visible_override),
            class_id_norm,
            float(np.clip(detection.get("conf", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("center_x", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("center_y", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("area", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("dx_from_center", 0.0), -1.0, 1.0)),
            float(np.clip(detection.get("dy_from_center", 0.0), -1.0, 1.0)),
        ], dtype=np.float32)
        return vals

    def _apply_target_ema(self, raw_features: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.target_ema_alpha, 0.0, 1.0))
        if np.allclose(self.smoothed_target_features, 0.0):
            self.smoothed_target_features = raw_features.astype(np.float32).copy()
        else:
            self.smoothed_target_features = (
                (1.0 - alpha) * self.smoothed_target_features + alpha * raw_features
            ).astype(np.float32)

        if raw_features[0] < 0.5:
            self.smoothed_target_features[0] *= 0.85
        return self.smoothed_target_features.copy()

    def _update_visual_target_features(self):
        if not self.enable_visual_inspection:
            # Полностью отключаем visual/camera pipeline, но НЕ меняем размер observation.
            self.raw_target_features[:] = 0.0
            self.smoothed_target_features[:] = 0.0

            if self.aux_visual_feature_dim > 0:
                self.aux_visual_features[:] = 0.0

            self._last_detections = []
            self._last_primary_detection = None
            return

        image = None
        detections = []
        selected = None
        smoothed = None

        if self.async_visual_update:
            image, detections, _, _, _ = self._read_async_visual_state()
            selected = None

            if image is not None:
                self._last_detections = detections
                selected = self._select_detection_for_current_waypoint(detections, image)

            if selected is not None:
                self._last_primary_detection = dict(selected)
                self._target_memory_steps = 0
                self._update_tracking(selected)
                self.raw_target_features = self._tracking_to_feature_vector()
            else:
                self._update_tracking(None)
                self._target_memory_steps += 1

                if self._track_active:
                    self.raw_target_features = self._tracking_to_feature_vector()
                    self.raw_target_features[0] = 0.0
                elif self._last_primary_detection is not None and self._target_memory_steps <= self.target_hold_steps:
                    self.raw_target_features = self._detection_to_feature_vector(
                        self._last_primary_detection,
                        visible_override=0.0,
                    )
                else:
                    self._last_primary_detection = None
                    self.raw_target_features = np.zeros(8, dtype=np.float32)

            smoothed = self._apply_target_ema(self.raw_target_features)
            self._debug_primary_yolo_state(
                stage="async-main",
                image_bgr=image,
                detections=detections,
                selected=selected,
                raw_features=self.raw_target_features,
                smoothed=smoothed,
            )
        else:
            image = self._get_scene_image_bgr()
            detections = []
            selected = None

            if image is not None and self._visual_backend_ready:
                if self.current_step % self.yolo_inference_interval == 0:
                    detections = self._run_primary_detector(image)
                    self._last_detections = detections
                    self._last_visual_infer_step = int(self.current_step)
                else:
                    detections = self._last_detections

                selected = self._select_detection_for_current_waypoint(detections, image)

            if selected is not None:
                self._last_primary_detection = dict(selected)
                self._target_memory_steps = 0
                self._update_tracking(selected)
                self.raw_target_features = self._tracking_to_feature_vector()
            else:
                self._update_tracking(None)
                self._target_memory_steps += 1

                if self._track_active:
                    self.raw_target_features = self._tracking_to_feature_vector()
                    self.raw_target_features[0] = 0.0
                elif self._last_primary_detection is not None and self._target_memory_steps <= self.target_hold_steps:
                    self.raw_target_features = self._detection_to_feature_vector(
                        self._last_primary_detection,
                        visible_override=0.0,
                    )
                else:
                    self._last_primary_detection = None
                    self.raw_target_features = np.zeros(8, dtype=np.float32)

            smoothed = self._apply_target_ema(self.raw_target_features)
            self._debug_primary_yolo_state(
                stage="sync",
                image_bgr=image,
                detections=detections,
                selected=selected,
                raw_features=self.raw_target_features,
                smoothed=smoothed,
            )

        if smoothed is None:
            smoothed = np.zeros(8, dtype=np.float32)

        self.update_yolo_features(
            target_visible=float(smoothed[0]),
            target_class_id_norm=float(smoothed[1]),
            target_conf=float(smoothed[2]),
            target_center_x=float(smoothed[3]),
            target_center_y=float(smoothed[4]),
            target_area=float(smoothed[5]),
            target_dx_from_center=float(smoothed[6]),
            target_dy_from_center=float(smoothed[7]),
        )
        # external diagnostic detectors: do NOT affect obs/policy
        # и не должны сидеть на критическом train-path
        # ===== segmentation for cable safety / aux visual features =====
        need_seg_update = (
                getattr(self, "seg_model_path", None) is not None
                and (
                        bool(getattr(self, "cable_safety_enabled", False))
                        or int(getattr(self, "aux_visual_feature_dim", 0)) > 0
                        or bool(getattr(self, "debug_print_seg", False))
                )
        )

        if need_seg_update:
            self._update_seg_debug_info(image)

        # Spark/debug detectors отдельно, чтобы не тянуть spark при safety-cable.
        need_spark_update = (
                bool(getattr(self, "run_external_debug_detectors", False))
                and getattr(self, "spark_model_path", None) is not None
        )

        if need_spark_update:
            self._update_spark_debug_info(image)

        det_count = len(detections) if detections is not None else 0
        selected_class_name = "none"
        selected_class_id = -1
        selected_conf = 0.0
        selected_area = 0.0
        selected_center_error = 0.0

        if selected is not None:
            selected_class_name = str(selected.get("class_name", "none"))
            selected_class_id = int(selected.get("class_id", -1))
            selected_conf = float(selected.get("conf", 0.0))
            selected_area = float(selected.get("area", 0.0))
            selected_center_error = float(np.linalg.norm([
                float(selected.get("dx_from_center", 0.0)),
                float(selected.get("dy_from_center", 0.0)),
            ]))

            selected_signature = (
                selected_class_id,
                round(float(selected.get("center_x", 0.0)), 3),
                round(float(selected.get("center_y", 0.0)), 3),
                round(selected_conf, 3),
            )
            if self._prev_selected_signature is not None and selected_signature != self._prev_selected_signature:
                self.yolo_selected_switch_count += 1
            self._prev_selected_signature = selected_signature
        else:
            self._prev_selected_signature = None

        self._last_yolo_info = {
            "yolo_det_count": int(det_count),
            "yolo_selected_class_name": selected_class_name,
            "yolo_selected_class_id": int(selected_class_id),
            "yolo_selected_conf": float(selected_conf),
            "yolo_selected_area": float(selected_area),
            "yolo_selected_center_error": float(selected_center_error),
            "yolo_visible": float(smoothed[0]),
            "yolo_class_norm": float(smoothed[1]),
            "yolo_conf": float(smoothed[2]),
            "yolo_cx": float(smoothed[3]),
            "yolo_cy": float(smoothed[4]),
            "yolo_area": float(smoothed[5]),
            "yolo_dx": float(smoothed[6]),
            "yolo_dy": float(smoothed[7]),
        }
        # ===== YOLO episode accumulators =====
        self.yolo_det_count_sum += float(det_count)

        if det_count > 0:
            self.yolo_steps_with_det += 1

        if selected is not None:
            self.yolo_conf_sum += float(selected_conf)
            self.yolo_area_sum += float(selected_area)
            self.yolo_center_err_sum += float(selected_center_error)



        if self.debug_print_yolo and (self.current_step % max(1, self.debug_draw_every_n_steps) == 0):
            if selected is not None:
                print(
                    "[DEBUG YOLO] "
                    f"step={self.current_step} det_count={det_count} "
                    f"selected={selected_class_name} "
                    f"conf={selected_conf:.3f} "
                    f"area={selected_area:.4f} "
                    f"dx={float(selected.get('dx_from_center', 0.0)):.3f} "
                    f"dy={float(selected.get('dy_from_center', 0.0)):.3f}"
                )
            else:
                print(
                    "[DEBUG YOLO] "
                    f"step={self.current_step} det_count={det_count} selected=None"
                )

            print(
                "[DEBUG YOLO FEATURES] "
                f"visible={float(smoothed[0]):.3f} "
                f"class_norm={float(smoothed[1]):.3f} "
                f"conf={float(smoothed[2]):.3f} "
                f"cx={float(smoothed[3]):.3f} "
                f"cy={float(smoothed[4]):.3f} "
                f"area={float(smoothed[5]):.4f} "
                f"dx={float(smoothed[6]):.3f} "
                f"dy={float(smoothed[7]):.3f}"
            )

        if (
                self.debug_draw_detections
                and image is not None
                and (self.current_step % self.debug_draw_every_n_steps == 0)
        ):
            deep_sort_tracks = (
                [dict(t) for t in self._async_latest_tracks]
                if self.async_visual_update
                else [dict(t) for t in self._last_deep_sort_tracks]
            )

            debug_frame = self._draw_debug(
                frame=image,
                detections=detections,
                deep_sort_tracks=deep_sort_tracks,
                selected_det=selected,
                smoothed_features=smoothed,
            )
            self._handle_debug_output(debug_frame)

        if self.aux_visual_feature_dim > 0:
            aux_blocks = []
            for provider in self.aux_feature_providers:
                try:
                    arr = np.asarray(provider(self, image, selected), dtype=np.float32).reshape(-1)
                    aux_blocks.append(arr)
                except Exception:
                    continue
            if aux_blocks:
                self.update_aux_visual_features(np.concatenate(aux_blocks, axis=0))
            else:
                self.aux_visual_features[:] = 0.0

    def _get_pair_desired_standoff(self, idx=None) -> float:
        fallback = float(getattr(self, "desired_standoff_distance", 0.0))
        if (not np.isfinite(fallback)) or fallback <= 0.0:
            fallback = 1.0

        try:
            if self.inspection_pairs is None or len(self.inspection_pairs) == 0:
                return float(fallback)

            if idx is None:
                idx = int(getattr(self, "current_waypoint_idx", 0))
            idx = int(np.clip(idx, 0, len(self.inspection_pairs) - 1))
            pair = self.inspection_pairs[idx]

            pair_desired = pair.get("desired_standoff_distance", None)
            if pair_desired is not None:
                pair_desired = float(pair_desired)
                if np.isfinite(pair_desired) and pair_desired > 0.0:
                    return float(pair_desired)

            route_point = np.asarray(pair.get("route_point"), dtype=np.float32)
            target_point = np.asarray(pair.get("target_point"), dtype=np.float32)
            if route_point.shape == (3,) and target_point.shape == (3,):
                geometric_desired = float(np.linalg.norm(route_point - target_point))
                if np.isfinite(geometric_desired) and geometric_desired > 0.0:
                    return float(geometric_desired)

        except Exception:
            pass

        return float(fallback)

    def _get_inspection_metrics(self):
        pos = self._current_position_vec()
        target = self._get_inspection_target()
        delta = target - pos
        dist_raw = float(np.linalg.norm(delta))
        desired_standoff = float(self._get_pair_desired_standoff())
        standoff_error_raw = float(dist_raw - desired_standoff)
        return pos, target, delta, dist_raw, standoff_error_raw

    def _visual_gate_ready_for_current_waypoint(self):
        state = self._get_current_alignment_state(final=False)

        ready_now = bool(state["mode"] == "visual" and state["ready"] > 0.5)

        hold_steps = int(self._get_current_inspection_pair_meta().get("visual_gate_hold_steps", 3))

        if ready_now:
            self.visual_gate_counter += 1
        else:
            self.visual_gate_counter = 0

        ready_hold = self.visual_gate_counter >= hold_steps
        return ready_hold, state["visible"], state["center_error"], state["yaw_error_geom"]

    def _get_inspection_obs_features(self) -> np.ndarray:
        pos = self._current_position_vec()
        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()
        desired_standoff = float(self._get_pair_desired_standoff())

        if point_mode == "transit":
            if self.active_local_waypoint is not None:
                ref_target = self.active_local_waypoint.copy()
            else:
                ref_target = self.mission_route[self.current_waypoint_idx].copy()

            delta = ref_target - pos
            dist_raw = float(np.linalg.norm(delta))
            standoff_error_raw = 0.0
        else:
            target = self._get_inspection_target()
            delta = target - pos
            dist_raw = float(np.linalg.norm(delta))
            standoff_error_raw = float(dist_raw - desired_standoff)

        max_goal_dist = np.sqrt(
            self.world_xy_limit ** 2 + self.world_xy_limit ** 2 + self.max_altitude ** 2
        )

        inspect_dx = float(np.clip(delta[0] / self.world_xy_limit, -1.0, 1.0))
        inspect_dy = float(np.clip(delta[1] / self.world_xy_limit, -1.0, 1.0))
        inspect_dz = float(np.clip(delta[2] / self.max_altitude, -1.0, 1.0))
        inspect_distance_norm = float(np.clip(dist_raw / max(max_goal_dist, 1e-6), 0.0, 1.0))
        standoff_error_norm = float(
            np.clip(standoff_error_raw / max(desired_standoff, 1e-6), -1.0, 1.0)
        )

        return np.array(
            [inspect_dx, inspect_dy, inspect_dz, inspect_distance_norm, standoff_error_norm],
            dtype=np.float32,
        )

    def _get_applied_command_obs(self) -> np.ndarray:
        """
        Семантика слота сохранена прежней (4 признака),
        но вместо applied command теперь отдаём cmd_tracking_error:
        desired_cmd - applied_cmd, нормированный в [-1, 1].

        desired_cmd берём из последней policy-команды после action smoothing,
        чтобы policy видела, насколько env исказил её намерение.
        """
        desired_vx = float(np.clip(self.last_smoothed_action[0], -1.0, 1.0)) * float(self.max_velocity)
        desired_vy = float(np.clip(self.last_smoothed_action[1], -1.0, 1.0)) * float(self.max_velocity)
        desired_vz = float(np.clip(self.last_smoothed_action[2], -1.0, 1.0)) * float(self.max_vertical_speed)
        desired_yaw = float(np.clip(self.last_smoothed_action[3], -1.0, 1.0)) * float(self.max_yaw_rate)

        err_vx = float(np.clip((desired_vx - self.last_applied_vx_cmd) / max(self.max_velocity, 1e-6), -1.0, 1.0))
        err_vy = float(np.clip((desired_vy - self.last_applied_vy_cmd) / max(self.max_velocity, 1e-6), -1.0, 1.0))
        err_vz = float(np.clip((desired_vz - self.last_applied_vz_cmd) / max(self.max_vertical_speed, 1e-6), -1.0, 1.0))
        err_yaw = float(
            np.clip((desired_yaw - self.last_applied_yaw_rate_cmd) / max(self.max_yaw_rate, 1e-6), -1.0, 1.0))

        return np.array([err_vx, err_vy, err_vz, err_yaw], dtype=np.float32)

    def _update_tracking(self, detection):
        if detection is None:
            self._track_missed += 1

            if self._track_missed > self.track_max_missed:
                self._track_active = False

            return

        cx = detection["center_x"]
        cy = detection["center_y"]
        area = detection["area"]
        conf = detection["conf"]

        if not self._track_active:
            # инициализация
            self._track_active = True
            self._track_cx = cx
            self._track_cy = cy
            self._track_area = area
        else:
            # EMA сглаживание
            self._track_cx = (
                    self.track_pos_alpha * cx +
                    (1.0 - self.track_pos_alpha) * self._track_cx
            )
            self._track_cy = (
                    self.track_pos_alpha * cy +
                    (1.0 - self.track_pos_alpha) * self._track_cy
            )
            self._track_area = (
                    self.track_area_alpha * area +
                    (1.0 - self.track_area_alpha) * self._track_area
            )

        self._track_conf = conf
        self._track_class_id = detection["class_id"]
        self._track_class_name = detection["class_name"]

        self._track_missed = 0

    def _tracking_to_feature_vector(self) -> np.ndarray:
        if self._track_active:
            return np.array([
                1.0,
                float(np.clip(self._track_class_id / 10.0, 0.0, 1.0)),
                float(np.clip(self._track_conf, 0.0, 1.0)),
                float(np.clip(self._track_cx, 0.0, 1.0)),
                float(np.clip(self._track_cy, 0.0, 1.0)),
                float(np.clip(self._track_area, 0.0, 1.0)),
                float(np.clip(self._track_cx - 0.5, -1.0, 1.0)),
                float(np.clip(self._track_cy - 0.5, -1.0, 1.0)),
            ], dtype=np.float32)
        return np.zeros(8, dtype=np.float32)
    # =========================================================
    # Gymnasium API
    # =========================================================

    def step(self, action: np.ndarray):
        policy_raw = np.zeros(4, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        n = min(4, int(action_arr.size))
        if n > 0:
            policy_raw[:n] = action_arr[:n]
        self._last_policy_raw_action = policy_raw.copy()

        obs, reward, terminated, truncated, info = super().step(action)

        if self.enable_heavy_diagnostics:
            control_debug = info.setdefault("control_debug", {})
            if isinstance(getattr(self, "_last_action_apply_diag", None), dict):
                control_debug.update(self._last_action_apply_diag)
            control_debug["raw_action_0"] = float(self._last_policy_raw_action[0])
            control_debug["raw_action_1"] = float(self._last_policy_raw_action[1])
            control_debug["raw_action_2"] = float(self._last_policy_raw_action[2])
            control_debug["raw_action_3"] = float(self._last_policy_raw_action[3])

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # ===== STOP async =====
        self._stop_visual_worker()
        self._stop_control_stream_worker()
        self._prev_route_point_distance_raw = None
        # ===== RESET STATES =====
        self.lost_target_steps = 0
        self.visible_steps = 0
        self.active_inspection_target = None

        self.visual_gate_counter = 0
        self.visual_gate_last_blocked = 0.0
        self.point_stall_steps = 0
        self.inspect_lock = False
        self.inspect_lock_steps = 0
        self.reset_visual_target_state()
        self.reset_spark_response_state()
        self._cable_hold_left = 0
        self._cable_hold_last_update_step = -1
        self._cable_hold_hard = False
        self._cable_hold_last_dx = 0.0
        self._cable_hold_last_dy = 0.0
        self._cable_hold_last_ratio = 0.0
        self._last_cable_safety_state = None
        self._last_cable_safety_state_step = -1
        self._corridor_hard_raw_streak = 0
        self._corridor_near_collision_streak = 0

        # ===== SELECT ROUTE BEFORE BASE RESET =====
        selected_pairs = None
        selected_start = None
        selected_route_idx = None
        original_start_points = self.start_points

        if self.inspection_pair_sets is not None and len(self.inspection_pair_sets) > 0:
            selected_route_idx = 0 if len(self.inspection_pair_sets) == 1 else int(
                self.np_random.integers(0, len(self.inspection_pair_sets))
            )
            self.current_route_idx = int(selected_route_idx)
            selected_pairs = self.inspection_pair_sets[self.current_route_idx]

            if self.start_points is not None and len(self.start_points) == len(self.inspection_pair_sets):
                selected_start = np.asarray(self.start_points[self.current_route_idx], dtype=np.float32).copy()

        if selected_pairs is not None:
            self.inspection_pairs = []
            for pair in selected_pairs:
                self.inspection_pairs.append({
                    "route_point": np.asarray(pair["route_point"], dtype=np.float32).copy(),
                    "target_point": np.asarray(pair["target_point"], dtype=np.float32).copy(),
                    "point_mode": str(pair.get("point_mode", "inspect")).lower(),
                    "visual_gate_required": bool(
                        pair.get("visual_gate_required", str(pair.get("point_mode", "inspect")).lower() == "inspect")
                    ),
                    "visual_gate_visible_threshold": float(pair.get("visual_gate_visible_threshold", 0.50)),
                    "visual_gate_center_threshold": float(pair.get("visual_gate_center_threshold", 0.20)),
                    "visual_gate_yaw_threshold": float(pair.get("visual_gate_yaw_threshold", 0.35)),
                    "visual_gate_hold_steps": int(pair.get("visual_gate_hold_steps", 3)),
                    "desired_standoff_distance": pair.get("desired_standoff_distance", None),
                })

        if selected_start is not None:
            self.start_position = selected_start.copy()
            self.start_points = np.asarray([selected_start], dtype=np.float32).copy()


        # ===== BASE RESET =====
        try:
            obs, info = super().reset(seed=seed, options=options)
        finally:
            self.start_points = original_start_points

        if selected_route_idx is not None:
            self.current_route_idx = int(selected_route_idx)

        self.filtered_vx_cmd = 0.0
        self.filtered_vy_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.filtered_yaw_rate_cmd = 0.0

        self.last_applied_vx_cmd = 0.0
        self.last_applied_vy_cmd = 0.0
        self.last_applied_vz_cmd = 0.0
        self.last_applied_yaw_rate_cmd = 0.0

        self.smoothed_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0

        obs = self._get_obs()
        self.prev_local_distance = self._get_local_distance_raw()
        self.prev_final_distance = self._get_goal_distance_raw()

        self.collision_latched = False
        self.ground_contact_steps = 0
        self.inactive_steps = 0
        self.no_progress_steps = 0
        self.failed_progress_steps = 0
        self.point_stall_steps = 0


        # ===== DEBUG ROUTE SYNC =====


        for key in [
            "target_visible_reward",
            "target_center_reward",
            "target_scale_reward",
            "standoff_reward",
            "inspection_hold_reward",
            "visual_align_reward",
        ]:
            self.episode_stats.setdefault(f"{key}_sum", 0.0)
            self.episode_stats.setdefault(f"{key}_abs_sum", 0.0)

        info["route_idx"] = int(self.current_route_idx)

        self._start_visual_worker()
        self._start_control_stream_worker()
        current_target = self._get_inspection_target()
        final_target = self._get_final_inspection_target()

        inspection_pairs_len = int(len(self.inspection_pairs)) if self.inspection_pairs is not None else 0
        current_route_points_len = int(len(self.current_route_points)) if self.current_route_points is not None else 0
        mission_route_len = int(len(self.mission_route)) if getattr(self, "mission_route", None) is not None else 0
        first_route_point = None
        start_route_xy_distance = None

        if self.current_route_points is not None and len(self.current_route_points) > 0:
            first_route_point = np.asarray(self.current_route_points[0], dtype=np.float32).copy()

        if self.start_position is not None and first_route_point is not None:
            try:
                start_route_xy_distance = float(np.linalg.norm(
                    np.asarray(self.start_position[:2], dtype=np.float32)
                    - np.asarray(first_route_point[:2], dtype=np.float32)
                ))
            except Exception:
                start_route_xy_distance = None

        info["goal_position"] = self.goal_position.copy()
        info["inspection_target_position"] = current_target.copy()
        info["final_inspection_target_position"] = final_target.copy()

        info["debug_current_waypoint_idx"] = int(self.current_waypoint_idx)
        info["debug_inspection_pairs_len"] = inspection_pairs_len
        info["debug_current_route_points_len"] = current_route_points_len
        info["debug_mission_route_len"] = mission_route_len

        if self.debug_print_reset:
            print("DEBUG ASYNC RESET selected_route_idx:", selected_route_idx)
            print("DEBUG ASYNC RESET selected_start:", selected_start)
            print("DEBUG ASYNC RESET start_position(after super):", self.start_position)
            print("DEBUG ASYNC RESET current_route_first_point:", first_route_point)
            print("DEBUG ASYNC RESET goal_position:", self.goal_position)
            print("DEBUG ASYNC RESET start_route_xy_distance:", start_route_xy_distance)
            print("DEBUG ASYNC RESET route_idx:", self.current_route_idx)
            print("DEBUG ASYNC RESET current_waypoint_idx:", self.current_waypoint_idx)
            print("DEBUG ASYNC RESET inspection_pairs_len:", inspection_pairs_len)
            print("DEBUG ASYNC RESET current_route_points_len:", current_route_points_len)
            print("DEBUG ASYNC RESET mission_route_len:", mission_route_len)
            print("DEBUG ASYNC RESET goal(final_route_point):", self.goal_position)
            print("DEBUG ASYNC RESET current_target(wp_idx=0):", current_target)
            print("DEBUG ASYNC RESET final_target(last_pair):", final_target)
            self.print_current_route_target()
        self._prev_progress_wp_idx = int(self.current_waypoint_idx)
        self._prev_progress_local_path_index = int(getattr(self, "local_path_index", -1))
        if self.active_local_waypoint is not None:
            self._prev_progress_local_target = np.asarray(self.active_local_waypoint, dtype=np.float32).copy()
        else:
            self._prev_progress_local_target = None
        self._local_target_changed_this_step = False
        return obs, info


    def close(self):
        self._stop_control_stream_worker()
        self._stop_visual_worker()

        if self.debug_video_writer is not None:
            self.debug_video_writer.release()
            self.debug_video_writer = None

        if self.debug_show_window:
            cv2.destroyAllWindows()

        super().close()

    # =========================================================
    # Observation
    # =========================================================

    def _get_obs(self) -> np.ndarray:
        self._update_visual_target_features()

        # base obs из nav-env
        base_obs = super()._get_obs()

        # скрываем финальную цель от policy
        base_obs = self._mask_final_goal_from_base_obs(base_obs)

        # visual block оставляем как раньше
        if self.yolo_feature_dim > 0:
            yolo_start = int(len(base_obs) - self.yolo_feature_dim)
            obs_visual_block = self._get_obs_visual_block()
            base_obs = base_obs.copy()
            base_obs[yolo_start:yolo_start + self.yolo_feature_dim] = obs_visual_block

        inspect_obs = self._get_inspection_obs_features()
        applied_cmd_obs = self._get_applied_command_obs()

        parts = [base_obs, inspect_obs, applied_cmd_obs]

        if self.debug_print_obs and (self.current_step % max(1, self.debug_draw_every_n_steps) == 0):
            yolo_start = int(len(base_obs) - self.yolo_feature_dim)
            yolo_obs = base_obs[yolo_start:].copy() if self.yolo_feature_dim > 0 else np.array([], dtype=np.float32)

        if self.aux_visual_feature_dim > 0:
            parts.append(self.aux_visual_features.astype(np.float32))

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _cancel_last_task_safe(self):
        try:
            self.client.cancelLastTask(vehicle_name=self.vehicle_name)
        except TypeError:
            try:
                self.client.cancelLastTask()
            except Exception:
                pass
        except Exception:
            pass

    def _move_drone_to_position(self, target_position: np.ndarray):
        """
        Inspect reset через жёсткую постановку pose.
        """
        if target_position is None:
            return

        point = np.asarray(target_position, dtype=np.float32).reshape(3).copy()

        target_x = float(point[0])
        target_y = float(point[1])
        target_z = float(point[2])

        self.start_position = point.copy()

        self._cancel_last_task_safe()

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
        except Exception:
            pass

        time.sleep(0.10)

        # ===== unlock flying-state BEFORE hard pose =====
        try:
            state0 = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            landed_state0 = int(state0.landed_state)
        except Exception:
            landed_state0 = 1

        if landed_state0 != 0:
            try:
                task = self.client.takeoffAsync(
                    timeout_sec=5.0,
                    vehicle_name=self.vehicle_name,
                )
                task.join()
            except TypeError:
                try:
                    task = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
                    task.join()
                except Exception:
                    pass
            except Exception:
                pass

            self._cancel_last_task_safe()
            time.sleep(0.15)

            try:
                self.client.enableApiControl(True, self.vehicle_name)
                self.client.armDisarm(True, self.vehicle_name)
            except Exception:
                pass

        pose = airsim.Pose(
            airsim.Vector3r(target_x, target_y, target_z),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )

        try:
            self.client.simSetVehiclePose(
                pose,
                True,
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            pass

        time.sleep(0.20)

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
        except Exception:
            pass

        # Короткий Z-hold warmup после pose.
        for _ in range(8):
            try:
                self.client.moveByVelocityZAsync(
                    vx=0.0,
                    vy=0.0,
                    z=float(target_z),
                    duration=0.10,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                    vehicle_name=self.vehicle_name,
                ).join()
            except Exception:
                break

            time.sleep(0.02)

        self._cancel_last_task_safe()

        try:
            self.client.simSetVehiclePose(
                pose,
                True,
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            pass

        time.sleep(0.10)

        try:
            state0 = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            landed_state0 = int(state0.landed_state)
        except Exception:
            landed_state0 = 1

        if landed_state0 != 0:
            try:
                task = self.client.takeoffAsync(
                    timeout_sec=5.0,
                    vehicle_name=self.vehicle_name,
                )
                task.join()
            except TypeError:
                try:
                    task = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
                    task.join()
                except Exception:
                    pass
            except Exception:
                pass

            time.sleep(0.15)

            self._cancel_last_task_safe()

            try:
                self.client.enableApiControl(True, self.vehicle_name)
                self.client.armDisarm(True, self.vehicle_name)
            except Exception:
                pass

            time.sleep(0.10)

            try:
                self.client.simSetVehiclePose(
                    pose,
                    True,
                    vehicle_name=self.vehicle_name,
                )
            except Exception:
                pass

            time.sleep(0.10)

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
        except Exception:
            pass

        # ===== FINAL RESET FREEZE: pose + zero velocity after takeoff unlock =====
        try:
            self._cancel_last_task_safe()
        except Exception:
            pass

        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except TypeError:
            try:
                self.client.hoverAsync().join()
            except Exception:
                pass
        except Exception:
            pass

        for _ in range(4):
            try:
                self.client.simSetVehiclePose(
                    pose,
                    True,
                    vehicle_name=self.vehicle_name,
                )
            except Exception:
                pass

            try:
                kin0 = airsim.KinematicsState()
                kin0.position = airsim.Vector3r(target_x, target_y, target_z)
                kin0.orientation = airsim.to_quaternion(0.0, 0.0, 0.0)

                kin0.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
                kin0.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
                kin0.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
                kin0.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)

                self.client.simSetKinematics(
                    kin0,
                    True,
                    vehicle_name=self.vehicle_name,
                )
            except AttributeError:
                pass
            except Exception:
                pass

            time.sleep(0.05)

        try:
            self._cancel_last_task_safe()
        except Exception:
            pass

        # Сброс фильтров команд после reset.
        self.filtered_vx_cmd = 0.0
        self.filtered_vy_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.filtered_yaw_rate_cmd = 0.0

        self.last_applied_vx_cmd = 0.0
        self.last_applied_vy_cmd = 0.0
        self.last_applied_vz_cmd = 0.0
        self.last_applied_yaw_rate_cmd = 0.0

        if hasattr(self, "last_applied_z_target_cmd"):
            self.last_applied_z_target_cmd = float(target_z)




        self.smoothed_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0

        # После hard pose reset не переносим старое состояние контакта с землёй.
        self.ground_contact_steps = 0
        self.last_landed_state = 0.0
        self.last_down_distance = float(self.max_sensor_distance)
        self.last_ground_speed = 0.0
        self.last_tilt_abs = 0.0

        self._invalidate_state_cache()

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position

        final_pos = np.array(
            [float(pos.x_val), float(pos.y_val), float(pos.z_val)],
            dtype=np.float32,
        )

        if abs(float(final_pos[2]) - target_z) > 1.0:
            raise RuntimeError(
                f"Inspect reset failed: final_z={float(final_pos[2]):.3f}, "
                f"target_z={target_z:.3f}, final_pos={final_pos.tolist()}"
            )

    def _get_current_route_z_target(self) -> float:
        """
        Для inspect высота должна идти от настоящего route_point,
        а не от active_local_waypoint/local_path.

        Иначе при просадке дрона local_path пересобирается от текущей
        упавшей позиции, и active_local_waypoint.z начинает тянуть вниз.
        """
        try:
            idx = int(np.clip(
                int(getattr(self, "current_waypoint_idx", 0)),
                0,
                len(self.inspection_pairs) - 1,
            ))

            if self.inspection_pairs is not None and len(self.inspection_pairs) > 0:
                return float(self.inspection_pairs[idx]["route_point"][2])

            if getattr(self, "mission_route", None):
                idx = int(np.clip(idx, 0, len(self.mission_route) - 1))
                return float(self.mission_route[idx][2])

            if self.active_local_waypoint is not None:
                return float(self.active_local_waypoint[2])

            return float(self.goal_position[2])
        except Exception:
            return float(getattr(self, "last_target_z", -3.0))

    def _get_inspect_command_duration(self) -> float:
        command_duration = getattr(self, "control_command_duration", None)

        # Команда должна жить дольше реального шага среды.
        # Иначе AirSim/SimpleFlight уходит в hover:
        # "API call was not received, entering hover mode for safety".
        min_duration = float(max(float(self.step_duration) * 2.5, 0.16))

        if command_duration is None:
            return min_duration

        return float(max(float(command_duration), min_duration))
    # =========================================================
    # Action
    # =========================================================
    def _force_local_waypoint_to_current_route_point(self) -> None:
        """
        Жёстко фиксирует active_local_waypoint на настоящем route_point.
        Нужно при Z-capture, иначе local_path пересобирается от падающей позиции
        и active_local_waypoint.z начинает ползти к земле.
        """
        try:
            idx = int(np.clip(
                int(getattr(self, "current_waypoint_idx", 0)),
                0,
                max(len(self.inspection_pairs) - 1, 0)
            ))

            if self.inspection_pairs is not None and len(self.inspection_pairs) > 0:
                route_wp = np.asarray(
                    self.inspection_pairs[idx]["route_point"],
                    dtype=np.float32,
                ).reshape(3)
            elif getattr(self, "mission_route", None):
                idx = int(np.clip(idx, 0, len(self.mission_route) - 1))
                route_wp = np.asarray(self.mission_route[idx], dtype=np.float32).reshape(3)
            else:
                return

            self.active_local_waypoint = route_wp.copy()
            self.local_path_world = [route_wp.copy()]
            self.local_path_index = 0

        except Exception:
            pass

    def _apply_action(self, action: np.ndarray):
        action_to_apply_vec = np.zeros(4, dtype=np.float32)
        action_to_apply_src = np.asarray(action, dtype=np.float32).reshape(-1)
        n_to_apply = min(4, int(action_to_apply_src.size))
        if n_to_apply > 0:
            action_to_apply_vec[:n_to_apply] = action_to_apply_src[:n_to_apply]
        spark_override_active = False

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        now = time.perf_counter()
        prev_apply_time = getattr(self, "_last_apply_wall_time", None)
        self._last_apply_wall_time = now

        real_apply_dt = 0.0
        if prev_apply_time is not None:
            real_apply_dt = now - prev_apply_time

        # if real_apply_dt > float(getattr(self, "last_command_duration", 0.3)) * 0.80:
        #     print(
        #         "[CMD GAP WARNING]",
        #         "step=", int(getattr(self, "current_step", -1)),
        #         "real_apply_dt=", round(float(real_apply_dt), 3),
        #         "command_duration=", round(float(getattr(self, "last_command_duration", 0.0)), 3),
        #     )
        kin = state.kinematics_estimated

        pos = kin.position
        vel = kin.linear_velocity
        roll, pitch, yaw = airsim.to_eularian_angles(kin.orientation)

        current_z = float(pos.z_val)
        current_vz = float(vel.z_val)

        self.last_current_z = current_z
        self.last_current_vz = current_vz
        self.last_current_roll = float(roll)
        self.last_current_pitch = float(pitch)
        self.last_current_yaw = float(yaw)

        # ===== spark supervisor =====
        if self.spark_response_enabled:
            self._update_spark_response_state()
            spark_override_action = self._get_spark_override_action()
            if spark_override_action is not None:
                action = spark_override_action.astype(np.float32)
                spark_override_active = True
                self.inspect_lock = False
                self.inspect_lock_steps = 0

        action_after_override_vec = np.zeros(4, dtype=np.float32)
        action_after_override_src = np.asarray(action, dtype=np.float32).reshape(-1)
        n_after_override = min(4, int(action_after_override_src.size))
        if n_after_override > 0:
            action_after_override_vec[:n_after_override] = action_after_override_src[:n_after_override]


        ax = float(action[0])
        ay = float(action[1])

        if abs(ax) < self.xy_deadzone:
            ax = 0.0
        if abs(ay) < self.xy_deadzone:
            ay = 0.0

        if ax != 0.0 and abs(ax) < self.min_xy_cmd_ratio:
            ax = float(np.sign(ax) * self.min_xy_cmd_ratio)
        if ay != 0.0 and abs(ay) < self.min_xy_cmd_ratio:
            ay = float(np.sign(ay) * self.min_xy_cmd_ratio)

        spark_hazard_mode = self._in_spark_hazard_mode()

        if spark_hazard_mode and not self.spark_response_active:
            ax *= self.spark_hazard_xy_scale
            ay *= self.spark_hazard_xy_scale
            self.inspect_lock = False
            self.inspect_lock_steps = 0

        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()

        manual_yaw_limit_deg = min(self.max_yaw_rate * 0.35, 12.0)
        auto_yaw_k = 20.0
        auto_yaw_limit_deg = min(self.max_yaw_rate * 0.75, 24.0)
        if spark_hazard_mode and not self.spark_response_active:
            manual_yaw_limit_deg *= self.spark_hazard_yaw_scale
            auto_yaw_limit_deg *= self.spark_hazard_yaw_scale
        manual_yaw_rate = float(action[3]) * manual_yaw_limit_deg
        auto_yaw_rate = 0.0
        yaw_target = None
        yaw_error_abs = 0.0

        local_distance_raw = float(self._get_local_distance_raw())

        inspect_route_reached = (
                point_mode == "inspect"
                and local_distance_raw <= self.waypoint_tolerance
        )

        # EMA-подобный pretrain:
        # в non-visual geom pretrain начинаем смотреть на inspect-target сразу,
        # а не только после точного достижения route_point.
        if self._use_ema_like_pretrain_mode():
            inspect_turn_zone = bool(point_mode == "inspect")
            approach_turn_zone = bool(point_mode == "inspect")
        else:
            # штатная async-логика
            inspect_turn_zone = bool(inspect_route_reached)
            approach_turn_zone = False

        inspect_lock_zone = (
                point_mode == "inspect"
                and local_distance_raw <= (self.waypoint_tolerance * 0.90)
        )

        route_target = self.active_local_waypoint.copy() if self.active_local_waypoint is not None else None
        inspect_target = self._get_inspection_target() if point_mode == "inspect" else None

        if point_mode == "transit":
            yaw_target = route_target

        elif point_mode == "inspect":
            if self._use_ema_like_pretrain_mode():
                # EMA-подобно: сразу смотрим на target_point
                if self.face_inspection_target and inspect_target is not None:
                    yaw_target = inspect_target.copy()
                else:
                    yaw_target = route_target
            else:
                # штатная async-логика
                if (
                        self.face_inspection_target
                        and inspect_target is not None
                        and (
                        inspect_route_reached
                        or float(getattr(self, "visual_gate_last_blocked", 0.0)) > 0.5
                )
                ):
                    yaw_target = inspect_target.copy()
                else:
                    yaw_target = route_target

        else:
            yaw_target = route_target

        if inspect_turn_zone:
            auto_yaw_k = max(auto_yaw_k, 28.0)
            auto_yaw_limit_deg = max(auto_yaw_limit_deg, min(self.max_yaw_rate * 0.90, 30.0))


        # ===== real auto-yaw computation =====
        if yaw_target is not None:
            current_x = float(pos.x_val)
            current_y = float(pos.y_val)
            target_x = float(yaw_target[0])
            target_y = float(yaw_target[1])

            target_xy_dist = float(np.hypot(target_x - current_x, target_y - current_y))

            yaw_error = 0.0
            if target_xy_dist > 0.15:
                desired_yaw = float(np.arctan2(target_y - current_y, target_x - current_x))
                yaw_error = float(self._wrap_angle(desired_yaw - float(yaw)))

            yaw_error_abs = abs(yaw_error)
            if yaw_error_abs < 0.04:
                yaw_error = 0.0
                yaw_error_abs = 0.0

            auto_yaw_rate = float(np.clip(
                auto_yaw_k * yaw_error,
                -auto_yaw_limit_deg,
                auto_yaw_limit_deg,
            ))
        else:
            yaw_error_abs = 0.0
            auto_yaw_rate = 0.0
        auto_mix = float(np.clip(1.0 - self.inspect_manual_yaw_mix, 0.0, 1.0))
        manual_mix = float(np.clip(self.inspect_manual_yaw_mix, 0.0, 1.0))

        if float(getattr(self, "visual_gate_last_blocked", 0.0)) > 0.5 or yaw_error_abs > 0.45:
            auto_mix = max(auto_mix, 0.78)
            manual_mix = min(manual_mix, 0.22)

        # Не "финальная", а любая текущая inspect-точка в ближней фазе.
        if inspect_lock_zone and yaw_error_abs > 0.18:
            auto_mix = max(auto_mix, 0.90)
            manual_mix = min(manual_mix, 0.10)
        elif inspect_turn_zone and yaw_error_abs > 0.12:
            auto_mix = max(auto_mix, 0.84)
            manual_mix = min(manual_mix, 0.16)

        yaw_rate_cmd_raw = auto_mix * auto_yaw_rate + manual_mix * manual_yaw_rate
        yaw_rate_cmd_raw = float(np.clip(yaw_rate_cmd_raw, -auto_yaw_limit_deg, auto_yaw_limit_deg))


        if self.altitude_pd_enabled and self.active_local_waypoint is not None:
            z_offset = float(action[2]) * self.altitude_action_z_offset

            # КРИТИЧНО:
            # Z берём от настоящего route_point, а не от active_local_waypoint.
            # active_local_waypoint может быть промежуточной точкой local_path
            # и при просадке дрона начинает уводить высоту вниз.
            route_z = float(self._get_current_route_z_target())
            target_z = route_z + z_offset

            # На старте запрещаем снижать целевую высоту ниже route_z.
            # AirSim NED: больше z => ниже, меньше z => выше.
            if int(getattr(self, "current_step", 0)) < 220:
                target_z = min(float(target_z), float(route_z))
            else:
                target_z = min(float(target_z), float(route_z + 0.25))
            try:
                down_dist = float(self._get_distance_sensor("DownDistance"))
            except Exception:
                down_dist = self.max_sensor_distance

            min_ground_clearance = 1.3

            if np.isfinite(down_dist) and down_dist < min_ground_clearance:
                # В AirSim NED меньше z => выше.
                lift_need = float(min_ground_clearance - down_dist + 0.4)
                safe_target_z = float(current_z - lift_need)
                target_z = min(float(target_z), safe_target_z)
            # Первые шаги после reset: не разрешаем резко снижаться.
            # AirSim NED: больше z => ниже.
            if int(getattr(self, "current_step", 0)) < 45:
                max_start_descent_m = 0.25
                target_z = min(float(target_z), float(current_z + max_start_descent_m))
            z_error = target_z - current_z
            if abs(z_error) < self.altitude_deadband_m:
                z_error = 0.0

            vz_cmd_raw = self.altitude_kp * z_error - self.altitude_kd * current_vz
            vz_limit = self.max_vertical_speed * 0.85
            vz_cmd_raw = float(np.clip(vz_cmd_raw, -vz_limit, vz_limit))

            self.last_target_z = float(target_z)
            self.last_z_error = float(z_error)

        else:
            vz_raw = float(action[2])
            if abs(vz_raw) < 0.10:
                vz_raw = 0.0
            vz_cmd_raw = float(vz_raw * self.max_vertical_speed)
            self.last_target_z = float(current_z)
            self.last_z_error = 0.0



        control_vis = self._get_control_visual_features()

        visible_now = float(control_vis[0]) if control_vis.shape[0] > 0 else 0.0
        dx_center_now = float(control_vis[6]) if control_vis.shape[0] > 6 else 0.0
        dy_center_now = float(control_vis[7]) if control_vis.shape[0] > 7 else 0.0

        center_error_now = float(np.clip(np.sqrt(dx_center_now ** 2 + dy_center_now ** 2), 0.0, 1.5))
        _, _, _, _, standoff_error_raw = self._get_inspection_metrics()
        abs_standoff_error_now = abs(float(standoff_error_raw))


        # Геометрическая защита у inspect-waypoint:
        # работает даже если YOLO временно пропал.
        route_close_zone = (
                point_mode == "inspect"
                and local_distance_raw <= (self.waypoint_tolerance * 1.25)
        )

        route_lock_zone = (
                point_mode == "inspect"
                and local_distance_raw <= (self.waypoint_tolerance * 0.95)
        )
        gate_visible_thr = float(meta.get("visual_gate_visible_threshold", self.visual_gate_visible_threshold))
        gate_center_thr = float(meta.get("visual_gate_center_threshold", self.visual_gate_center_threshold))
        gate_yaw_thr = float(meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))

        near_hold_zone = (
            point_mode == "inspect"
            and visible_now > gate_visible_thr
            and center_error_now < min(gate_center_thr + 0.12, 0.40)
            and abs_standoff_error_now < (self.standoff_tolerance * 1.5)
        )

        almost_ready_zone = (
            point_mode == "inspect"
            and visible_now > gate_visible_thr
            and center_error_now < min(gate_center_thr + 0.06, 0.34)
            and yaw_error_abs < (gate_yaw_thr + 0.10)
            and abs_standoff_error_now < (self.standoff_tolerance * 1.25)
        )
        # ===== INSPECT XY DYNAMIC LIMITS =====
        # Дальние переходы: быстрее.
        # Близко к waypoint: аккуратнее, чтобы не пролетать точку.

        gate_blocked = float(getattr(self, "visual_gate_last_blocked", 0.0))

        try:
            current_wp_for_xy_limit = np.asarray(
                self.current_route_points[int(self.current_waypoint_idx)],
                dtype=np.float32,
            )
        except Exception:
            if self.active_local_waypoint is not None:
                current_wp_for_xy_limit = np.asarray(self.active_local_waypoint, dtype=np.float32)
            else:
                current_wp_for_xy_limit = np.array(
                    [float(pos.x_val), float(pos.y_val), float(pos.z_val)],
                    dtype=np.float32,
                )

        to_wp_xy_limit = current_wp_for_xy_limit[:2] - np.array(
            [float(pos.x_val), float(pos.y_val)],
            dtype=np.float32,
        )

        route_xy_dist_for_limit = float(np.linalg.norm(to_wp_xy_limit))

        base_xy_speed_limit = float(self.inspect_xy_speed_limit)
        base_xy_delta_limit = float(self.inspect_xy_delta_limit)

        # Базовые значения:
        # раньше почти всегда было около 1.5 / 0.45.
        # На дальних участках поднимаем до 2.4 / 0.70.
        if (
                route_xy_dist_for_limit > 2.0
                and not bool(route_close_zone)
                and not bool(route_lock_zone)
                and not bool(near_hold_zone)
                and not bool(almost_ready_zone)
        ):
            xy_speed_limit = max(base_xy_speed_limit, 1.80)
            xy_delta_limit = max(base_xy_delta_limit, 0.32)

        elif route_xy_dist_for_limit > 1.0:
            xy_speed_limit = max(base_xy_speed_limit, 1.35)
            xy_delta_limit = max(base_xy_delta_limit, 0.28)

        else:
            xy_speed_limit = base_xy_speed_limit
            xy_delta_limit = base_xy_delta_limit

        yaw_delta_limit = float(self.max_delta_yaw_rate_cmd)

        # Если visual gate блокирует проход, слегка душим XY.
        if gate_blocked > 0.5:
            xy_speed_limit *= 0.70
            xy_delta_limit *= 0.85

        # В зоне поворота не режем XY слишком сильно.
        if inspect_turn_zone:
            xy_speed_limit *= 1.00
            xy_delta_limit *= 0.98
            yaw_delta_limit *= 1.08

            if yaw_error_abs > 0.30:
                xy_speed_limit *= 0.97
                xy_delta_limit *= 0.97

        # Близко к route_point — начинаем замедляться.
        if route_close_zone:
            xy_speed_limit *= 0.96
            xy_delta_limit *= 0.95
            yaw_delta_limit *= 1.15

            if yaw_error_abs > max(gate_yaw_thr + 0.06, 0.30):
                xy_speed_limit *= 0.96
                xy_delta_limit *= 0.96

        # Почти у точки — аккуратно.
        if route_lock_zone:
            xy_speed_limit *= 0.88
            xy_delta_limit *= 0.90
            yaw_delta_limit *= 1.18

        if near_hold_zone:
            xy_speed_limit *= 0.78
            xy_delta_limit *= 0.82
            yaw_delta_limit *= 1.22

        if almost_ready_zone:
            xy_speed_limit *= 0.68
            xy_delta_limit *= 0.74
            yaw_delta_limit *= 1.26


        vx_cmd_raw = float(ax * xy_speed_limit)
        vy_cmd_raw = float(ay * xy_speed_limit)

        vz_cmd = self._ema_rate_limit(
            self.filtered_vz_cmd, vz_cmd_raw, self.cmd_filter_alpha_z, self.max_delta_vz_cmd
        )

        yaw_rate_cmd = self._ema_rate_limit(
            self.filtered_yaw_rate_cmd, yaw_rate_cmd_raw, self.cmd_filter_alpha_yaw, yaw_delta_limit
        )

        front_dist = float(
            self._get_distance_sensor("FrontDistance") / max(self.max_sensor_distance, 1e-6)
        )
        left_dist = float(
            self._get_distance_sensor("LeftDistance") / max(self.max_sensor_distance, 1e-6)
        )
        right_dist = float(
            self._get_distance_sensor("RightDistance") / max(self.max_sensor_distance, 1e-6)
        )

        forward_dir = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        lateral_dir = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        v_xy_raw_world = np.array([vx_cmd_raw, vy_cmd_raw], dtype=np.float32)

        vx_body_target = float(np.dot(v_xy_raw_world, forward_dir))
        vy_body_target = float(np.dot(v_xy_raw_world, lateral_dir))

        # vx_body_cmd = self._ema_rate_limit(
        #     self.filtered_vx_cmd, vx_body_target, self.inspect_xy_filter_alpha, xy_delta_limit
        # )
        # vy_body_cmd = self._ema_rate_limit(
        #     self.filtered_vy_cmd, vy_body_target, self.inspect_xy_filter_alpha, xy_delta_limit
        # )
        # v_xy = forward_speed_cmd * forward_dir + lateral_speed_cmd * lateral_dir
        #
        # forward_speed_cmd = float(vx_body_cmd)
        # lateral_speed_cmd = float(vy_body_cmd)
        vx_world_filtered = self._ema_rate_limit(
            self.filtered_vx_cmd,
            vx_cmd_raw,
            self.inspect_xy_filter_alpha,
            xy_delta_limit,
        )

        vy_world_filtered = self._ema_rate_limit(
            self.filtered_vy_cmd,
            vy_cmd_raw,
            self.inspect_xy_filter_alpha,
            xy_delta_limit,
        )

        v_xy = np.array([vx_world_filtered, vy_world_filtered], dtype=np.float32)


        forward_speed_cmd = float(np.dot(v_xy, forward_dir))
        lateral_speed_cmd = float(np.dot(v_xy, lateral_dir))
        # route_close_zone ещё не повод душить боковой проход.
        protect_xy_zone = bool(near_hold_zone or almost_ready_zone or inspect_lock_zone)
        soft_front = 0.22 if protect_xy_zone else 0.16
        hard_front = 0.11 if protect_xy_zone else 0.08

        forward_speed_cmd = float(np.dot(v_xy, forward_dir))
        if forward_speed_cmd > 0.0 and front_dist < soft_front:
            k_front = float(np.clip(
                (front_dist - hard_front) / max(soft_front - hard_front, 1e-6),
                0.0,
                1.0,
            ))
            allowed_forward = forward_speed_cmd * k_front
            v_xy = v_xy - (forward_speed_cmd - allowed_forward) * forward_dir

        is_transit_point = (point_mode != "inspect")
        # В near-hold режиме дополнительно режем боковой дрейф.
        if protect_xy_zone:
            side_soft = 0.16
            side_hard = 0.08

            lateral_speed_cmd = float(np.dot(v_xy, lateral_dir))

            # lateral_speed_cmd > 0 -> движение в right_dir
            if lateral_speed_cmd > 0.0 and right_dist < side_soft:
                k_right = float(np.clip(
                    (right_dist - side_hard) / max(side_soft - side_hard, 1e-6),
                    0.0,
                    1.0,
                ))
                allowed_lateral = lateral_speed_cmd * k_right
                v_xy = v_xy - (lateral_speed_cmd - allowed_lateral) * lateral_dir

            # lateral_speed_cmd < 0 -> движение в left_dir
            lateral_speed_cmd = float(np.dot(v_xy, lateral_dir))
            if lateral_speed_cmd < 0.0 and left_dist < side_soft:
                k_left = float(np.clip(
                    (left_dist - side_hard) / max(side_soft - side_hard, 1e-6),
                    0.0,
                    1.0,
                ))
                allowed_lateral = lateral_speed_cmd * k_left
                v_xy = v_xy - (lateral_speed_cmd - allowed_lateral) * lateral_dir

            min_guard_dist = float(min(front_dist, left_dist, right_dist))
            if min_guard_dist < 0.14:
                k_all = float(np.clip(
                    (min_guard_dist - 0.07) / max(0.14 - 0.07, 1e-6),
                    0.0,
                    1.0,
                ))
                v_xy *= k_all

        # ===== softened continuous yaw/lateral scaling =====
        if (near_hold_zone or almost_ready_zone or inspect_lock_zone):
            forward_speed_cmd = float(np.dot(v_xy, forward_dir))
            lateral_speed_cmd = float(np.dot(v_xy, lateral_dir))

            gate_soft = max(gate_yaw_thr + 0.02, 0.22)
            gate_hard = max(gate_yaw_thr + 0.14, 0.34)

            if yaw_error_abs <= gate_soft:
                yaw_ok = 1.0
            elif yaw_error_abs >= gate_hard:
                yaw_ok = 0.0
            else:
                yaw_ok = float((gate_hard - yaw_error_abs) / max(gate_hard - gate_soft, 1e-6))

            if inspect_lock_zone:
                # В финальной фазе всё ещё держим дрон аккуратнее,
                # но не выжигаем боковой ход почти до нуля.
                forward_floor = 0.26
                lateral_floor = 0.12

                forward_pow = 1.20
                lateral_pow = 1.55

                xy_delta_limit *= (0.82 + 0.18 * yaw_ok)
                yaw_delta_limit *= (1.10 + 0.30 * (1.0 - yaw_ok))
            else:
                # На подходе сохраняем и forward, и lateral заметно живее.
                forward_floor = 0.50
                lateral_floor = 0.24

                forward_pow = 1.05
                lateral_pow = 1.30

                xy_delta_limit *= (0.92 + 0.08 * yaw_ok)
                yaw_delta_limit *= (1.04 + 0.18 * (1.0 - yaw_ok))

            forward_scale = float(
                forward_floor + (1.0 - forward_floor) * (yaw_ok ** forward_pow)
            )
            lateral_scale = float(
                lateral_floor + (1.0 - lateral_floor) * (yaw_ok ** lateral_pow)
            )

            forward_speed_cmd *= forward_scale
            lateral_speed_cmd *= lateral_scale

            v_xy = forward_speed_cmd * forward_dir + lateral_speed_cmd * lateral_dir


        # Финальную команду переводим в body frame дрона
        # Финальная команда в WORLD frame, как в рабочем PPO_nav.
        # v_xy уже находится в world frame.
        vx_world_cmd = float(v_xy[0])
        vy_world_cmd = float(v_xy[1])
        # ===== INSPECT XY CRUISE ASSIST =====
        # Если waypoint далеко по XY, добавляем устойчивую маршевую компоненту
        # в сторону текущей route_point. Это убирает stop-go / пилу на длинных
        # переходах и не трогает Z-контур.
        try:
            current_wp_for_xy = np.asarray(
                self.current_route_points[int(self.current_waypoint_idx)],
                dtype=np.float32,
            )
        except Exception:
            current_wp_for_xy = np.asarray(self.active_local_waypoint, dtype=np.float32)

        to_wp_xy = current_wp_for_xy[:2] - np.array(
            [float(pos.x_val), float(pos.y_val)],
            dtype=np.float32,
        )

        route_xy_dist = float(np.linalg.norm(to_wp_xy))

        if route_xy_dist > 1e-6:
            route_dir_xy = to_wp_xy / route_xy_dist
        else:
            route_dir_xy = np.zeros(2, dtype=np.float32)

        try:
            cable_state_step = self._get_cable_safety_state(mutate=True)
        except Exception:
            cable_state_step = {
                "visible": 0.0,
                "ratio": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "center_dist": 1.0,
                "stability": 0.0,
                "soft": False,
                "hard": False,
                "hold_left": int(getattr(self, "_cable_hold_left", 0)),
            }

        try:
            cable_ratio_for_route = float(cable_state_step.get("ratio", 0.0))
            cable_center_for_route = float(cable_state_step.get("center_dist", 1.0))

            # ВАЖНО:
            # visible=1.0 НЕ является причиной отключать cruise assist.
            # Для инспекции ЛЭП кабель почти всегда в кадре.
            cable_active_for_route = (
                bool(cable_state_step.get("hard", False))
                or (
                    bool(cable_state_step.get("soft", False))
                    and cable_center_for_route < float(self.cable_center_soft)
                )
                or (
                    cable_ratio_for_route > float(self.cable_hard_ratio)
                    and cable_center_for_route < float(self.cable_center_soft)
                )
            )
        except Exception:
            cable_active_for_route = False

        xy_far_move = (
                route_xy_dist > float(self.inspect_xy_far_dist)

                and not bool(near_hold_zone)
                and not bool(almost_ready_zone)
                and not bool(route_lock_zone)
                and float(gate_blocked) < 0.5

                # Если кабель виден или cable safety уже держит soft/hard,
                # запрещаем cruise-assist тянуть дрон через кабель.
                and not bool(cable_active_for_route)
        )

        if xy_far_move:
            current_xy_cmd = np.array(
                [float(vx_world_cmd), float(vy_world_cmd)],
                dtype=np.float32,
            )

            forward_speed = float(np.dot(current_xy_cmd, route_dir_xy))

            # Если policy даёт слишком слабую или обратную команду,
            # мягко подтягиваем её в сторону waypoint.
            min_forward_speed = float(self.inspect_xy_cruise_speed)

            if forward_speed < min_forward_speed:
                add_speed = min_forward_speed - forward_speed
                current_xy_cmd = current_xy_cmd + (
                        float(self.inspect_xy_route_assist_gain) * add_speed * route_dir_xy
                )

            vx_world_cmd = float(current_xy_cmd[0])
            vy_world_cmd = float(current_xy_cmd[1])
        # =========================================================
        # FINAL EXECUTOR: real command send
        # + emergency vertical velocity capture
        # =========================================================

        command_duration = self._get_inspect_command_duration()
        self.last_command_duration = float(command_duration)

        if int(getattr(self, "current_step", 0)) < 20:
            try:
                self.client.enableApiControl(True, self.vehicle_name)
                self.client.armDisarm(True, self.vehicle_name)
            except Exception:
                pass

        command_z = float(getattr(self, "last_target_z", current_z))
        if not np.isfinite(command_z):
            command_z = float(current_z)

        route_z_for_capture = float(self._get_current_route_z_target())

        self._debug_approach_turn_zone = float(approach_turn_zone)
        self._debug_inspect_turn_zone = float(inspect_turn_zone)
        self._debug_inspect_lock_zone = float(inspect_lock_zone)

        # ВАЖНО:
        # Не переключаемся на moveByVelocityAsync(vz=...).
        # Для inspect держим единый vertical-mode через moveByVelocityZAsync(z=...).
        z_below_route_m = float(current_z) - float(route_z_for_capture)
        step_i = int(getattr(self, "current_step", 0))

        # Emergency Z-hold нужен только сразу после reset/старта,
        # а не на каждом новом waypoint. Иначе при переходе на точку
        # с более отрицательной Z он глушит XY и дрон висит/лезет только по высоте.
        wp_i = int(getattr(self, "current_waypoint_idx", 0))

        try:
            start_xy = np.asarray(self.start_position[:2], dtype=np.float32)
            cur_xy = np.asarray([float(pos.x_val), float(pos.y_val)], dtype=np.float32)
            xy_from_start = float(np.linalg.norm(cur_xy - start_xy))
        except Exception:
            xy_from_start = 999.0

        emergency_z_hold = (
                wp_i == 0
                and step_i < 80
                and xy_from_start < 3.0
                and (
                        z_below_route_m > 1.15
                        or (
                                z_below_route_m > 0.65
                                and float(current_vz) > 0.15
                        )
                )
        )

        # ===== FIX: diagnostic/applied VZ value for observation/logging =====
        # Реальная вертикаль исполняется через moveByVelocityZAsync(z=command_z),
        # а это значение нужно только для applied_cmd_obs / логов.
        vz_for_obs_cmd = float(np.clip(
            float(vz_cmd),
            -float(self.max_vertical_speed),
            float(self.max_vertical_speed),
        ))

        if emergency_z_hold:
            command_z = float(route_z_for_capture)

            # Диагностическая вертикальная команда для applied_cmd_obs/log.
            # Реальное удержание высоты всё равно идёт через moveByVelocityZAsync(z=command_z).
            recovery_vz = float(np.clip(
                0.75 + 0.35 * float(z_below_route_m),
                0.75,
                min(float(self.max_vertical_speed), 1.6),
            ))
            vz_for_obs_cmd = -recovery_vz

            # ВАЖНО:
            # Не обнуляем XY. Иначе при большом перепаде Z дрон зависает по XY
            # и только набирает/сбрасывает высоту.
            emergency_xy_speed_limit = min(float(xy_speed_limit), 0.45)

            vxy_norm = float(np.hypot(float(vx_world_cmd), float(vy_world_cmd)))
            if vxy_norm > emergency_xy_speed_limit:
                scale = emergency_xy_speed_limit / max(vxy_norm, 1e-6)
                vx_world_cmd = float(vx_world_cmd * scale)
                vy_world_cmd = float(vy_world_cmd * scale)

            # Yaw тоже не гасим в ноль, только ограничиваем.
            yaw_rate_cmd = float(np.clip(float(yaw_rate_cmd), -3.0, 3.0))

            self.filtered_vx_cmd = float(vx_world_cmd)
            self.filtered_vy_cmd = float(vy_world_cmd)
            self.filtered_vz_cmd = float(vz_for_obs_cmd)
            self.filtered_yaw_rate_cmd = float(yaw_rate_cmd)

        else:
            self.filtered_vx_cmd = float(vx_world_cmd)
            self.filtered_vy_cmd = float(vy_world_cmd)
            self.filtered_vz_cmd = float(vz_for_obs_cmd)
            self.filtered_yaw_rate_cmd = float(yaw_rate_cmd)



        try:
            command_z = float(self._get_current_route_z_target())
        except Exception:
            command_z = float(getattr(self, "last_target_z", current_z))

        self.last_target_z = float(command_z)
        self.last_z_error = float(command_z - current_z)

        # applied_cmd_obs/logs: это только диагностическое значение,
        # реальная вертикаль исполняется через command_z.
        try:
            vz_world_cmd = float(vz_for_obs_cmd)
        except NameError:
            # fallback на случай, если в текущей версии переменная названа иначе
            vz_world_cmd = float(vz_for_obs_cmd)

        vx_send = float(vx_world_cmd)
        vy_send = float(vy_world_cmd)
        vz_send = float(vz_world_cmd)
        yaw_rate_send = float(yaw_rate_cmd)

        cable_state = cable_state_step

        cable_visible = float(cable_state.get("visible", 0.0)) > 0.5
        cable_soft = bool(cable_state.get("soft", False))
        cable_hard = bool(cable_state.get("hard", False))
        cable_ratio = float(cable_state.get("ratio", 0.0))
        cable_center_dist = float(cable_state.get("center_dist", 1.0))

        # ВАЖНО:
        # Для инспекции ЛЭП visible=True — это нормальный режим, а не авария.
        # Активная safety-коррекция нужна только при soft/hard или если кабель реально
        # близко к центру/занимает опасную площадь.
        cable_active = (
                cable_hard
                or cable_soft
                or (
                        cable_visible
                        and cable_ratio > float(self.cable_soft_ratio)
                        and cable_center_dist < float(self.cable_center_soft)
                )
        )

        if getattr(self, "cable_safety_enabled", False) and cable_active:
            cable_dx = float(cable_state.get("dx", 0.0))

            yaw_now = float(getattr(self, "last_current_yaw", 0.0))

            forward_dir = np.array([np.cos(yaw_now), np.sin(yaw_now)], dtype=np.float32)
            right_dir = np.array([-np.sin(yaw_now), np.cos(yaw_now)], dtype=np.float32)

            # dx < 0: кабель слева в кадре, безопасное направление — вправо.
            # dx > 0: кабель справа в кадре, безопасное направление — влево.
            away_sign = -np.sign(cable_dx)

            if abs(cable_dx) < 0.04:
                away_sign = float(getattr(self, "_last_cable_away_sign", 1.0))
            else:
                self._last_cable_away_sign = float(away_sign)

            vxy_now = np.array([float(vx_send), float(vy_send)], dtype=np.float32)

            lateral_now = float(np.dot(vxy_now, right_dir))
            forward_now = float(np.dot(vxy_now, forward_dir))

            # =====================================================
            # 1) Главное правило:
            # НЕ толкаем дрон боком от кабеля.
            # Только убираем боковое движение В СТОРОНУ кабеля.
            # =====================================================
            moving_toward_cable = (lateral_now * away_sign) < 0.0

            if moving_toward_cable:
                vxy_now = vxy_now - lateral_now * right_dir

            # =====================================================
            # 2) Soft:
            # мягко ограничиваем forward только если кабель близко к центру.
            # Если кабель просто сбоку в кадре — вдоль линии лететь можно.
            # =====================================================
            if cable_soft and cable_center_dist < float(self.cable_center_soft):
                forward_now = float(np.dot(vxy_now, forward_dir))

                if forward_now > 0.0:
                    vxy_now = vxy_now - 0.25 * forward_now * forward_dir

                yaw_rate_send *= 0.85

            # =====================================================
            # 3) Hard:
            # реально опасная зона — сильно ограничиваем forward.
            # =====================================================
            if cable_hard:
                forward_now = float(np.dot(vxy_now, forward_dir))

                if forward_now > 0.0:
                    vxy_now = vxy_now - 0.70 * forward_now * forward_dir

                yaw_rate_send *= 0.55

            vx_send = float(vxy_now[0])
            vy_send = float(vxy_now[1])

        # ===== CABLE SPEED CAP =====
        # Не режем скорость просто при visible=True.
        # Режем только при soft/hard.
        if getattr(self, "cable_safety_enabled", False):
            cable_vxy_cap = None

            if cable_hard:
                cable_vxy_cap = 0.45
            elif cable_soft and cable_center_dist < float(self.cable_center_soft):
                cable_vxy_cap = 0.95

            if cable_vxy_cap is not None:
                vxy_norm_cable = float(np.linalg.norm([vx_send, vy_send]))

                if vxy_norm_cable > cable_vxy_cap:
                    scale = cable_vxy_cap / max(vxy_norm_cable, 1e-6)
                    vx_send *= scale
                    vy_send *= scale
        xy_norm_after_cable = float(np.linalg.norm([vx_send, vy_send]))
        if xy_norm_after_cable > float(xy_speed_limit):
            scale = float(xy_speed_limit) / max(xy_norm_after_cable, 1e-6)
            vx_send *= scale
            vy_send *= scale

        vx_send = float(np.nan_to_num(vx_send, nan=0.0, posinf=0.0, neginf=0.0))
        vy_send = float(np.nan_to_num(vy_send, nan=0.0, posinf=0.0, neginf=0.0))
        vz_send = float(np.nan_to_num(vz_send, nan=0.0, posinf=0.0, neginf=0.0))
        yaw_rate_send = float(np.nan_to_num(yaw_rate_send, nan=0.0, posinf=0.0, neginf=0.0))
        # Обновляем applied-команды, которые потом идут в observation/debug.
        self.last_applied_vx_cmd = vx_send
        self.last_applied_vy_cmd = vy_send
        self.last_applied_vz_cmd = vz_send
        self.last_applied_yaw_rate_cmd = yaw_rate_send

        # Это оставляем только как справочную целевую Z, не как команду AirSim.
        if hasattr(self, "last_applied_z_target_cmd"):
            self.last_applied_z_target_cmd = float(command_z)

        stream_alive = (
                getattr(self, "_control_stream_thread", None) is not None
                and self._control_stream_thread.is_alive()
        )

        last_stream_send = float(getattr(self, "_control_stream_last_send_time", 0.0))
        stream_dt = -1.0
        if last_stream_send > 0.0:
            stream_dt = time.perf_counter() - last_stream_send



        if getattr(self, "control_stream_enabled", False):
            self._publish_control_stream_command(
                vx=float(vx_send),
                vy=float(vy_send),
                vz=float(vz_send),
                yaw_rate=float(yaw_rate_send),
            )

            # ===== CABLE / STREAM DEBUG =====
            if (
                    int(getattr(self, "current_step", 0)) % 25 == 0
                    or (
                    float(cable_state.get("visible", 0.0)) > 0.5
                    and not bool(cable_state.get("soft", False))
                    and not bool(cable_state.get("hard", False))
            )
            ):
                stream_alive = (
                        getattr(self, "_control_stream_thread", None) is not None
                        and self._control_stream_thread.is_alive()
                )

                last_stream_send = float(getattr(self, "_control_stream_last_send_time", 0.0))
                stream_dt = -1.0
                if last_stream_send > 0.0:
                    stream_dt = time.perf_counter() - last_stream_send

                last_cmd_time = float(getattr(self, "_latest_stream_cmd_time", 0.0))
                latest_cmd_age = -1.0
                if last_cmd_time > 0.0:
                    latest_cmd_age = time.perf_counter() - last_cmd_time

                latest_cmd = getattr(
                    self,
                    "_latest_stream_cmd",
                    np.zeros(4, dtype=np.float32),
                ).copy()


        else:
            try:
                self._last_move_task = self.client.moveByVelocityAsync(
                    vx=float(vx_send),
                    vy=float(vy_send),
                    vz=float(vz_send),
                    duration=float(command_duration),
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate_send)),
                    vehicle_name=self.vehicle_name,
                )
            except Exception as exc:
                print("[AIRSIM CMD ERROR] moveByVelocityAsync failed:", repr(exc))

        policy_raw_vec = np.zeros(4, dtype=np.float32)
        policy_raw_src = np.asarray(
            getattr(self, "_last_policy_raw_action", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        n_policy_raw = min(4, int(policy_raw_src.size))
        if n_policy_raw > 0:
            policy_raw_vec[:n_policy_raw] = policy_raw_src[:n_policy_raw]

        last_cmd_time = float(getattr(self, "_latest_stream_cmd_time", 0.0))
        command_age = -1.0
        if last_cmd_time > 0.0:
            command_age = time.perf_counter() - last_cmd_time

        spark_action_override_changed = bool(
            np.any(np.abs(action_after_override_vec - action_to_apply_vec) > 1e-6)
        )
        self._last_spark_trace_diag = {
            "spark_override_active_dbg": float(bool(spark_override_active)),
            "spark_action_override_changed_dbg": float(spark_action_override_changed),
            "spark_action_to_apply_0_dbg": float(action_to_apply_vec[0]),
            "spark_action_to_apply_1_dbg": float(action_to_apply_vec[1]),
            "spark_action_to_apply_2_dbg": float(action_to_apply_vec[2]),
            "spark_action_to_apply_3_dbg": float(action_to_apply_vec[3]),
            "spark_action_after_override_0_dbg": float(action_after_override_vec[0]),
            "spark_action_after_override_1_dbg": float(action_after_override_vec[1]),
            "spark_action_after_override_2_dbg": float(action_after_override_vec[2]),
            "spark_action_after_override_3_dbg": float(action_after_override_vec[3]),
        }

        if self.enable_heavy_diagnostics:
            self._last_action_apply_diag = {
                "raw_action_0": float(policy_raw_vec[0]),
                "raw_action_1": float(policy_raw_vec[1]),
                "raw_action_2": float(policy_raw_vec[2]),
                "raw_action_3": float(policy_raw_vec[3]),
                "action_to_apply_0": float(action_to_apply_vec[0]),
                "action_to_apply_1": float(action_to_apply_vec[1]),
                "action_to_apply_2": float(action_to_apply_vec[2]),
                "action_to_apply_3": float(action_to_apply_vec[3]),
                "action_after_override_0": float(action_after_override_vec[0]),
                "action_after_override_1": float(action_after_override_vec[1]),
                "action_after_override_2": float(action_after_override_vec[2]),
                "action_after_override_3": float(action_after_override_vec[3]),
                "vx_cmd_raw": float(vx_cmd_raw),
                "vy_cmd_raw": float(vy_cmd_raw),
                "vz_cmd_raw": float(vz_cmd_raw),
                "yaw_rate_cmd_raw": float(yaw_rate_cmd_raw),
                "vx_send": float(vx_send),
                "vy_send": float(vy_send),
                "vz_send": float(vz_send),
                "yaw_rate_send": float(yaw_rate_send),
                "last_applied_vx": float(self.last_applied_vx_cmd),
                "last_applied_vy": float(self.last_applied_vy_cmd),
                "last_applied_vz": float(self.last_applied_vz_cmd),
                "last_applied_yaw_rate": float(self.last_applied_yaw_rate_cmd),
                "cable_safety_active": float(bool(cable_active)),
                "emergency_z_hold_active": float(bool(emergency_z_hold)),
                "visual_hazard_active": float(getattr(self, "visual_hazard_active", False)),
                "spark_override_active": float(bool(spark_override_active)),
                "stream_alive": float(bool(stream_alive)),
                "command_age": float(command_age),
            }
        else:
            self._last_action_apply_diag = {}

        time.sleep(float(self.step_duration))

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray, waypoint_reached: bool = False):
        reward = 0.0
        info = {}
        heavy_diag = bool(getattr(self, "enable_heavy_diagnostics", False))
        pos = self._current_position_vec()
        target = self._get_inspection_target()
        desired_standoff_current = float(self._get_pair_desired_standoff())

        delta = target - pos
        dist = float(np.linalg.norm(delta))
        standoff_error = dist - desired_standoff_current
        abs_standoff_error = abs(standoff_error)

        nav = self._decode_nav_obs(obs)

        local_distance = nav["local_distance_norm"]
        final_distance = self._get_final_distance_norm_raw()
        altitude_norm = nav["altitude_norm"]
        local_dz = nav["local_dz"]
        route_deviation_norm = nav["route_deviation_norm"]

        depth_left = nav["depth_left"]
        depth_center = nav["depth_center"]
        depth_right = nav["depth_right"]
        depth_up = nav["depth_up"]

        collision_flag = nav["collision"]
        velocity_vec = np.array([nav["vx"], nav["vy"], nav["vz"]], dtype=np.float32)
        speed_norm = float(np.linalg.norm(velocity_vec))
        action_norm = float(np.linalg.norm(action))

        total_risk = float(np.mean([
            nav["risk_front"],
            nav["risk_left"],
            nav["risk_right"],
            nav["risk_up"],
            nav["risk_down"],
        ]))
        min_obstacle_dist = min(depth_center, depth_left, depth_right, depth_up)
        yolo_visible = float(self.yolo_features[0])
        yolo_cx = float(self.yolo_features[3])
        yolo_cy = float(self.yolo_features[4])

        _, _, _, inspect_distance_raw, standoff_error_raw = self._get_inspection_metrics()
        local_distance_raw = self._get_local_distance_raw()


        current_meta = self._get_current_inspection_pair_meta()
        point_mode = str(current_meta.get("point_mode", "inspect")).lower()
        align = self._get_current_alignment_state(final=False)
        inspect_mode = align["mode"]
        visible = align["visible"]
        conf = align["conf"]
        area = align["area"]
        center_error = align["center_error"]
        yaw_error_geom = align["yaw_error_geom"]
        align_score = align["align_score"]
        visible_thr = float(align.get("visible_thr", self.visual_gate_visible_threshold))
        center_thr = float(align.get("center_thr", self.visual_gate_center_threshold))
        yaw_thr = float(align.get("yaw_thr", self.visual_gate_yaw_threshold))
        reward_zone = (
                point_mode == "inspect"
                and local_distance_raw <= (self.waypoint_tolerance * 0.90)
        )
        gate_blocked = bool(inspect_mode == "visual"
                            and self.visual_gate_last_blocked > 0.5)

        can_pay_dense_inspect = (
                point_mode == "inspect"
                and reward_zone
                and not gate_blocked
        )

        vis = self._get_control_visual_features()

        visible = float(vis[0]) if vis.shape[0] > 0 else 0.0
        conf = float(vis[2]) if vis.shape[0] > 2 else 0.0
        area = float(vis[5]) if vis.shape[0] > 5 else 0.0
        dx_center = float(vis[6]) if vis.shape[0] > 6 else 0.0
        dy_center = float(vis[7]) if vis.shape[0] > 7 else 0.0
        center_error = float(np.clip(np.sqrt(dx_center ** 2 + dy_center ** 2), 0.0, 1.5))
        visual_gate_required_now = bool(current_meta.get("visual_gate_required", False))
        visual_ready_now = bool(
            (visible > visible_thr)
            and (center_error < center_thr)
            and (yaw_error_geom < yaw_thr)
        )
        visual_fail_visible = bool(visual_gate_required_now and not (visible > visible_thr))
        visual_fail_center = bool(visual_gate_required_now and not (center_error < center_thr))
        visual_fail_yaw = bool(visual_gate_required_now and not (yaw_error_geom < yaw_thr))
        visual_fail_center_dx = bool(visual_fail_center and abs(dx_center) >= abs(dy_center))
        visual_fail_center_dy = bool(visual_fail_center and abs(dy_center) > abs(dx_center))
        visual_gate_hold_counter = int(getattr(self, "visual_gate_counter", 0))
        visual_gate_hold_required = int(current_meta.get("visual_gate_hold_steps", 3))
        visual_hold_wait = bool(
            visual_gate_required_now
            and visual_ready_now
            and visual_gate_hold_counter < visual_gate_hold_required
        )
        if point_mode == "inspect":
            point_mode_dbg = 1.0
        elif point_mode == "transit":
            point_mode_dbg = 0.0
        else:
            point_mode_dbg = -1.0
        wp_idx_dbg = float(self.current_waypoint_idx)
        # -------------------------------------------------
        # Геометрия траектории инспекции
        # -------------------------------------------------
        local_z_error_penalty = 0.0 if abs(local_dz) < 0.05 else -0.35 * abs(local_dz)
        reward += local_z_error_penalty
        info["local_z_error_penalty"] = local_z_error_penalty

        progress_delta_raw = 0.0
        progress_delta_raw_before_guard = 0.0
        progress_delta_raw_after_guard = 0.0
        progress_delta_raw_clipped = 0.0
        progress_delta_was_clipped = 0.0
        progress_delta = 0.0
        progress_reward = 0.0
        progress_replan_guard_active = 0.0
        current_progress_wp_idx = int(self.current_waypoint_idx)
        current_progress_local_path_index = int(getattr(self, "local_path_index", -1))
        current_progress_local_target = None
        if self.active_local_waypoint is not None:
            current_progress_local_target = np.asarray(self.active_local_waypoint, dtype=np.float32).copy()
        prev_progress_wp_idx = int(getattr(self, "_prev_progress_wp_idx", -1))
        prev_progress_local_path_index = int(getattr(self, "_prev_progress_local_path_index", -1))
        prev_progress_local_target = getattr(self, "_prev_progress_local_target", None)
        same_waypoint_for_progress = (
            prev_progress_wp_idx >= 0
            and current_progress_wp_idx == prev_progress_wp_idx
        )
        local_target_changed_for_progress = False
        if same_waypoint_for_progress:
            if (current_progress_local_target is None) != (prev_progress_local_target is None):
                local_target_changed_for_progress = True
            elif current_progress_local_target is not None and prev_progress_local_target is not None:
                local_target_changed_for_progress = bool(
                    np.linalg.norm(current_progress_local_target - prev_progress_local_target) > 1e-4
                )
            if current_progress_local_path_index != prev_progress_local_path_index:
                local_target_changed_for_progress = True
        if (not waypoint_reached) and local_target_changed_for_progress:
            progress_replan_guard_active = 1.0
            self._local_target_changed_this_step = True
        else:
            self._local_target_changed_this_step = False
        if self.prev_local_distance is not None:
            if waypoint_reached:
                progress_delta_raw = 0.0
                progress_delta_raw_before_guard = 0.0
                progress_delta_raw_after_guard = 0.0
                progress_delta_raw_clipped = 0.0
                progress_delta_was_clipped = 0.0
                progress_delta = 0.0
                progress_reward = 0.0
            else:
                progress_delta_raw_before_guard = float(self.prev_local_distance - local_distance_raw)
                progress_delta_raw = float(progress_delta_raw_before_guard)
                if progress_replan_guard_active > 0.5:
                    progress_delta_raw = 0.0
                    progress_delta_raw_after_guard = 0.0
                    progress_delta_raw_clipped = 0.0
                    progress_delta_was_clipped = 0.0
                    progress_delta = 0.0
                    progress_reward = 0.0
                else:
                    progress_delta_raw_after_guard = float(progress_delta_raw)
                    progress_delta_raw_clipped = float(np.clip(progress_delta_raw, -2.0, 2.0))
                    progress_delta_was_clipped = float(abs(progress_delta_raw - progress_delta_raw_clipped) > 1e-9)
                    progress_delta = float(progress_delta_raw_clipped)
                    progress_reward = progress_delta * 10.0
            reward += progress_reward
        info["progress_delta_raw"] = progress_delta_raw
        info["progress_delta_raw_before_guard"] = progress_delta_raw_before_guard
        info["progress_delta_raw_after_guard"] = progress_delta_raw_after_guard
        info["progress_replan_guard_active"] = progress_replan_guard_active
        info["progress_delta_raw_clipped"] = progress_delta_raw_clipped
        info["progress_delta_was_clipped"] = progress_delta_was_clipped
        info["progress_reward"] = progress_reward

        # ===== route-point progress reward =====
        route_point_progress_reward = 0.0
        route_point_progress_delta_raw = 0.0

        if self.mission_route and 0 <= self.current_waypoint_idx < len(self.mission_route):
            pos_now = self._current_position_vec()
            current_wp = self.mission_route[self.current_waypoint_idx]
            route_point_distance_raw = float(np.linalg.norm(current_wp - pos_now))

            prev_route_point_distance_raw = getattr(self, "_prev_route_point_distance_raw", None)
            if prev_route_point_distance_raw is not None:
                route_point_progress_delta_raw = float(prev_route_point_distance_raw - route_point_distance_raw)

                # очень мягкий shaping, чтобы не перебить основную логику
                route_point_progress_reward = 2.5 * route_point_progress_delta_raw
                route_point_progress_reward = float(np.clip(route_point_progress_reward, -0.05, 0.08))
                reward += route_point_progress_reward

            self._prev_route_point_distance_raw = route_point_distance_raw

            info["route_point_distance_raw"] = route_point_distance_raw
        else:
            self._prev_route_point_distance_raw = None

        info["route_point_progress_delta_raw"] = route_point_progress_delta_raw
        info["route_point_progress_reward"] = route_point_progress_reward
        point_stall_limit_steps = max(
            1,
            int(round(self.point_stall_timeout_sec / max(self.step_duration, 1e-6)))
        )

        near_target_stall_zone = (
                point_mode == "inspect"
                and local_distance_raw <= (self.waypoint_tolerance * 1.15)
                and not waypoint_reached
        )

        gate_holds_point = bool(
            self.disable_point_stall_when_gate_blocked
            and self.visual_gate_last_blocked > 0.5
            and near_target_stall_zone
        )

        if (
                near_target_stall_zone
                and not gate_holds_point
                and abs(progress_delta_raw) < self.point_stall_progress_epsilon
        ):
            self.point_stall_steps += 1
        else:
            self.point_stall_steps = 0
        # Временное ослабление
        point_stall_penalty = 0.0

        info["near_target_stall_zone_dbg"] = float(near_target_stall_zone)
        info["point_stall_steps"] = int(self.point_stall_steps)
        info["point_stall_limit_steps"] = int(point_stall_limit_steps)
        info["point_stall_penalty"] = float(point_stall_penalty)
        info["point_stall_gate_hold_dbg"] = float(gate_holds_point)

        waypoint_reward = 0.0
        if waypoint_reached:
            waypoint_reward = 6.0
            reward += waypoint_reward
        info["waypoint_reward"] = waypoint_reward

        goal_ready_for_reward, goal_dbg = self._is_final_goal_ready()

        goal_reward = 0.0
        if goal_dbg["near_final_goal"] > 0.5 and goal_ready_for_reward:
            goal_reward = 140.0
            reward += goal_reward

        info["goal_ready_for_reward_dbg"] = goal_dbg["goal_ready_for_reward"]
        info["goal_visual_ready"] = goal_dbg["goal_visual_ready"]
        info["goal_geom_ready"] = goal_dbg["goal_geom_ready"]
        info["goal_reward"] = goal_reward
        # -------------------------------------------------
        # Термы, специфичные для инспекции
        # -------------------------------------------------
        standoff_reward = 0.0
        abs_standoff_error = abs(standoff_error_raw)
        tol = max(self.standoff_tolerance, 1e-6)

        if can_pay_dense_inspect:
            if abs_standoff_error <= tol:
                standoff_reward = 0.18 * (1.0 - abs_standoff_error / tol)
            elif abs_standoff_error <= 2.0 * tol:
                standoff_reward = -0.10 * ((abs_standoff_error - tol) / tol)
            else:
                standoff_reward = -0.18 * min(
                    abs_standoff_error / max(desired_standoff_current, 1e-6), 2.0
                )

        reward += standoff_reward
        info["standoff_reward"] = standoff_reward
        info["desired_standoff_current"] = float(desired_standoff_current)

        target_visible_reward = 0.0
        target_center_reward = 0.0
        target_scale_reward = 0.0
        inspection_hold_reward = 0.0

        lost_target_guard_active = self._is_lost_target_guard_active()

        if visible > 0.5:
            self.visible_steps += 1
            self.lost_target_steps = 0
        elif lost_target_guard_active:
            self.lost_target_steps += 1
        else:
            self.lost_target_steps = 0

        info["lost_target_guard_active_dbg"] = float(lost_target_guard_active)
        info["lost_target_steps_dbg"] = int(self.lost_target_steps)

        visual_align_reward = 0.0
        if can_pay_dense_inspect:
            if inspect_mode == "visual":
                scale_error = abs(area - self.desired_bbox_area)
                scale_score = float(max(0.0, 1.0 - scale_error / max(self.desired_bbox_area, 1e-6)))
                visual_align_reward = 0.24 * align_score + 0.06 * scale_score + 0.04 * conf
                reward += self.yolo_reward_scale * visual_align_reward
            elif inspect_mode == "geom":
                visual_align_reward = 0.10 * align_score
                reward += visual_align_reward

        # Маленький бонус только за реально собранное устойчивое состояние.
        inspection_hold_reward = 0.0

        info["target_visible_reward"] = target_visible_reward
        info["target_center_reward"] = target_center_reward
        info["target_scale_reward"] = target_scale_reward
        info["inspection_hold_reward"] = inspection_hold_reward
        info["visual_align_reward"] = float(visual_align_reward)

        info["reward_zone_dbg"] = float(reward_zone)


        camping_penalty = 0.0
        stalled_in_reward_zone = (
                reward_zone
                and not waypoint_reached
                and abs(progress_delta_raw) < 0.003
        )

        # Временное упрощение reward для обучения:
        # camping penalty отключён.
        info["camping_penalty"] = camping_penalty
        info["stalled_in_reward_zone_dbg"] = float(stalled_in_reward_zone)
        info["gate_blocked_dbg"] = float(gate_blocked)
        info["can_pay_dense_inspect_dbg"] = float(can_pay_dense_inspect)

        visual_gate_penalty = 0.0
        # Временное упрощение reward для обучения:
        # visual gate penalty отключён.
        info["visual_gate_penalty"] = visual_gate_penalty


        standoff_ok = bool(abs_standoff_error < self.standoff_tolerance)
        self._update_inspection_lock(
            align_score=float(align_score),
            standoff_ok=standoff_ok,
            gate_blocked=gate_blocked,
        )

        inspect_lock_reward = 0.0


        info["inspect_lock_reward"] = float(inspect_lock_reward)
        info["inspect_lock_dbg"] = float(self.inspect_lock)
        info["inspect_lock_steps_dbg"] = int(self.inspect_lock_steps)

        # -------------------------------------------------
        # Безопасность и стабилизация
        # -------------------------------------------------
        collision_penalty = 0.0
        collision_active_dbg = float(collision_flag > 0.5)
        collision_object_name_dbg = ""
        collision_impact_x_dbg = 0.0
        collision_impact_y_dbg = 0.0
        collision_impact_z_dbg = 0.0
        collision_position_x_dbg = 0.0
        collision_position_y_dbg = 0.0
        collision_position_z_dbg = 0.0
        if heavy_diag:
            try:
                wp_idx_for_collision_dbg = int(getattr(self, "current_waypoint_idx", -1))
                if (collision_flag > 0.5) or (wp_idx_for_collision_dbg in (4, 5)):
                    col_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
                    collision_object_name_dbg = str(getattr(col_info, "object_name", "") or "")
                    impact = getattr(col_info, "impact_point", None)
                    if impact is not None:
                        collision_impact_x_dbg = float(getattr(impact, "x_val", 0.0))
                        collision_impact_y_dbg = float(getattr(impact, "y_val", 0.0))
                        collision_impact_z_dbg = float(getattr(impact, "z_val", 0.0))
                    cpos = getattr(col_info, "position", None)
                    if cpos is not None:
                        collision_position_x_dbg = float(getattr(cpos, "x_val", 0.0))
                        collision_position_y_dbg = float(getattr(cpos, "y_val", 0.0))
                        collision_position_z_dbg = float(getattr(cpos, "z_val", 0.0))
            except Exception:
                pass
        if collision_flag > 0.5:
            collision_penalty = -60.0
            reward += collision_penalty
        info["collision_penalty"] = collision_penalty

        obstacle_penalty = 0.0

        if point_mode == "inspect":
            safe_threshold = 0.24
            obstacle_weight = 0.16
        else:
            safe_threshold = 0.30
            obstacle_weight = 0.35

        if min_obstacle_dist < safe_threshold:
            proximity = 1.0 - (min_obstacle_dist / max(safe_threshold, 1e-6))
            obstacle_penalty = -obstacle_weight * float(proximity ** 2)
            reward += obstacle_penalty


        info["obstacle_penalty"] = obstacle_penalty
        current_meta = self._get_current_inspection_pair_meta()
        point_mode = str(current_meta.get("point_mode", "inspect")).lower()
        hold_visible_thr = float(current_meta.get("visual_gate_visible_threshold", self.visual_gate_visible_threshold))
        hold_center_thr = float(current_meta.get("visual_gate_center_threshold", self.visual_gate_center_threshold))
        hold_yaw_thr = float(current_meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))

        near_hold_zone = (
                point_mode == "inspect"
                and visible > hold_visible_thr
                and center_error < min(hold_center_thr + 0.10, 0.40)
                and yaw_error_geom < (hold_yaw_thr + 0.12)
                and abs_standoff_error < (self.standoff_tolerance * 1.5)
        )

        almost_ready_hold_zone = (
                point_mode == "inspect"
                and visible > hold_visible_thr
                and center_error < min(hold_center_thr + 0.04, 0.32)
                and yaw_error_geom < (hold_yaw_thr + 0.08)
                and abs_standoff_error < (self.standoff_tolerance * 1.25)
        )


        near_hold_motion_penalty = 0.0
        applied_horiz_speed = float(np.linalg.norm([self.last_applied_vx_cmd, self.last_applied_vy_cmd]))


        info["near_hold_zone_dbg"] = float(near_hold_zone)
        info["almost_ready_hold_zone_dbg"] = float(almost_ready_hold_zone)
        info["near_hold_motion_penalty"] = near_hold_motion_penalty
        info["applied_horiz_speed_dbg"] = float(applied_horiz_speed)

        # ===== CAPS/ASAP по реально приложенной команде =====
        current_applied_cmd = self._get_applied_command_vector_norm()

        delta_cmd = current_applied_cmd - self.prev_applied_cmd
        delta_xy = float(np.linalg.norm(delta_cmd[:2]))
        delta_z = float(abs(delta_cmd[2]))
        delta_yaw = float(abs(delta_cmd[3]))

        caps_smooth_penalty = -(
                self.caps_w_xy * (delta_xy ** 2)
                + self.caps_w_z * (delta_z ** 2)
                + self.caps_w_yaw * (delta_yaw ** 2)
        )
        reward += caps_smooth_penalty
        info["caps_smooth_penalty"] = caps_smooth_penalty
        info["delta_xy"] = delta_xy
        info["delta_z"] = delta_z
        info["delta_yaw"] = delta_yaw

        prev_delta_cmd = self.prev_applied_cmd - self.prev_prev_applied_cmd
        jerk_cmd = delta_cmd - prev_delta_cmd

        jerk_xy = float(np.linalg.norm(jerk_cmd[:2]))
        jerk_z = float(abs(jerk_cmd[2]))
        jerk_yaw = float(abs(jerk_cmd[3]))

        asap_smooth_penalty = -(
                self.asap_w_xy * (jerk_xy ** 2)
                + self.asap_w_z * (jerk_z ** 2)
                + self.asap_w_yaw * (jerk_yaw ** 2)
        )
        reward += asap_smooth_penalty
        info["asap_smooth_penalty"] = asap_smooth_penalty
        info["jerk_xy"] = jerk_xy
        info["jerk_z"] = jerk_z
        info["jerk_yaw"] = jerk_yaw

        obstacle_clearance_reward = 0.0
        if not self.disable_obstacle_clearance_reward and self.prev_min_obstacle_dist is not None:
            obstacle_clearance_reward = 1.0 * (min_obstacle_dist - self.prev_min_obstacle_dist)
            obstacle_clearance_reward = float(np.clip(obstacle_clearance_reward, -0.04, 0.04))
            reward += obstacle_clearance_reward
        info["obstacle_clearance_reward"] = obstacle_clearance_reward

        waypoint = self.active_local_waypoint
        mismatch_penalty = 0.0

        if waypoint is not None and point_mode == "inspect" and not self.disable_target_waypoint_mismatch_for_inspect:
            mismatch = float(np.linalg.norm(waypoint - target))
            mismatch_penalty = -0.05 * np.clip(mismatch / 12.0, 0.0, 1.0)
            reward += mismatch_penalty

        target_facing_reward = 0.0
        target_not_facing_penalty = 0.0
        yaw_error_geom = 0.0
        facing_gate = 0.0

        if point_mode == "inspect":
            _, _, target_delta, _, _ = self._get_inspection_metrics()
            desired_yaw_geom = float(np.arctan2(target_delta[1], target_delta[0]))
            yaw_error_geom = abs(self._wrap_angle(desired_yaw_geom - float(self.last_current_yaw)))

            facing_gate = float(
                local_distance_raw <= max(self.waypoint_tolerance * 3.0, self.desired_standoff_distance * 1.20)
                or abs_standoff_error < (self.standoff_tolerance + 0.8)
            )

            if facing_gate > 0.5 and not gate_blocked:
                target_facing_reward = 0.06 * max(0.0, 1.0 - yaw_error_geom / 0.70)
                reward += target_facing_reward

                if yaw_error_geom > 1.25:
                    target_not_facing_penalty = -0.008 * min((yaw_error_geom - 1.25) / 0.75, 1.0)
                    reward += target_not_facing_penalty

        info["target_facing_reward"] = target_facing_reward
        info["target_not_facing_penalty"] = target_not_facing_penalty
        info["target_yaw_error_rad"] = yaw_error_geom
        info["target_facing_gate_dbg"] = float(facing_gate)
        info["target_facing_reward"] = target_facing_reward
        info["target_not_facing_penalty"] = target_not_facing_penalty
        info["target_yaw_error_rad"] = yaw_error_geom
        info["target_facing_gate_dbg"] = float(facing_gate)
        action_penalty = -0.004 * action_norm
        reward += action_penalty
        info["action_penalty"] = action_penalty

        altitude_penalty = 0.0

        if point_mode == "transit":
            if altitude_norm < 0.16:
                ratio = (0.16 - altitude_norm) / 0.16
                altitude_penalty -= 0.08 * float(np.clip(ratio, 0.0, 1.0))
            if altitude_norm < 0.10:
                ratio = (0.10 - altitude_norm) / 0.10
                altitude_penalty -= 0.14 * float(np.clip(ratio, 0.0, 1.0))
            if altitude_norm > 0.96:
                ratio = (altitude_norm - 0.96) / 0.04
                altitude_penalty -= 0.02 * float(np.clip(ratio, 0.0, 1.0))
        else:
            if altitude_norm < 0.20:
                ratio = (0.20 - altitude_norm) / 0.20
                altitude_penalty -= 0.22 * float(np.clip(ratio, 0.0, 1.0))
            if altitude_norm < 0.12:
                ratio = (0.12 - altitude_norm) / 0.12
                altitude_penalty -= 0.35 * float(np.clip(ratio, 0.0, 1.0))
            if altitude_norm > 0.94:
                ratio = (altitude_norm - 0.94) / 0.06
                altitude_penalty -= 0.06 * float(np.clip(ratio, 0.0, 1.0))

        reward += altitude_penalty
        info["altitude_penalty"] = altitude_penalty

        local_z_hold_reward = 0.0
        if abs(local_dz) < 0.08:
            local_z_hold_reward = 0.03 * (1.0 - abs(local_dz) / 0.08)
            reward += local_z_hold_reward
        info["local_z_hold_reward"] = local_z_hold_reward

        forward_facing_reward = 0.0
        info["forward_facing_reward"] = forward_facing_reward

        route_deviation_penalty = -0.08 * route_deviation_norm
        reward += route_deviation_penalty
        info["route_deviation_penalty"] = route_deviation_penalty

        planner_alignment_reward = 0.0
        info["planner_alignment_reward"] = planner_alignment_reward

        if point_mode == "transit":
            risk_penalty = -0.02 * total_risk
        else:
            risk_penalty = -0.045 * total_risk

        reward += risk_penalty
        info["risk_penalty"] = risk_penalty
        inactive_penalty = 0.0
        if (
            speed_norm < self.inactive_speed_threshold
            and action_norm < self.inactive_action_threshold
            and visible < 0.5
        ):
            inactive_penalty = -0.01
            reward += inactive_penalty
        info["inactive_penalty"] = inactive_penalty

        active_progress_reward = 0.0
        if progress_delta_raw > self.no_progress_threshold and speed_norm >= self.inactive_speed_threshold:
            active_progress_reward = self.active_progress_reward_weight
            reward += active_progress_reward
        info["active_progress_reward"] = active_progress_reward

        cable_state = self._get_cable_safety_state(mutate=False)

        cable_safety_penalty = 0.0

        if cable_state["visible"] > 0.5:
            cable_ratio = float(cable_state["ratio"])
            center_dist = float(cable_state["center_dist"])

            # Штраф за большую площадь кабеля в кадре.
            ratio_penalty = np.clip(
                (cable_ratio - self.cable_soft_ratio) / max(self.cable_hard_ratio - self.cable_soft_ratio, 1e-6),
                0.0,
                1.0,
            )

            # Штраф за кабель в центре изображения.
            center_penalty = np.clip(
                (self.cable_center_soft - center_dist) / max(self.cable_center_soft, 1e-6),
                0.0,
                1.0,
            )

            cable_safety_penalty = -0.35 * float(ratio_penalty) - 0.25 * float(center_penalty)

            if cable_state["hard"]:
                cable_safety_penalty -= 0.75

        reward += cable_safety_penalty

        info["cable_safety_penalty"] = float(cable_safety_penalty)
        info["cable_safety_active"] = float(bool(cable_state["soft"] or cable_state["hard"]))
        if heavy_diag:
            info["cable_visible_dbg"] = float(cable_state["visible"])
            info["cable_ratio_dbg"] = float(cable_state["ratio"])
            info["cable_center_dist_dbg"] = float(cable_state["center_dist"])
            info["cable_safety_soft_dbg"] = float(cable_state["soft"])
            info["cable_safety_hard_dbg"] = float(cable_state["hard"])
            info["cable_trigger_ratio_dbg"] = float(cable_state.get("cable_trigger_ratio_dbg", 0.0))
            info["cable_trigger_center_dbg"] = float(cable_state.get("cable_trigger_center_dbg", 0.0))
            info["cable_soft_raw_dbg"] = float(cable_state.get("cable_soft_raw_dbg", 0.0))
            info["cable_hard_raw_dbg"] = float(cable_state.get("cable_hard_raw_dbg", 0.0))
            info["cable_soft_final_dbg"] = float(cable_state.get("cable_soft_final_dbg", 0.0))
            info["cable_hard_final_dbg"] = float(cable_state.get("cable_hard_final_dbg", 0.0))
            info["cable_hold_left_dbg"] = float(cable_state.get("cable_hold_left_dbg", 0.0))
            info["cable_trigger_hold_dbg"] = float(cable_state.get("cable_trigger_hold_dbg", 0.0))

            current_wp_idx_corridor = int(getattr(self, "current_waypoint_idx", -1))
            prev_wp_idx_corridor = int(getattr(self, "_prev_progress_wp_idx", current_wp_idx_corridor))
            collision_wp4_wp5_dbg = float(
                (current_wp_idx_corridor in (4, 5))
                or (prev_wp_idx_corridor == 4 and current_wp_idx_corridor == 5)
            )
            collision_near_wp5_dbg = float(current_wp_idx_corridor == 5)
            collision_corridor_dbg = float(collision_wp4_wp5_dbg > 0.5)

            last_apply_diag = getattr(self, "_last_action_apply_diag", {}) or {}
            corridor_vx_send_dbg = float(last_apply_diag.get("vx_send", getattr(self, "last_applied_vx_cmd", 0.0)))
            corridor_vy_send_dbg = float(last_apply_diag.get("vy_send", getattr(self, "last_applied_vy_cmd", 0.0)))
            corridor_vz_send_dbg = float(last_apply_diag.get("vz_send", getattr(self, "last_applied_vz_cmd", 0.0)))
            corridor_yaw_rate_send_dbg = float(last_apply_diag.get("yaw_rate_send", getattr(self, "last_applied_yaw_rate_cmd", 0.0)))
            corridor_vxy_send_dbg = float(np.linalg.norm([corridor_vx_send_dbg, corridor_vy_send_dbg]))
            corridor_vx_cmd_raw_dbg = float(last_apply_diag.get("vx_cmd_raw", 0.0))
            corridor_vy_cmd_raw_dbg = float(last_apply_diag.get("vy_cmd_raw", 0.0))
            corridor_yaw_rate_cmd_raw_dbg = float(last_apply_diag.get("yaw_rate_cmd_raw", 0.0))

            corridor_cable_active_dbg = float(bool(cable_state.get("soft", False) or cable_state.get("hard", False)))
            corridor_cable_soft_raw_dbg = float(cable_state.get("cable_soft_raw_dbg", 0.0))
            corridor_cable_hard_raw_dbg = float(cable_state.get("cable_hard_raw_dbg", 0.0))
            corridor_cable_soft_final_dbg = float(cable_state.get("cable_soft_final_dbg", 0.0))
            corridor_cable_hard_final_dbg = float(cable_state.get("cable_hard_final_dbg", 0.0))
            corridor_cable_ratio_dbg = float(cable_state.get("ratio", 0.0))
            corridor_cable_center_dist_dbg = float(cable_state.get("center_dist", 0.0))
            corridor_cable_trigger_ratio_dbg = float(cable_state.get("cable_trigger_ratio_dbg", 0.0))
            corridor_cable_trigger_center_dbg = float(cable_state.get("cable_trigger_center_dbg", 0.0))
            corridor_cable_trigger_hold_dbg = float(cable_state.get("cable_trigger_hold_dbg", 0.0))
            corridor_cable_hold_left_dbg = float(cable_state.get("cable_hold_left_dbg", 0.0))
            corridor_cable_dx_dbg = float(cable_state.get("dx", 0.0))
            corridor_cable_dy_dbg = float(cable_state.get("dy", 0.0))

            yaw_now_corridor = float(getattr(self, "last_current_yaw", 0.0))
            right_dir_corridor = np.array([-np.sin(yaw_now_corridor), np.cos(yaw_now_corridor)], dtype=np.float32)
            lateral_now_corridor = float(np.dot(np.array([corridor_vx_send_dbg, corridor_vy_send_dbg], dtype=np.float32), right_dir_corridor))
            motion_sign_corridor = float(np.sign(corridor_cable_dx_dbg))
            corridor_motion_dot_cable_dbg = float(lateral_now_corridor * motion_sign_corridor)
            moving_toward_reliable = bool(abs(corridor_cable_dx_dbg) >= 0.04)
            corridor_moving_toward_cable_dbg = float(moving_toward_reliable and (corridor_motion_dot_cable_dbg > 0.0))

            cable_hard_raw_bool = bool(corridor_cable_hard_raw_dbg > 0.5)
            cable_ratio_high_bool = bool(corridor_cable_ratio_dbg >= float(self.cable_hard_ratio))
            cable_center_close_bool = bool(corridor_cable_center_dist_dbg <= float(self.cable_center_hard))
            if moving_toward_reliable:
                virtual_cable_near_collision_candidate_dbg = float(
                    (collision_wp4_wp5_dbg > 0.5)
                    and (cable_hard_raw_bool or cable_ratio_high_bool or cable_center_close_bool)
                    and (corridor_moving_toward_cable_dbg > 0.5)
                )
            else:
                virtual_cable_near_collision_candidate_dbg = float(
                    (collision_wp4_wp5_dbg > 0.5)
                    and cable_hard_raw_bool
                )

            if collision_wp4_wp5_dbg > 0.5:
                if cable_hard_raw_bool:
                    self._corridor_hard_raw_streak = int(getattr(self, "_corridor_hard_raw_streak", 0)) + 1
                else:
                    self._corridor_hard_raw_streak = 0

                if virtual_cable_near_collision_candidate_dbg > 0.5:
                    self._corridor_near_collision_streak = int(getattr(self, "_corridor_near_collision_streak", 0)) + 1
                else:
                    self._corridor_near_collision_streak = 0
            else:
                self._corridor_hard_raw_streak = 0
                self._corridor_near_collision_streak = 0

            info["collision_corridor_dbg"] = float(collision_corridor_dbg)
            info["collision_wp4_wp5_dbg"] = float(collision_wp4_wp5_dbg)
            info["collision_near_wp5_dbg"] = float(collision_near_wp5_dbg)
            info["collision_active_dbg"] = float(collision_active_dbg)
            info["collision_penalty_dbg"] = float(collision_penalty)
            info["collision_object_name_dbg"] = collision_object_name_dbg
            info["collision_impact_x_dbg"] = float(collision_impact_x_dbg)
            info["collision_impact_y_dbg"] = float(collision_impact_y_dbg)
            info["collision_impact_z_dbg"] = float(collision_impact_z_dbg)
            info["collision_position_x_dbg"] = float(collision_position_x_dbg)
            info["collision_position_y_dbg"] = float(collision_position_y_dbg)
            info["collision_position_z_dbg"] = float(collision_position_z_dbg)

            info["corridor_vx_send_dbg"] = float(corridor_vx_send_dbg)
            info["corridor_vy_send_dbg"] = float(corridor_vy_send_dbg)
            info["corridor_vz_send_dbg"] = float(corridor_vz_send_dbg)
            info["corridor_yaw_rate_send_dbg"] = float(corridor_yaw_rate_send_dbg)
            info["corridor_vxy_send_dbg"] = float(corridor_vxy_send_dbg)
            info["corridor_vx_cmd_raw_dbg"] = float(corridor_vx_cmd_raw_dbg)
            info["corridor_vy_cmd_raw_dbg"] = float(corridor_vy_cmd_raw_dbg)
            info["corridor_yaw_rate_cmd_raw_dbg"] = float(corridor_yaw_rate_cmd_raw_dbg)

            info["corridor_cable_active_dbg"] = float(corridor_cable_active_dbg)
            info["corridor_cable_soft_raw_dbg"] = float(corridor_cable_soft_raw_dbg)
            info["corridor_cable_hard_raw_dbg"] = float(corridor_cable_hard_raw_dbg)
            info["corridor_cable_soft_final_dbg"] = float(corridor_cable_soft_final_dbg)
            info["corridor_cable_hard_final_dbg"] = float(corridor_cable_hard_final_dbg)
            info["corridor_cable_ratio_dbg"] = float(corridor_cable_ratio_dbg)
            info["corridor_cable_center_dist_dbg"] = float(corridor_cable_center_dist_dbg)
            info["corridor_cable_trigger_ratio_dbg"] = float(corridor_cable_trigger_ratio_dbg)
            info["corridor_cable_trigger_center_dbg"] = float(corridor_cable_trigger_center_dbg)
            info["corridor_cable_trigger_hold_dbg"] = float(corridor_cable_trigger_hold_dbg)
            info["corridor_cable_hold_left_dbg"] = float(corridor_cable_hold_left_dbg)
            info["corridor_cable_dx_dbg"] = float(corridor_cable_dx_dbg)
            info["corridor_cable_dy_dbg"] = float(corridor_cable_dy_dbg)
            info["corridor_motion_dot_cable_dbg"] = float(corridor_motion_dot_cable_dbg)
            info["corridor_moving_toward_cable_dbg"] = float(corridor_moving_toward_cable_dbg)
            info["virtual_cable_near_collision_candidate_dbg"] = float(virtual_cable_near_collision_candidate_dbg)
            info["corridor_hard_raw_streak_dbg"] = float(getattr(self, "_corridor_hard_raw_streak", 0))
            info["corridor_near_collision_streak_dbg"] = float(getattr(self, "_corridor_near_collision_streak", 0))

        step_penalty = -0.001
        reward += step_penalty
        info["step_penalty"] = step_penalty

        if heavy_diag:
            info["seg_visible_dbg"] = float(self._last_seg_info["seg_visible"])
            info["seg_det_count_dbg"] = int(self._last_seg_info["seg_det_count"])
            info["seg_cable_ratio_dbg"] = float(self._last_seg_info["seg_cable_ratio"])
            info["seg_tower_ratio_dbg"] = float(self._last_seg_info["seg_tower_ratio"])
            info["seg_center_dx_dbg"] = float(self._last_seg_info["seg_center_dx"])
            info["seg_center_dy_dbg"] = float(self._last_seg_info["seg_center_dy"])
            info["seg_line_angle_sin_dbg"] = float(self._last_seg_info["seg_line_angle_sin"])
            info["seg_line_angle_cos_dbg"] = float(self._last_seg_info["seg_line_angle_cos"])
            info["seg_mask_stability_dbg"] = float(self._last_seg_info["seg_mask_stability"])
            info["seg_cable_center_dx_dbg"] = float(self._last_seg_info["seg_cable_center_dx"])
            info["seg_cable_center_dy_dbg"] = float(self._last_seg_info["seg_cable_center_dy"])
            info["seg_tower_center_dx_dbg"] = float(self._last_seg_info["seg_tower_center_dx"])
            info["seg_tower_center_dy_dbg"] = float(self._last_seg_info["seg_tower_center_dy"])
            info["seg_cable_angle_sin_dbg"] = float(self._last_seg_info["seg_cable_angle_sin"])
            info["seg_cable_angle_cos_dbg"] = float(self._last_seg_info["seg_cable_angle_cos"])

            info["spark_visible_dbg"] = float(self._last_spark_info["spark_visible"])
            info["spark_det_count_dbg"] = int(self._last_spark_info["spark_det_count"])
            info["spark_max_conf_dbg"] = float(self._last_spark_info["spark_max_conf"])
            info["spark_mean_conf_dbg"] = float(self._last_spark_info["spark_mean_conf"])
            info["spark_max_area_dbg"] = float(self._last_spark_info["spark_max_area"])
            info["spark_center_dx_dbg"] = float(self._last_spark_info["spark_center_dx"])
            info["spark_center_dy_dbg"] = float(self._last_spark_info["spark_center_dy"])
            info["spark_event_score_dbg"] = float(self._last_spark_info["spark_event_score"])
            info["spark_persistent_dbg"] = float(self._last_spark_info["spark_persistent"])

            info["yolo_det_count"] = float(self._last_yolo_info.get("yolo_det_count", 0))
            info["yolo_selected_class_id"] = float(self._last_yolo_info.get("yolo_selected_class_id", -1))
            info["yolo_selected_conf"] = float(self._last_yolo_info.get("yolo_selected_conf", 0.0))
            info["yolo_selected_area"] = float(self._last_yolo_info.get("yolo_selected_area", 0.0))
            info["yolo_selected_center_error"] = float(self._last_yolo_info.get("yolo_selected_center_error", 0.0))
            info["yolo_visible_dbg"] = float(self._last_yolo_info.get("yolo_visible", 0.0))
            info["yolo_class_norm_dbg"] = float(self._last_yolo_info.get("yolo_class_norm", 0.0))
            info["yolo_conf_dbg"] = float(self._last_yolo_info.get("yolo_conf", 0.0))
            info["yolo_cx_dbg"] = float(self._last_yolo_info.get("yolo_cx", 0.0))
            info["yolo_cy_dbg"] = float(self._last_yolo_info.get("yolo_cy", 0.0))
            info["yolo_area_dbg"] = float(self._last_yolo_info.get("yolo_area", 0.0))
            info["yolo_dx_dbg"] = float(self._last_yolo_info.get("yolo_dx", 0.0))
            info["yolo_dy_dbg"] = float(self._last_yolo_info.get("yolo_dy", 0.0))
        info["speed_norm"] = speed_norm
        info["action_norm"] = action_norm
        info["local_distance"] = local_distance
        info["final_distance"] = final_distance
        info["inspect_distance_raw"] = inspect_distance_raw
        info["standoff_error_raw"] = standoff_error_raw
        info["visible"] = visible
        info["conf"] = conf
        info["bbox_area"] = area
        info["center_error"] = center_error
        info["visual_memory_steps"] = int(self._target_memory_steps)
        info["yolo_reward_scale"] = float(self.yolo_reward_scale)
        if heavy_diag:
            info["locked_track_id_dbg"] = float(-1 if self.current_locked_track_id is None else self.current_locked_track_id)
            info["locked_track_waypoint_idx_dbg"] = float(self.current_locked_waypoint_idx)
            info["locked_track_missed_dbg"] = float(self.current_locked_track_missed)
            if self.enable_visual_inspection:
                info["raw_target_visible"] = float(self.raw_target_features[0])
                info["smoothed_target_visible"] = float(self.smoothed_target_features[0])
        current_meta = self._get_current_inspection_pair_meta()
        point_mode = str(current_meta.get("point_mode", "inspect")).lower()
        info["current_waypoint_idx_dbg"] = int(self.current_waypoint_idx)


        info["point_mode_is_inspect_dbg"] = float(point_mode == "inspect")
        info["point_mode_is_transit_dbg"] = float(point_mode == "transit")
        info["visual_gate_required_dbg"] = float(bool(current_meta.get("visual_gate_required", False)))
        info["wp_idx_dbg"] = float(wp_idx_dbg)

        spark_info = self._last_spark_info or {}
        spark_trace_diag = getattr(self, "_last_spark_trace_diag", {}) or {}
        info["spark_visible_dbg"] = float(spark_info.get("spark_visible", 0.0))
        info["spark_det_count_dbg"] = float(spark_info.get("spark_det_count", 0.0))
        info["spark_max_conf_dbg"] = float(spark_info.get("spark_max_conf", 0.0))
        info["spark_max_area_dbg"] = float(spark_info.get("spark_max_area", 0.0))
        info["spark_bbox_x1_dbg"] = float(spark_info.get("spark_bbox_x1", 0.0))
        info["spark_bbox_y1_dbg"] = float(spark_info.get("spark_bbox_y1", 0.0))
        info["spark_bbox_x2_dbg"] = float(spark_info.get("spark_bbox_x2", 0.0))
        info["spark_bbox_y2_dbg"] = float(spark_info.get("spark_bbox_y2", 0.0))
        info["spark_bbox_area_dbg"] = float(spark_info.get("spark_bbox_area", 0.0))
        info["spark_center_dx_dbg"] = float(spark_info.get("spark_center_dx", 0.0))
        info["spark_center_dy_dbg"] = float(spark_info.get("spark_center_dy", 0.0))
        info["spark_event_score_dbg"] = float(spark_info.get("spark_event_score", 0.0))
        info["spark_confirm_steps_dbg"] = float(getattr(self, "spark_confirm_steps", 0))
        info["spark_clear_steps_dbg"] = float(getattr(self, "spark_clear_steps", 0))
        info["spark_alarm_latched_dbg"] = float(bool(getattr(self, "spark_alarm_latched", False)))
        info["spark_response_active_dbg"] = float(bool(getattr(self, "spark_response_active", False)))
        info["spark_response_consumed_dbg"] = float(bool(getattr(self, "spark_response_consumed", False)))
        info["spark_override_steps_left_dbg"] = float(getattr(self, "spark_override_steps_left", 0))
        info["spark_hazard_mode_dbg"] = float(bool(self._in_spark_hazard_mode()))
        info["spark_last_seen_waypoint_dbg"] = float(getattr(self, "spark_last_seen_waypoint", -1))
        info["spark_last_response_waypoint_dbg"] = float(getattr(self, "spark_last_response_waypoint", -1))
        info["spark_iou_threshold_dbg"] = float(getattr(self, "spark_iou_threshold", 0.45))
        spark_camera_name_dbg = getattr(self, "spark_camera_name", None)
        if spark_camera_name_dbg is None:
            info["spark_camera_id_dbg"] = -1.0
        else:
            try:
                info["spark_camera_id_dbg"] = float(spark_camera_name_dbg)
            except Exception:
                info["spark_camera_id_dbg"] = -2.0
        spark_last_infer_step_dbg = int(getattr(self, "_last_spark_infer_step", -10 ** 9))
        if spark_last_infer_step_dbg <= -10 ** 8:
            spark_infer_interval_steps_dbg = -1.0
            spark_infer_interval_sec_dbg = -1.0
        else:
            spark_infer_interval_steps_dbg = float(max(0, int(self.current_step) - spark_last_infer_step_dbg))
            spark_infer_interval_sec_dbg = float(spark_infer_interval_steps_dbg * float(self.step_duration))
        info["spark_infer_count_dbg"] = float(getattr(self, "_spark_infer_count", 0))
        info["spark_last_infer_step_dbg"] = float(spark_last_infer_step_dbg)
        info["spark_infer_interval_steps_dbg"] = float(spark_infer_interval_steps_dbg)
        info["spark_infer_interval_sec_dbg"] = float(spark_infer_interval_sec_dbg)
        info["spark_infer_time_ms_dbg"] = float(getattr(self, "_last_spark_infer_ms", 0.0))
        info["spark_override_active_dbg"] = float(spark_trace_diag.get("spark_override_active_dbg", 0.0))
        info["spark_action_override_changed_dbg"] = float(spark_trace_diag.get("spark_action_override_changed_dbg", 0.0))
        info["spark_action_to_apply_0_dbg"] = float(spark_trace_diag.get("spark_action_to_apply_0_dbg", 0.0))
        info["spark_action_to_apply_1_dbg"] = float(spark_trace_diag.get("spark_action_to_apply_1_dbg", 0.0))
        info["spark_action_to_apply_2_dbg"] = float(spark_trace_diag.get("spark_action_to_apply_2_dbg", 0.0))
        info["spark_action_to_apply_3_dbg"] = float(spark_trace_diag.get("spark_action_to_apply_3_dbg", 0.0))
        info["spark_action_after_override_0_dbg"] = float(spark_trace_diag.get("spark_action_after_override_0_dbg", 0.0))
        info["spark_action_after_override_1_dbg"] = float(spark_trace_diag.get("spark_action_after_override_1_dbg", 0.0))
        info["spark_action_after_override_2_dbg"] = float(spark_trace_diag.get("spark_action_after_override_2_dbg", 0.0))
        info["spark_action_after_override_3_dbg"] = float(spark_trace_diag.get("spark_action_after_override_3_dbg", 0.0))

        if heavy_diag:
            info["visual_ready_dbg"] = float(visual_ready_now)
            info["visual_fail_visible"] = float(visual_fail_visible)
            info["visual_fail_center"] = float(visual_fail_center)
            info["visual_fail_center_dx"] = float(visual_fail_center_dx)
            info["visual_fail_center_dy"] = float(visual_fail_center_dy)
            info["visual_fail_yaw"] = float(visual_fail_yaw)
            info["visual_hold_wait"] = float(visual_hold_wait)
            info["visual_gate_hold_counter"] = float(visual_gate_hold_counter)
            info["visual_gate_hold_required"] = float(visual_gate_hold_required)
            info["target_visible_now"] = float(visible)
            info["target_conf_now"] = float(conf)
            info["target_area_now"] = float(area)
            info["target_center_dx_now"] = float(dx_center)
            info["target_center_dy_now"] = float(dy_center)
            info["center_error_now"] = float(center_error)
            info["yaw_error_geom_now"] = float(yaw_error_geom)
            info["visible_threshold_now"] = float(visible_thr)
            info["center_threshold_now"] = float(center_thr)
            info["yaw_threshold_now"] = float(yaw_thr)
            info["point_mode_dbg"] = float(point_mode_dbg)

            info["spark_alarm_latched_dbg"] = float(self.spark_alarm_latched)
            info["spark_response_consumed_dbg"] = float(self.spark_response_consumed)
            info["spark_response_active_dbg"] = float(self.spark_response_active)
            info["spark_override_steps_left_dbg"] = float(self.spark_override_steps_left)
            info["spark_confirm_steps_dbg"] = float(self.spark_confirm_steps)
            info["spark_clear_steps_dbg"] = float(self.spark_clear_steps)
            info["spark_hazard_mode_dbg"] = float(self._in_spark_hazard_mode())
            info["spark_last_seen_waypoint_dbg"] = float(self.spark_last_seen_waypoint)
            info["spark_last_response_waypoint_dbg"] = float(self.spark_last_response_waypoint)



        self.prev_local_distance = local_distance_raw
        self.prev_final_distance = final_distance
        self.prev_min_obstacle_dist = min_obstacle_dist
        self._prev_progress_wp_idx = current_progress_wp_idx
        self._prev_progress_local_path_index = current_progress_local_path_index
        self._prev_progress_local_target = (
            current_progress_local_target.copy()
            if current_progress_local_target is not None
            else None
        )
        return float(reward), info

    def _build_mission_route(self):
        if self.inspection_pairs is not None and len(self.inspection_pairs) > 0:
            self.mission_route = [
                pair["route_point"].copy() for pair in self.inspection_pairs
            ]
        elif self.current_route_points is not None and len(self.current_route_points) > 0:
            self.mission_route = [np.asarray(p, dtype=np.float32).copy() for p in self.current_route_points]
        elif self.route_points is not None and len(self.route_points) > 0:
            self.mission_route = [np.asarray(p, dtype=np.float32).copy() for p in self.route_points]
        else:
            self.mission_route = [self.goal_position.copy()]

        self.current_waypoint_idx = 0
        self.local_path_world = []
        self.local_path_index = 0
        self.active_local_waypoint = self.mission_route[0].copy()
        self.route_total_length = 0.0

        anchor = self.start_position.copy() if self.start_position is not None else np.array([0.0, 0.0, -3.0],
                                                                                             dtype=np.float32)
        for p in self.mission_route:
            self.route_total_length += float(np.linalg.norm(p - anchor))
            anchor = p.copy()
        self.route_total_length = max(self.route_total_length, 1e-6)

    def _accumulate_step_stats(self, reward_info, obs):
        super()._accumulate_step_stats(reward_info, obs)
        for key in [
            "target_visible_reward",
            "target_center_reward",
            "target_scale_reward",
            "standoff_reward",
            "inspection_hold_reward",
            "visual_align_reward",
        ]:
            val = float(reward_info.get(key, 0.0))
            self.episode_stats.setdefault(f"{key}_sum", 0.0)
            self.episode_stats.setdefault(f"{key}_abs_sum", 0.0)
            self.episode_stats[f"{key}_sum"] += val
            self.episode_stats[f"{key}_abs_sum"] += abs(val)

    def _build_episode_summary(self, final_distance_norm, final_distance_raw, term_reason, truncated, mean_heading_error_cos, mean_speed, obs):
        summary = super()._build_episode_summary(
            final_distance_norm=final_distance_norm,
            final_distance_raw=final_distance_raw,
            term_reason=term_reason,
            truncated=truncated,
            mean_heading_error_cos=mean_heading_error_cos,
            mean_speed=mean_speed,
            obs=obs,
        )



        target = self._get_inspection_target()

        episode_len = max(int(self.current_step), 1)
        steps_with_det = max(int(self.yolo_steps_with_det), 1)
        visible_ratio = float(self.visible_steps) / float(episode_len)
        mean_det_count = float(self.yolo_det_count_sum) / float(episode_len)

        # средние только по шагам, где был selected target
        selected_steps = max(int(self.yolo_steps_with_det), 1)
        summary["seg_visible_ratio"] = float(self.seg_steps_with_det) / max(float(self.current_step), 1.0)
        summary["seg_mean_det_count"] = float(self.seg_det_count_sum) / max(float(self.current_step), 1.0)
        summary["seg_mean_cable_ratio"] = float(self.seg_cable_ratio_sum) / max(float(self.current_step), 1.0)
        summary["seg_mean_tower_ratio"] = float(self.seg_tower_ratio_sum) / max(float(self.current_step), 1.0)
        summary["seg_mean_center_error"] = float(self.seg_center_err_sum) / max(float(self.seg_steps_with_det), 1.0)
        summary["seg_mean_mask_stability"] = float(self.seg_mask_stability_sum) / max(float(self.seg_steps_with_det),
                                                                                      1.0)
        summary["seg_cable_visible_ratio"] = float(self.seg_cable_steps_with_det) / max(float(self.current_step), 1.0)
        summary["seg_tower_visible_ratio"] = float(self.seg_tower_steps_with_det) / max(float(self.current_step), 1.0)

        summary["seg_mean_cable_center_dx"] = float(self.seg_cable_center_dx_sum) / max(
            float(self.seg_cable_steps_with_det), 1.0)
        summary["seg_mean_cable_center_dy"] = float(self.seg_cable_center_dy_sum) / max(
            float(self.seg_cable_steps_with_det), 1.0)

        summary["seg_mean_tower_center_dx"] = float(self.seg_tower_center_dx_sum) / max(
            float(self.seg_tower_steps_with_det), 1.0)
        summary["seg_mean_tower_center_dy"] = float(self.seg_tower_center_dy_sum) / max(
            float(self.seg_tower_steps_with_det), 1.0)

        summary["seg_mean_cable_angle_sin"] = float(self.seg_cable_angle_sin_sum) / max(
            float(self.seg_cable_steps_with_det), 1.0)
        summary["seg_mean_cable_angle_cos"] = float(self.seg_cable_angle_cos_sum) / max(
            float(self.seg_cable_steps_with_det), 1.0)
        summary["spark_visible_ratio"] = float(self.spark_steps_with_det) / max(float(self.current_step), 1.0)
        summary["spark_event_count"] = int(self.spark_event_count)
        summary["spark_mean_max_conf"] = float(self.spark_max_conf_sum) / max(float(self.spark_steps_with_det), 1.0)
        summary["spark_mean_conf"] = float(self.spark_mean_conf_sum) / max(float(self.spark_steps_with_det), 1.0)
        summary["spark_mean_max_area"] = float(self.spark_max_area_sum) / max(float(self.spark_steps_with_det), 1.0)
        summary["spark_mean_event_score"] = float(self.spark_event_score_sum) / max(float(self.spark_steps_with_det),
                                                                                    1.0)
        summary["spark_persistent_count"] = int(self.spark_persistent_count)

        summary.update({
            "visible_steps": int(self.visible_steps),
            "lost_target_steps": int(self.lost_target_steps),
            "inspection_target_x": float(target[0]),
            "inspection_target_y": float(target[1]),
            "inspection_target_z": float(target[2]),
            "visual_target_memory_steps": int(self._target_memory_steps),
            "visual_target_visible": float(self.yolo_features[0]) if self.yolo_feature_dim > 0 else 0.0,
            "visual_target_conf": float(self.yolo_features[2]) if self.yolo_feature_dim > 2 else 0.0,

            "visible_ratio": float(visible_ratio),
            "mean_det_count": float(mean_det_count),
            "steps_with_detection": int(self.yolo_steps_with_det),
            "yolo_selected_switch_count": int(self.yolo_selected_switch_count),

            "mean_selected_conf": float(self.yolo_conf_sum / selected_steps),
            "mean_selected_area": float(self.yolo_area_sum / selected_steps),
            "mean_selected_center_error": float(self.yolo_center_err_sum / selected_steps),
        })
        return summary

    def _get_inspection_target(self) -> np.ndarray:

        if self.inspection_pairs is None or len(self.inspection_pairs) == 0:
            raise RuntimeError("inspection_pairs REQUIRED for inspect env")

        idx = int(np.clip(self.current_waypoint_idx, 0, len(self.inspection_pairs) - 1))

        target = self.inspection_pairs[idx]["target_point"].astype(np.float32)

        self.active_inspection_target = target
        return target.copy()

    def _geom_inspect_ready_for_current_waypoint(self):
        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()

        pos = self._current_position_vec()
        current_wp = self.mission_route[self.current_waypoint_idx]
        route_dist = float(np.linalg.norm(current_wp - pos))

        # значения по умолчанию, чтобы не ловить UnboundLocalError
        yaw_error = 0.0
        yaw_ready = True
        target_xy_dist = 0.0

        # Для transit проверяем только достижение route_point
        if point_mode != "inspect":
            return {
                "ready": bool(route_dist <= self.waypoint_tolerance),
                "route_dist": float(route_dist),
                "yaw_error": 0.0,
                "target_xy_dist": 0.0,
                "yaw_ready": True,
            }

        _, _, target_delta, _, _ = self._get_inspection_metrics()

        target_xy_dist = float(np.linalg.norm(target_delta[:2]))
        yaw_thr = float(meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))

        # Если XY-вектор до target почти нулевой, yaw считаем уже готовым
        if target_xy_dist <= 0.15:
            yaw_error = 0.0
            yaw_ready = True
        else:
            desired_yaw = float(np.arctan2(target_delta[1], target_delta[0]))
            yaw_error = abs(self._wrap_angle(desired_yaw - float(self.last_current_yaw)))
            yaw_ready = bool(yaw_error <= yaw_thr)

        ready = bool(route_dist <= self.waypoint_tolerance and yaw_ready)

        return {
            "ready": bool(ready),
            "route_dist": float(route_dist),
            "yaw_error": float(yaw_error),
            "target_xy_dist": float(target_xy_dist),
            "yaw_ready": bool(yaw_ready),
        }
    def _get_current_inspection_pair_meta(self) -> dict:
        if self.inspection_pairs is None or len(self.inspection_pairs) == 0:
            raise RuntimeError("inspection_pairs REQUIRED for inspect env")

        idx = int(np.clip(self.current_waypoint_idx, 0, len(self.inspection_pairs) - 1))
        return self.inspection_pairs[idx]
    def _is_lost_target_guard_active(self) -> bool:
        """
        lost_target должен считаться только для точек,
        где реально требуется визуальное подтверждение,
        и только в зоне инспекции, а не на всём подлёте.
        """
        if not bool(getattr(self, "enable_visual_inspection", False)):
            return False

        if not bool(getattr(self, "require_target_visibility", False)):
            return False

        if not bool(getattr(self, "_visual_backend_ready", False)):
            return False

        try:
            meta = self._get_current_inspection_pair_meta()
            point_mode = str(meta.get("point_mode", "inspect")).lower()
            visual_required = bool(meta.get("visual_gate_required", False))
        except Exception:
            return False

        if point_mode != "inspect":
            return False

        if not visual_required:
            return False

        try:
            local_dist = float(self._get_local_distance_raw())
        except Exception:
            local_dist = 999.0

        gate_phase = (
            local_dist <= max(float(self.waypoint_tolerance) * 1.25, 0.75)
            or float(getattr(self, "visual_gate_last_blocked", 0.0)) > 0.5
        )

        return bool(gate_phase)
    def _get_current_inspection_mode(self, final: bool = False) -> str:
        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()

        if point_mode != "inspect":
            return "transit"

        if self.enable_visual_inspection and bool(meta.get("visual_gate_required", False)):
            return "visual"

        return "geom"

    def _use_ema_like_pretrain_mode(self) -> bool:
        """
        EMA-подобный режим нужен только для геометрического pretrain:
        - флаг явно включен;
        - visual inspection сейчас не используется;
        - текущая inspect-точка не требует visual gate.
        """
        if not getattr(self, "ema_like_pretrain_mode", False):
            return False

        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()

        if point_mode != "inspect":
            return False

        if self.enable_visual_inspection:
            return False

        if bool(meta.get("visual_gate_required", False)):
            return False

        return True
    def _get_current_alignment_state(self, final: bool = False) -> Dict[str, float]:
        meta = self._get_current_inspection_pair_meta()
        mode = self._get_current_inspection_mode(final=final)

        vis = self._get_control_visual_features()
        visible = float(vis[0]) if vis.shape[0] > 0 else 0.0
        conf = float(vis[2]) if vis.shape[0] > 2 else 0.0
        area = float(vis[5]) if vis.shape[0] > 5 else 0.0
        dx_center = float(vis[6]) if vis.shape[0] > 6 else 0.0
        dy_center = float(vis[7]) if vis.shape[0] > 7 else 0.0
        center_error = float(np.clip(np.sqrt(dx_center ** 2 + dy_center ** 2), 0.0, 1.5))

        _, _, target_delta, _, _ = self._get_inspection_metrics()
        target_xy_dist = float(np.linalg.norm(target_delta[:2]))
        if target_xy_dist <= 0.15:
            yaw_error_geom = 0.0
        else:
            desired_yaw_geom = float(np.arctan2(target_delta[1], target_delta[0]))
            yaw_error_geom = abs(self._wrap_angle(desired_yaw_geom - float(self.last_current_yaw)))

        visible_thr = float(meta.get("visual_gate_visible_threshold", self.visual_gate_visible_threshold))
        center_thr = float(meta.get("visual_gate_center_threshold", self.visual_gate_center_threshold))
        yaw_thr = float(meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))

        visual_ready = (
                visible > visible_thr
                and center_error < center_thr
                and yaw_error_geom < yaw_thr
        )

        geom_ready = False
        if mode == "geom":
            geom_ready = bool(self._geom_inspect_ready_for_current_waypoint()["ready"])

        if mode == "visual":
            ready = visual_ready
            align_score = float(
                max(0.0, 1.0 - center_error / max(center_thr, 1e-6))
                * max(0.0, 1.0 - yaw_error_geom / max(yaw_thr, 1e-6))
                * np.clip(visible, 0.0, 1.0)
            )
        elif mode == "geom":
            ready = geom_ready
            align_score = float(max(0.0, 1.0 - yaw_error_geom / max(yaw_thr, 1e-6)))
        else:
            ready = True
            align_score = 0.0

        return {
            "mode": mode,
            "ready": float(ready),
            "visible": visible,
            "conf": conf,
            "area": area,
            "center_error": center_error,
            "yaw_error_geom": yaw_error_geom,
            "align_score": align_score,
            "visible_thr": visible_thr,
            "center_thr": center_thr,
            "yaw_thr": yaw_thr,
        }

    def _update_waypoint_progress(self) -> bool:
        if not self.mission_route:
            return False

        pos = self._current_position_vec()
        current_wp = self.mission_route[self.current_waypoint_idx]
        dist = float(np.linalg.norm(current_wp - pos))
        wp_idx = int(getattr(self, "current_waypoint_idx", 0))

        if self._wp_pass_last_idx != wp_idx:
            self._wp_pass_last_idx = wp_idx
            self._wp_pass_best_dist = float(dist)
            self._wp_pass_increase_steps = 0

        if dist < self._wp_pass_best_dist:
            self._wp_pass_best_dist = float(dist)
            self._wp_pass_increase_steps = 0
        elif (
                self._wp_pass_best_dist <= float(self.waypoint_pass_tolerance)
                and dist > self._wp_pass_best_dist + float(self.waypoint_pass_margin)
        ):
            self._wp_pass_increase_steps += 1

        try:
            cable_state = self._get_cable_safety_state(mutate=False)
            cable_danger = (
                    float(cable_state.get("visible", 0.0)) > 0.5
                    and (
                            bool(cable_state.get("soft", False))
                            or bool(cable_state.get("hard", False))
                    )
            )
        except Exception:
            cable_danger = False

        route_reached_geom = dist <= float(self.waypoint_tolerance)

        route_passed_near = (
                self._wp_pass_best_dist <= float(self.waypoint_pass_tolerance)
                and self._wp_pass_increase_steps >= int(self.waypoint_pass_increase_steps_req)
                and not bool(cable_danger)
        )


        if int(getattr(self, "current_step", 0)) % 50 == 0:
            delta = current_wp - pos
            xy_dist = float(np.linalg.norm(delta[:2]))
            z_dist = float(abs(delta[2]))

        # ===== DEBUG WAYPOINT GEOMETRY =====


        is_final_wp = (self.current_waypoint_idx >= len(self.mission_route) - 1)
        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()
        mode = self._get_current_inspection_mode(final=False)
        use_visual_gate = (mode == "visual") and not is_final_wp


        spark_hazard_mode = self._in_spark_hazard_mode()

        if is_final_wp:
            self.active_local_waypoint = current_wp.copy()
            self.local_path_world = [current_wp.copy()]
            self.local_path_index = 0
            self.visual_gate_last_blocked = 0.0
            self.point_stall_steps = 0
            return False

        if route_reached_geom or route_passed_near:


            # Если участок помечен как hazardous, не висим на visual/geom gate,
            # а продолжаем маршрут осторожно.
            if spark_hazard_mode:
                self.visual_gate_last_blocked = 0.0
                self.visual_gate_counter = 0
                self.current_waypoint_idx += 1
                self._wp_pass_best_dist = 1e9
                self._wp_pass_increase_steps = 0
                self._wp_pass_last_idx = -1

                self.current_locked_track_id = None
                self.current_locked_waypoint_idx = -1
                self.current_locked_track_missed = 0

                self.shift_mode_steps_left = 0
                self.shift_vector[:] = 0.0
                self.point_stall_steps = 0
                self._rebuild_local_path(force=True)

                self.print_current_route_target()

                return True
            # 1) Если включён visual gate — сначала он
            if use_visual_gate:
                gate_ready, visible, center_error, yaw_error_geom = self._visual_gate_ready_for_current_waypoint()


                if not gate_ready:
                    self.visual_gate_last_blocked = 1.0

                    self.no_progress_steps = 0
                    self.inactive_steps = 0
                    self.failed_progress_steps = 0
                    self.point_stall_steps = 0

                    self.active_local_waypoint = current_wp.copy()
                    self.local_path_world = [current_wp.copy()]
                    self.local_path_index = 0

                    return False

            require_geom_gate = bool(meta.get("geom_gate_required", point_mode == "inspect"))

            if point_mode == "inspect" and (not use_visual_gate) and require_geom_gate:
                geom_ready = self._geom_inspect_ready_for_current_waypoint()



                if not bool(geom_ready.get("ready", False)):
                    self.visual_gate_last_blocked = 1.0

                    # Не даём no_progress/inactive убить эпизод, пока дрон разворачивается.
                    self.no_progress_steps = 0
                    self.inactive_steps = 0
                    self.failed_progress_steps = 0
                    self.point_stall_steps = 0

                    # Фиксируем локальную цель на текущей route_point,
                    # чтобы дрон не переключился на следующую точку до yaw-ready.
                    self.active_local_waypoint = current_wp.copy()
                    self.local_path_world = [current_wp.copy()]
                    self.local_path_index = 0

                    return False

            elif point_mode == "inspect" and (not use_visual_gate):
                pass

            self.visual_gate_last_blocked = 0.0
            self.visual_gate_counter = 0
            self.current_waypoint_idx += 1

            self.current_locked_track_id = None
            self.current_locked_waypoint_idx = -1
            self.current_locked_track_missed = 0

            self.shift_mode_steps_left = 0
            self.shift_vector[:] = 0.0
            self.point_stall_steps = 0
            self._rebuild_local_path(force=True)

            self.print_current_route_target()

            return True

        self.visual_gate_last_blocked = 0.0
        return False
    # =========================================================
    # Termination
    # =========================================================

    def _check_terminated(self, obs: np.ndarray):
        collision_flag = self._get_collision_flag_raw()
        final_distance_raw = self._get_goal_distance_raw()
        altitude_norm = self._get_altitude_norm_raw()

        final_goal_reached, _ = self._is_final_goal_ready()

        if collision_flag > 0.5:
            return True, "collision"
        if final_goal_reached:
            return True, "goal_reached"
        if self._is_out_of_bounds():
            try:
                pos_dbg = self._current_position_vec()
                print(
                    "[OOB]",
                    "step=", int(self.current_step),
                    "wp_idx=", int(self.current_waypoint_idx),
                    "pos=", pos_dbg,
                    "abs_x=", abs(float(pos_dbg[0])),
                    "abs_y=", abs(float(pos_dbg[1])),
                    "world_xy_limit=", float(self.world_xy_limit),
                    "active_local_waypoint=", self.active_local_waypoint,
                    "current_wp=", self.mission_route[self.current_waypoint_idx] if self.mission_route else None,
                )
            except Exception as exc:
                print("[OOB] debug failed:", exc)
            return True, "out_of_bounds"
        if altitude_norm < 0.08:
            return True, "too_low"
        if (
                self._is_lost_target_guard_active()
                and self.lost_target_steps >= self.lost_target_max_steps
        ):
            return True, "lost_target"
        if self.no_progress_steps >= max(self.no_progress_max_steps * 3, 80):
            return True, "no_progress"

        return False, None

    def _draw_debug(self, frame, detections, deep_sort_tracks=None, selected_det=None, smoothed_features=None):
        if frame is None:
            return None

        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]

        cx_frame, cy_frame = w // 2, h // 2
        cv2.circle(debug_frame, (cx_frame, cy_frame), 5, (255, 255, 255), -1)

        selected_track_id = None
        if selected_det is not None and "track_id" in selected_det:
            try:
                selected_track_id = int(selected_det["track_id"])
            except Exception:
                selected_track_id = None

        # ===== 1) raw YOLO detections =====
        # ===== 1) raw YOLO detections =====
        for det in detections or []:
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
            conf = float(det.get("conf", 0.0))
            cls_name = str(det.get("class_name", "obj"))

            color = (0, 180, 255)
            thickness = 1

            is_selected_det = False
            if selected_det is not None:
                # В async-режиме selected_det и detections — это копии словарей,
                # поэтому сравнение по `is` не работает.
                if "track_id" in selected_det and "track_id" in det:
                    try:
                        is_selected_det = int(selected_det["track_id"]) == int(det["track_id"])
                    except Exception:
                        is_selected_det = False

                if not is_selected_det:
                    same_cls = str(selected_det.get("class_name", "")) == cls_name
                    cx_diff = abs(float(selected_det.get("center_x", 0.0)) - float(det.get("center_x", 0.0)))
                    cy_diff = abs(float(selected_det.get("center_y", 0.0)) - float(det.get("center_y", 0.0)))
                    area_diff = abs(float(selected_det.get("area", 0.0)) - float(det.get("area", 0.0)))

                    is_selected_det = same_cls and cx_diff < 0.01 and cy_diff < 0.01 and area_diff < 0.01

            if is_selected_det:
                color = (0, 255, 0)
                thickness = 3

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                debug_frame,
                f"YOLO {cls_name} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # ===== 2) Deep SORT tracks =====
        for trk in deep_sort_tracks or []:
            x1 = int(trk["x1"])
            y1 = int(trk["y1"])
            x2 = int(trk["x2"])
            y2 = int(trk["y2"])
            trk_id = int(trk.get("track_id", -1))

            color = (255, 200, 0)   # все DS tracks
            thickness = 2

            if self.current_locked_track_id is not None and trk_id == int(self.current_locked_track_id):
                color = (255, 0, 255)   # locked track
                thickness = 3

            if selected_track_id is not None and trk_id == selected_track_id:
                color = (0, 255, 0)     # выбранный трек
                thickness = 3

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                debug_frame,
                f"DS id={trk_id}",
                (x1, min(h - 5, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            tcx = int(np.clip(trk.get("center_x", 0.0), 0.0, 1.0) * w)
            tcy = int(np.clip(trk.get("center_y", 0.0), 0.0, 1.0) * h)
            cv2.circle(debug_frame, (tcx, tcy), 3, color, -1)

        # ===== 3) EMA-центр признаков, реально уходящих агенту =====
        if smoothed_features is not None and len(smoothed_features) >= 5:
            visible = float(smoothed_features[0])
            center_x = float(smoothed_features[3])
            center_y = float(smoothed_features[4])

            if visible > 0.01:
                ex = int(np.clip(center_x, 0.0, 1.0) * w)
                ey = int(np.clip(center_y, 0.0, 1.0) * h)
                cv2.drawMarker(
                    debug_frame,
                    (ex, ey),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=14,
                    thickness=2,
                )

        # ===== 4) Верхняя панель статуса =====
        visible = conf = area = dx = dy = 0.0
        if smoothed_features is not None and len(smoothed_features) >= 8:
            visible = float(smoothed_features[0])
            conf = float(smoothed_features[2])
            area = float(smoothed_features[5])
            dx = float(smoothed_features[6])
            dy = float(smoothed_features[7])

        line1 = (
            f"wp={int(self.current_waypoint_idx)} "
            f"locked_id={-1 if self.current_locked_track_id is None else int(self.current_locked_track_id)} "
            f"sel_id={-1 if selected_track_id is None else int(selected_track_id)} "
            f"missed={int(self.current_locked_track_missed)}"
        )
        line2 = (
            f"tracks={len(deep_sort_tracks or [])} dets={len(detections or [])} "
            f"visible={visible:.0f} conf={conf:.2f} area={area:.3f} dx={dx:.3f} dy={dy:.3f}"
        )

        cv2.putText(
            debug_frame,
            line1,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_frame,
            line2,
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return debug_frame

    def _update_inspection_lock(self, align_score: float, standoff_ok: bool, gate_blocked: bool):
        """
        Hysteresis:
        lock входим только если alignment устойчивый и дистанция нормальная.
        Выходим — если alignment упал ниже exit-threshold.
        """

        score = float(align_score)

        if gate_blocked:
            score *= 0.85

        if standoff_ok:
            score += 0.08

        if not self.inspect_lock:
            if score >= self.inspect_enter_threshold:
                self.inspect_lock = True
                self.inspect_lock_steps = 0
        else:
            self.inspect_lock_steps += 1
            if score < self.inspect_exit_threshold and self.inspect_lock_steps > self.inspect_lock_min_steps:
                self.inspect_lock = False

    def _update_deep_sort_tracks(self, detections, image_bgr):
        log_ds = bool(self.debug_print_yolo)
        if log_ds:
            print(
                "[DEEP_SORT] call",
                "tracker_none=", self.deep_sort_tracker is None,
                "image_none=", image_bgr is None,
                "det_count=", len(detections) if detections is not None else -1,
            )
        if self.deep_sort_tracker is None or image_bgr is None:
            return []

        h, w = image_bgr.shape[:2]
        ds_inputs = []
        for det in detections:
            x1 = float(det["x1"])
            y1 = float(det["y1"])
            x2 = float(det["x2"])
            y2 = float(det["y2"])
            ds_inputs.append((
                [x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)],
                float(det["conf"]),
                str(det["class_name"]),
            ))
        if log_ds:
            print("[DEEP_SORT] ds_inputs_count=", len(ds_inputs))
            if len(ds_inputs) > 0:
                print("[DEEP_SORT] first_input=", ds_inputs[0])

        tracks = self.deep_sort_tracker.update_tracks(ds_inputs, frame=image_bgr)

        if log_ds:
            print("[DEEP_SORT] raw_tracks_count=", len(tracks))
        confirmed_count = 0
        tentative_count = 0
        for trk in tracks:
            try:
                if trk.is_confirmed():
                    confirmed_count += 1
                else:
                    tentative_count += 1
            except Exception:
                pass
        if log_ds:
            print(
                "[DEEP_SORT] track_stats",
                "confirmed=", confirmed_count,
                "tentative=", tentative_count,
        )
        out = []
        for trk in tracks:
            # пропускаем только совсем "мёртвые" треки, tentative тоже показываем
            if getattr(trk, "time_since_update", 0) > 1:
                continue

            ltrb = trk.to_ltrb()
            x1, y1, x2, y2 = [float(v) for v in ltrb]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            cx = 0.5 * (x1 + x2) / max(w, 1.0)
            cy = 0.5 * (y1 + y2) / max(h, 1.0)
            area = (bw * bh) / max(float(w * h), 1.0)

            out.append({
                "track_id": int(trk.track_id),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": float(np.clip(cx, 0.0, 1.0)),
                "center_y": float(np.clip(cy, 0.0, 1.0)),
                "area": float(np.clip(area, 0.0, 1.0)),
                "dx_from_center": float(np.clip(cx - 0.5, -1.0, 1.0)),
                "dy_from_center": float(np.clip(cy - 0.5, -1.0, 1.0)),
            })
        if log_ds:
            print("[DEEP_SORT] out_count=", len(out))
            if len(out) > 0:
                print("[DEEP_SORT] first_out=", out[0])
        return out

    def _get_route_visual_prior(self):
        pos = self._current_position_vec()
        target = self._get_inspection_target()

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

        delta = target - pos
        dist = float(np.linalg.norm(delta))

        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_error = self._wrap_angle(desired_yaw - float(yaw))

        hfov_rad = np.deg2rad(self.camera_hfov_deg)
        expected_cx = float(np.clip(0.5 + yaw_error / max(hfov_rad, 1e-6), 0.0, 1.0))

        # грубая оценка площади: ближе = больше bbox
        expected_area = float(np.clip(
            self.desired_bbox_area * (self.desired_standoff_distance ** 2) / max(dist ** 2, 1e-6),
            0.0,
            1.0,
        ))

        return {
            "expected_cx": expected_cx,
            "expected_area": expected_area,
            "distance_raw": dist,
        }


    def reset_spark_response_state(self):
        self.spark_alarm_latched = False
        self.spark_response_consumed = False
        self.spark_response_active = False
        self.spark_override_steps_left = 0
        self.spark_confirm_steps = 0
        self.spark_clear_steps = 0
        self.spark_last_seen_waypoint = -1
        self.spark_last_response_waypoint = -1

    def _spark_response_triggered(self) -> bool:
        if not self.spark_response_enabled:
            return False

        spark = self._last_spark_info or {}

        spark_visible = float(spark.get("spark_visible", 0.0))
        spark_conf = float(spark.get("spark_max_conf", 0.0))
        spark_event = float(spark.get("spark_event_score", 0.0))
        spark_persistent = float(spark.get("spark_persistent", 0.0))

        return bool(
            (spark_visible > 0.5 and spark_conf >= self.spark_confirm_conf_thr)
            or (spark_event >= self.spark_confirm_event_thr)
            or (spark_persistent > 0.5 and spark_conf >= max(0.30, self.spark_confirm_conf_thr * 0.7))
        )

    def _in_spark_hazard_mode(self) -> bool:
        return bool(self.spark_response_enabled and self.spark_alarm_latched)

    def _update_spark_response_state(self):
        if not self.spark_response_enabled:
            return

        triggered = self._spark_response_triggered()

        if triggered:
            self.spark_clear_steps = 0
            self.spark_confirm_steps += 1
            self.spark_last_seen_waypoint = int(self.current_waypoint_idx)

            # вход в spark-эпизод
            if not self.spark_alarm_latched and self.spark_confirm_steps >= self.spark_confirm_steps_req:
                self.spark_alarm_latched = True
                self.spark_response_consumed = False

            # единоразовая реакция на spark-эпизод
            if self.spark_alarm_latched and not self.spark_response_consumed:
                self.spark_response_active = True
                self.spark_override_steps_left = self.spark_override_duration_steps
                self.spark_response_consumed = True
                self.spark_last_response_waypoint = int(self.current_waypoint_idx)
        else:
            self.spark_confirm_steps = 0

            if self.spark_alarm_latched:
                self.spark_clear_steps += 1

                # выходим из hazardous mode только если spark реально пропала
                if self.spark_clear_steps >= self.spark_clear_steps_req:
                    self.spark_alarm_latched = False
                    self.spark_response_consumed = False
                    self.spark_response_active = False
                    self.spark_override_steps_left = 0
                    self.spark_clear_steps = 0

    def _get_spark_override_action(self) -> Optional[np.ndarray]:
        if not self.spark_response_enabled or not self.spark_response_active:
            return None

        if self.spark_override_steps_left <= 0:
            self.spark_response_active = False
            return None

        spark = self._last_spark_info or {}
        spark_dx = float(np.clip(spark.get("spark_center_dx", 0.0), -1.0, 1.0))

        # [forward/back, left/right, z_intent, yaw]
        ax = -self.spark_response_backoff_ratio
        ay = 0.0
        az = 0.0
        ayaw = 0.0

        # вспышка справа -> уход влево; слева -> уход вправо
        if spark_dx > 0.08:
            ay = -self.spark_response_lateral_ratio
        elif spark_dx < -0.08:
            ay = self.spark_response_lateral_ratio

        self.spark_override_steps_left -= 1
        if self.spark_override_steps_left <= 0:
            self.spark_response_active = False

        return np.array([ax, ay, az, ayaw], dtype=np.float32)


    def _select_track_for_current_waypoint(self, tracks):
        if not tracks:
            self.current_locked_track_missed += 1
            if self.current_locked_track_missed > self.track_keep_max_missed:
                self.current_locked_track_id = None
                self.current_locked_waypoint_idx = -1
            return None

        prior = self._get_route_visual_prior()
        expected_cx = float(prior["expected_cx"])
        expected_area = float(prior["expected_area"])

        # 1) если track уже зафиксирован на этом waypoint — держим его
        if (
                self.current_locked_track_id is not None
                and self.current_locked_waypoint_idx == int(self.current_waypoint_idx)
        ):
            for trk in tracks:
                if int(trk["track_id"]) == int(self.current_locked_track_id):
                    self.current_locked_track_missed = 0
                    return trk

        # 2) иначе выбираем track, который лучше всего соответствует текущей target_point
        candidates = []
        for trk in tracks:
            cx_err = abs(float(trk["center_x"]) - expected_cx)
            area_ratio = max(
                float(trk["area"]) / max(expected_area, 1e-6),
                expected_area / max(float(trk["area"]), 1e-6),
            )

            if cx_err > self.track_route_gate_cx:
                continue
            if area_ratio > self.track_route_gate_area_ratio:
                continue

            score = 0.0
            score -= 2.0 * cx_err
            score -= 0.5 * abs(float(trk["area"]) - expected_area)
            candidates.append((score, trk))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x[0])[1]
        self.current_locked_track_id = int(best["track_id"])
        self.current_locked_waypoint_idx = int(self.current_waypoint_idx)
        self.current_locked_track_missed = 0
        return best

    def _select_detection_for_current_waypoint(self, detections, image_bgr):
        """
        Рабочий route-aware selection:
        - если Deep SORT недоступен -> fallback на старый anti-flicker
        - если доступен -> выбираем track для текущего waypoint
          и маппим его обратно на raw detection текущего кадра
        """
        if not detections:
            self._last_deep_sort_tracks = []
            self.current_locked_track_missed += 1
            if self.current_locked_track_missed > self.track_keep_max_missed:
                self.current_locked_track_id = None
                self.current_locked_waypoint_idx = -1
            return None

        if self.deep_sort_tracker is None or image_bgr is None:
            self._last_deep_sort_tracks = []
            return self._select_with_tracking(detections)

        tracks = self._update_deep_sort_tracks(detections, image_bgr)
        self._last_deep_sort_tracks = [dict(t) for t in tracks]
        selected_track = self._select_track_for_current_waypoint(tracks)

        if selected_track is None:
            # Fallback: не убиваем visual target, если route-aware gate оказался слишком жёстким.
            # Берём лучший raw detection, как в EMA-env.
            fallback_det = self._select_primary_detection(detections)

            if fallback_det is not None:
                self.current_locked_track_id = None
                self.current_locked_waypoint_idx = -1
                self.current_locked_track_missed = 0
                return dict(fallback_det)

            self.current_locked_track_missed += 1
            if self.current_locked_track_missed > self.track_keep_max_missed:
                self.current_locked_track_id = None
                self.current_locked_waypoint_idx = -1
            return None

        self.current_locked_track_missed = 0

        best_det = None
        best_score = -1e9

        trk_cx = float(selected_track["center_x"])
        trk_cy = float(selected_track["center_y"])
        trk_area = float(selected_track["area"])

        for det in detections:
            cx = float(det["center_x"])
            cy = float(det["center_y"])
            area = float(det["area"])

            center_dist = float(np.sqrt((cx - trk_cx) ** 2 + (cy - trk_cy) ** 2))
            area_diff = abs(area - trk_area)

            score = -3.0 * center_dist - 0.6 * area_diff
            if score > best_score:
                best_score = score
                best_det = dict(det)

        if best_det is None:
            return None

        best_det["track_id"] = int(selected_track["track_id"])
        return best_det

    def _get_geometry_hint_features(self) -> np.ndarray:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        _, _, yaw = airsim.to_eularian_angles(kin.orientation)

        pos = self._current_position_vec()
        meta = self._get_current_inspection_pair_meta()
        point_mode = str(meta.get("point_mode", "inspect")).lower()

        if point_mode == "inspect":
            ref_target = self._get_inspection_target().copy()
        else:
            if self.active_local_waypoint is not None:
                ref_target = self.active_local_waypoint.copy()
            else:
                ref_target = self.mission_route[self.current_waypoint_idx].copy()

        delta = ref_target - pos

        # world -> body frame
        cy = float(np.cos(yaw))
        sy = float(np.sin(yaw))

        body_x = cy * float(delta[0]) + sy * float(delta[1])
        body_y = -sy * float(delta[0]) + cy * float(delta[1])

        z_error = float(ref_target[2] - pos[2])

        desired_yaw = float(np.arctan2(float(delta[1]), float(delta[0])))
        yaw_error = float(self._wrap_angle(desired_yaw - float(yaw)))

        ref_body_x_norm = float(np.clip(body_x / max(self.world_xy_limit, 1e-6), -1.0, 1.0))
        ref_body_y_norm = float(np.clip(body_y / max(self.world_xy_limit, 1e-6), -1.0, 1.0))
        z_error_norm = float(np.clip(z_error / max(self.max_altitude, 1e-6), -1.0, 1.0))
        yaw_error_norm = float(np.clip(yaw_error / np.pi, -1.0, 1.0))

        return np.array(
            [ref_body_x_norm, ref_body_y_norm, z_error_norm, yaw_error_norm],
            dtype=np.float32,
        )


    def _handle_debug_output(self, debug_frame):


        if debug_frame is None:
            return

        if self.debug_show_window:
            window_name = "YOLO inspect debug"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)  # можете поставить свой размер
            cv2.imshow(window_name, debug_frame)
            cv2.waitKey(1)
        if self.debug_save_video:
            h, w = debug_frame.shape[:2]
            if self.debug_video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.debug_video_writer = cv2.VideoWriter(
                    self.debug_video_path,
                    fourcc,
                    max(1.0 / max(self.step_duration * self.debug_draw_every_n_steps, 1e-6), 1.0),
                    (w, h),
                )

            if self.debug_video_writer is not None:
                self.debug_video_writer.write(debug_frame)

    def print_current_route_target(self):
        if self.inspection_pairs is not None and len(self.inspection_pairs) > 0:
            idx = int(np.clip(self.current_waypoint_idx, 0, len(self.inspection_pairs) - 1))
            pair = self.inspection_pairs[idx]

            print(
                f"[CUR POINT] wp_idx={idx} "
                f"route_point={pair['route_point']} "
                f"target_point={pair['target_point']}"
            )
            return

        print("[CUR POINT] inspection pairs not set")

        print("[CUR POINT] inspection pairs not set")

    def _get_control_visual_features(self) -> np.ndarray:
        """
        Единый visual block для control/reward.
        - если реальный visual pipeline включён: реальные yolo/tracking features
        - если выключен: те же geometry hints, что policy видит в observation
        """
        if self.yolo_feature_dim <= 0:
            return np.zeros(0, dtype=np.float32)

        if self.enable_visual_inspection:
            block = np.zeros(max(self.yolo_feature_dim, 8), dtype=np.float32)
            n = min(self.yolo_feature_dim, self.yolo_features.shape[0])
            block[:n] = self.yolo_features[:n]
            return block[:self.yolo_feature_dim]

        return self._get_obs_visual_block()

    def _is_final_goal_ready(self):
        final_distance_raw = self._get_goal_distance_raw()
        is_final_wp = self.current_waypoint_idx >= len(self.mission_route) - 1
        near_final_goal = bool(is_final_wp and final_distance_raw < self.goal_tolerance)

        if not near_final_goal:
            return False, {
                "near_final_goal": 0.0,
                "goal_ready_for_reward": 0.0,
                "goal_visual_ready": 0.0,
                "goal_geom_ready": 0.0,
            }

        goal_meta = self._get_current_inspection_pair_meta()
        final_point_mode = str(goal_meta.get("point_mode", "inspect")).lower()
        final_requires_visual = bool(goal_meta.get("visual_gate_required", False))

        visible = float(self.smoothed_target_features[0]) if self.yolo_feature_dim > 0 else 0.0
        dx_center = float(self.smoothed_target_features[6]) if self.yolo_feature_dim > 6 else 0.0
        dy_center = float(self.smoothed_target_features[7]) if self.yolo_feature_dim > 7 else 0.0
        center_error = float(np.clip(np.sqrt(dx_center ** 2 + dy_center ** 2), 0.0, 1.5))

        _, _, target_delta_goal, _, _ = self._get_inspection_metrics()
        desired_yaw_goal = float(np.arctan2(target_delta_goal[1], target_delta_goal[0]))
        yaw_error_goal = abs(self._wrap_angle(desired_yaw_goal - float(self.last_current_yaw)))

        goal_visible_thr = float(goal_meta.get("visual_gate_visible_threshold", self.visual_gate_visible_threshold))
        goal_center_thr = float(goal_meta.get("visual_gate_center_threshold", self.visual_gate_center_threshold))
        goal_yaw_thr = float(goal_meta.get("visual_gate_yaw_threshold", self.visual_gate_yaw_threshold))

        goal_visual_ready = (
                visible > goal_visible_thr
                and center_error < goal_center_thr
                and yaw_error_goal < goal_yaw_thr
        )

        geom_ready = self._geom_inspect_ready_for_current_waypoint()
        goal_geom_ready = bool(geom_ready["ready"])

        if final_point_mode == "inspect":
            goal_ready_for_reward = goal_visual_ready if final_requires_visual else goal_geom_ready
        else:
            goal_ready_for_reward = True

        return bool(goal_ready_for_reward), {
            "near_final_goal": float(near_final_goal),
            "goal_ready_for_reward": float(goal_ready_for_reward),
            "goal_visual_ready": float(goal_visual_ready),
            "goal_geom_ready": float(goal_geom_ready),
        }

    def _get_final_inspection_target(self) -> np.ndarray:
        if self.inspection_pairs is None or len(self.inspection_pairs) == 0:
            raise RuntimeError("inspection_pairs REQUIRED for inspect env")
        return np.asarray(self.inspection_pairs[-1]["target_point"], dtype=np.float32).copy()
    def _get_obs_visual_block(self) -> np.ndarray:
        """
        Для observation:
        - размер держим стабильным;
        - если visual выключен, используем geometry hints ТОЛЬКО в obs,
          не записывая их в self.yolo_features.
        """
        block = np.zeros(max(self.yolo_feature_dim, 8), dtype=np.float32)

        if self.enable_visual_inspection and self.yolo_feature_dim > 0:
            n = min(self.yolo_feature_dim, block.shape[0])
            block[:n] = self.yolo_features[:n]
            return block[:self.yolo_feature_dim] if self.yolo_feature_dim > 0 else np.zeros(0, dtype=np.float32)

        if self.yolo_feature_dim > 0:
            geom_hints = self._get_geometry_hint_features()
            n = min(len(geom_hints), self.yolo_feature_dim)
            block[:n] = geom_hints[:n]
            return block[:self.yolo_feature_dim]

        return np.zeros(0, dtype=np.float32)

    def _get_cable_safety_state(self, mutate: bool = True) -> dict:
        step_i = int(getattr(self, "current_step", 0))
        if not mutate:
            cached_state = getattr(self, "_last_cable_safety_state", None)
            cached_step = int(getattr(self, "_last_cable_safety_state_step", -1))
            if isinstance(cached_state, dict) and cached_step == step_i:
                return dict(cached_state)

        seg = getattr(self, "_last_seg_info", {}) or {}

        seg_visible = float(seg.get("seg_visible", 0.0)) > 0.5
        cable_ratio = float(seg.get("seg_cable_ratio", 0.0))
        cable_dx = float(seg.get("seg_cable_center_dx", 0.0))
        cable_dy = float(seg.get("seg_cable_center_dy", 0.0))
        stability = float(seg.get("seg_mask_stability", 0.0))

        center_dist = float(np.sqrt(cable_dx ** 2 + cable_dy ** 2))

        cable_visible = bool(seg_visible and cable_ratio > 0.00025)

        soft_ratio_hit = bool(cable_ratio > float(self.cable_soft_ratio))
        hard_ratio_hit = bool(cable_ratio > float(self.cable_hard_ratio))
        soft_stability_ratio_hit = bool(
            stability > 0.30 and cable_ratio > float(self.cable_soft_ratio) * 0.55
        )
        hard_stability_ratio_hit = bool(
            stability > 0.40 and cable_ratio > float(self.cable_hard_ratio) * 0.70
        )
        soft_center_hit = bool(center_dist < float(self.cable_center_soft))
        hard_center_hit = bool(center_dist < float(self.cable_center_hard))

        cable_trigger_ratio_raw = bool(
            cable_visible
            and (
                soft_ratio_hit
                or hard_ratio_hit
                or soft_stability_ratio_hit
                or hard_stability_ratio_hit
            )
        )
        cable_trigger_center_raw = bool(
            cable_visible and (soft_center_hit or hard_center_hit)
        )

        soft = bool(
            cable_visible and (
                soft_ratio_hit
                or soft_center_hit
                or soft_stability_ratio_hit
            )
        )

        hard = bool(
            cable_visible and (
                hard_ratio_hit
                or hard_center_hit
                or hard_stability_ratio_hit
            )
        )

        cable_soft_raw = bool(soft)
        cable_hard_raw = bool(hard)

        hold_left_state = int(getattr(self, "_cable_hold_left", 0))
        hold_last_step_state = int(getattr(self, "_cable_hold_last_update_step", step_i))
        hold_hard_state = bool(getattr(self, "_cable_hold_hard", False))
        hold_dx_state = float(getattr(self, "_cable_hold_last_dx", 0.0))
        hold_dy_state = float(getattr(self, "_cable_hold_last_dy", 0.0))
        hold_ratio_state = float(getattr(self, "_cable_hold_last_ratio", 0.0))

        # ===== hold / hysteresis =====
        # ВАЖНО:
        # уменьшаем hold не при каждом вызове функции,
        # а только по реальному изменению current_step.
        if hard or soft:
            hold_left_new = int(getattr(self, "cable_safety_hold_steps", 35))
            hold_hard_new = bool(hard)
            hold_dx_new = float(cable_dx)
            hold_dy_new = float(cable_dy)
            hold_ratio_new = float(cable_ratio)
            hold_last_step_new = int(step_i)

        else:
            hold_left_new = int(hold_left_state)
            hold_hard_new = bool(hold_hard_state)
            hold_dx_new = float(hold_dx_state)
            hold_dy_new = float(hold_dy_state)
            hold_ratio_new = float(hold_ratio_state)
            hold_last_step_new = int(hold_last_step_state)

            if hold_left_new > 0:
                if hold_last_step_new < 0:
                    hold_last_step_new = step_i

                delta_steps = max(0, step_i - hold_last_step_new)

                if delta_steps > 0:
                    hold_left_new = max(0, hold_left_new - delta_steps)
                    hold_last_step_new = int(step_i)

                if hold_left_new > 0:
                    cable_visible = True
                    soft = True
                    hard = bool(hold_hard_new)

                    cable_dx = float(hold_dx_new)
                    cable_dy = float(hold_dy_new)
                    cable_ratio = float(hold_ratio_new)
                    center_dist = float(np.sqrt(cable_dx ** 2 + cable_dy ** 2))

        if mutate:
            self._cable_hold_left = int(hold_left_new)
            self._cable_hold_hard = bool(hold_hard_new)
            self._cable_hold_last_dx = float(hold_dx_new)
            self._cable_hold_last_dy = float(hold_dy_new)
            self._cable_hold_last_ratio = float(hold_ratio_new)
            self._cable_hold_last_update_step = int(hold_last_step_new)

        hold_left_final = int(hold_left_new)
        cable_soft_final = bool(soft)
        cable_hard_final = bool(hard)
        cable_trigger_hold = bool(
            (not (cable_soft_raw or cable_hard_raw))
            and (cable_soft_final or cable_hard_final)
            and hold_left_final > 0
        )

        state = {
            "visible": float(cable_visible),
            "ratio": float(cable_ratio),
            "dx": float(cable_dx),
            "dy": float(cable_dy),
            "center_dist": float(center_dist),
            "stability": float(stability),
            "soft": bool(cable_soft_final),
            "hard": bool(cable_hard_final),
            "hold_left": int(hold_left_final),
            "cable_trigger_ratio_dbg": float(cable_trigger_ratio_raw),
            "cable_trigger_center_dbg": float(cable_trigger_center_raw),
            "cable_soft_raw_dbg": float(cable_soft_raw),
            "cable_hard_raw_dbg": float(cable_hard_raw),
            "cable_soft_final_dbg": float(cable_soft_final),
            "cable_hard_final_dbg": float(cable_hard_final),
            "cable_hold_left_dbg": float(hold_left_final),
            "cable_trigger_hold_dbg": float(cable_trigger_hold),
        }

        if mutate:
            self._last_cable_safety_state = dict(state)
            self._last_cable_safety_state_step = int(step_i)

        return state
    def _post_reset_route_sync(self) -> None:
        if self.inspection_pairs is None or len(self.inspection_pairs) == 0:
            return

        route_points = np.asarray(
            [pair["route_point"] for pair in self.inspection_pairs],
            dtype=np.float32,
        ).copy()

        try:
            cur_pos = self._current_position_vec()
            current_z = float(cur_pos[2])

            if route_points.shape[0] > 0:
                original_first_z = float(route_points[0, 2])

                # AirSim NED:
                # меньше z => выше,
                # больше z => ниже.
                #
                # Первая route-точка не должна быть ниже текущей позиции
                # более чем на 0.25 м сразу после reset.
                desired_first_z = min(original_first_z, current_z + 0.25)
                z_shift = float(desired_first_z - original_first_z)

                if abs(z_shift) > 1e-4:
                    route_points[:, 2] += z_shift

                    for i in range(min(len(self.inspection_pairs), route_points.shape[0])):
                        self.inspection_pairs[i]["route_point"] = route_points[i].copy()

                        tp = np.asarray(
                            self.inspection_pairs[i]["target_point"],
                            dtype=np.float32,
                        ).copy()
                        tp[2] = float(tp[2] + z_shift)
                        self.inspection_pairs[i]["target_point"] = tp

                    print(
                        "[INSPECT POST RESET Z SHIFT]",
                        "current_z=", round(current_z, 3),
                        "original_first_z=", round(original_first_z, 3),
                        "desired_first_z=", round(desired_first_z, 3),
                        "z_shift=", round(z_shift, 3),
                    )

        except Exception as exc:
            print("[INSPECT POST RESET] route z safety failed:", exc)

        self.current_route_points = route_points
        self.current_waypoint_idx = 0
        self.goal_position = self.current_route_points[-1].copy()

        self._build_mission_route()
        self._rebuild_local_path(force=True)

        self.active_inspection_target = None

        print(
            "[INSPECT ROUTE SYNC]",
            "current_pos=", self._current_position_vec(),
            "wp_idx=", self.current_waypoint_idx,
            "active_local_waypoint=", self.active_local_waypoint,
            "first_route_point=", self.current_route_points[0],
            "goal=", self.goal_position,
        )
    def _reset_spark_info_for_empty_frame(self):
        self._last_spark_info = {
            "spark_visible": 0.0,
            "spark_det_count": 0,
            "spark_max_conf": 0.0,
            "spark_mean_conf": 0.0,
            "spark_max_area": 0.0,
            "spark_bbox_x1": 0.0,
            "spark_bbox_y1": 0.0,
            "spark_bbox_x2": 0.0,
            "spark_bbox_y2": 0.0,
            "spark_bbox_area": 0.0,
            "spark_center_dx": 0.0,
            "spark_center_dy": 0.0,
            "spark_event_score": 0.0,
            "spark_persistent": 0.0,
        }

    def _update_spark_debug_info(self, image_bgr: Optional[np.ndarray]):
        if self.spark_camera_name is None and image_bgr is None:
            self._reset_spark_info_for_empty_frame()
            return

        step = int(getattr(self, "current_step", 0))
        last_step = int(getattr(self, "_last_spark_infer_step", -10 ** 9))
        interval = max(1, int(getattr(self, "spark_inference_interval", 1)))
        if (step - last_step) < interval:
            return

        self._lazy_load_external_debug_models()
        if self._spark_model is None:
            self._reset_spark_info_for_empty_frame()
            return

        spark_frame_bgr = image_bgr
        if self.spark_camera_name is not None:
            spark_frame_bgr = self._get_scene_image_bgr(camera_name=self.spark_camera_name)

        if spark_frame_bgr is None:
            self._reset_spark_info_for_empty_frame()
            return

        h, w = spark_frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            self._reset_spark_info_for_empty_frame()
            return

        self._last_spark_infer_step = step
        self._spark_infer_count = int(getattr(self, "_spark_infer_count", 0)) + 1
        spark_t0 = time.perf_counter()
        try:
            results = self._spark_model.predict(
                source=spark_frame_bgr,
                conf=self.spark_conf_threshold,
                iou=self.spark_iou_threshold,
                imgsz=self.spark_input_size,
                verbose=False,
                device=self.spark_device,
            )
        except Exception:
            self._last_spark_infer_ms = 1000.0 * (time.perf_counter() - spark_t0)
            self._reset_spark_info_for_empty_frame()
            return
        self._last_spark_infer_ms = 1000.0 * (time.perf_counter() - spark_t0)

        if not results:
            self._reset_spark_info_for_empty_frame()
            return

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if (
                boxes is None
                or getattr(boxes, "xyxy", None) is None
                or getattr(boxes, "conf", None) is None
                or getattr(boxes, "cls", None) is None
        ):
            self._reset_spark_info_for_empty_frame()
            return

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        clss = boxes.cls.detach().cpu().numpy().astype(np.int32)

        n = min(len(xyxy), len(confs), len(clss))
        if n <= 0:
            self._reset_spark_info_for_empty_frame()
            return

        xyxy = xyxy[:n]
        confs = confs[:n]
        clss = clss[:n]

        det_count = 0
        max_conf = 0.0
        mean_conf = 0.0
        max_area = 0.0
        best_x1 = 0.0
        best_y1 = 0.0
        best_x2 = 0.0
        best_y2 = 0.0
        best_bbox_area = 0.0
        best_center_dx = 0.0
        best_center_dy = 0.0

        frame_area = float(max(h * w, 1))

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            if bw <= 1e-6 or bh <= 1e-6:
                continue

            class_name = str(names.get(int(cls_id), int(cls_id))).lower()

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area_norm = float((bw * bh) / frame_area)

            det_count += 1
            mean_conf += float(conf)
            if float(conf) > max_conf:
                max_conf = float(conf)
                max_area = area_norm
                best_x1 = float(x1)
                best_y1 = float(y1)
                best_x2 = float(x2)
                best_y2 = float(y2)
                best_bbox_area = float(bw * bh)
                best_center_dx = float(np.clip((cx / max(w, 1)) - 0.5, -1.0, 1.0))
                best_center_dy = float(np.clip((cy / max(h, 1)) - 0.5, -1.0, 1.0))

            if self.debug_print_seg:
                print(
                    f"[DEBUG SPARK DET] step={int(self.current_step)} "
                    f"cls_id={int(cls_id)} class_name='{class_name}' "
                    f"conf={float(conf):.3f} area_norm={area_norm:.6f}"
                )

        if det_count <= 0:
            self._reset_spark_info_for_empty_frame()
            self._spark_persist_steps = 0
            return

        mean_conf /= float(det_count)

        event_score = float(max_conf * max_area * 10.0)
        if max_conf >= max(0.25, self.spark_conf_threshold):
            self._spark_persist_steps += 1
        else:
            self._spark_persist_steps = 0

        spark_persistent = 1.0 if self._spark_persist_steps >= 2 else 0.0

        self._last_spark_info = {
            "spark_visible": 1.0,
            "spark_det_count": int(det_count),
            "spark_max_conf": float(max_conf),
            "spark_mean_conf": float(mean_conf),
            "spark_max_area": float(max_area),
            "spark_bbox_x1": float(best_x1),
            "spark_bbox_y1": float(best_y1),
            "spark_bbox_x2": float(best_x2),
            "spark_bbox_y2": float(best_y2),
            "spark_bbox_area": float(best_bbox_area),
            "spark_center_dx": float(best_center_dx),
            "spark_center_dy": float(best_center_dy),
            "spark_event_score": float(event_score),
            "spark_persistent": float(spark_persistent),
        }

        self.spark_steps_with_det += 1
        self.spark_event_count += int(det_count)
        self.spark_max_conf_sum += float(max_conf)
        self.spark_mean_conf_sum += float(mean_conf)
        self.spark_max_area_sum += float(max_area)
        self.spark_event_score_sum += float(event_score)
        self.spark_persistent_count += int(spark_persistent)
    def _decode_nav_obs(self, obs: np.ndarray) -> Dict[str, float]:
        return self._decode_base_obs(obs)

    def _get_collision_flag_raw(self) -> float:
        try:
            return float(self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided)
        except Exception:
            return 0.0

    def _get_altitude_norm_raw(self) -> float:
        pos = self._current_position_vec()
        altitude = float(-pos[2])
        return float(np.clip(altitude / max(self.max_altitude, 1e-6), 0.0, 1.0))

    def _get_final_distance_norm_raw(self) -> float:
        max_goal_dist = np.sqrt(
            self.world_xy_limit ** 2 +
            self.world_xy_limit ** 2 +
            self.max_altitude ** 2
        )
        return float(np.clip(self._get_goal_distance_raw() / max(max_goal_dist, 1e-6), 0.0, 1.0))

    def _publish_control_stream_command(
            self,
            vx: float,
            vy: float,
            vz: float,
            yaw_rate: float,
    ) -> None:
        if not getattr(self, "control_stream_enabled", False):
            return

        with self._control_stream_lock:
            self._latest_stream_cmd[:] = np.array(
                [float(vx), float(vy), float(vz), float(yaw_rate)],
                dtype=np.float32,
            )
            self._latest_stream_cmd_time = time.perf_counter()

    def _start_control_stream_worker(self) -> None:
        if not getattr(self, "control_stream_enabled", False):
            return

        if self._control_stream_thread is not None and self._control_stream_thread.is_alive():
            return

        self._control_stream_stop_event = threading.Event()

        with self._control_stream_lock:
            self._latest_stream_cmd[:] = 0.0
            self._latest_stream_cmd_time = time.perf_counter()

        self._control_stream_thread = threading.Thread(
            target=self._control_stream_loop,
            daemon=True,
        )
        self._control_stream_thread.start()
        print(
            "[CONTROL STREAM] started",
            "hz=", float(self.control_stream_hz),
            "stale_sec=", float(self.control_stream_stale_sec),
            "duration=", float(self.control_stream_command_duration),
        )

    def _stop_control_stream_worker(self) -> None:
        stop_event = getattr(self, "_control_stream_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        thread = getattr(self, "_control_stream_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        self._control_stream_thread = None
        self._control_stream_stop_event = None

        try:
            client = getattr(self, "_control_stream_client", None)
            if client is not None:
                client.moveByVelocityAsync(
                    0.0,
                    0.0,
                    0.0,
                    0.15,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                    vehicle_name=self.vehicle_name,
                )
        except Exception:
            pass

    def _control_stream_loop(self) -> None:
        period = 1.0 / max(float(self.control_stream_hz), 1.0)
        stream_duration = max(
            float(self.control_stream_command_duration),
            period * 2.5,
            0.12,
        )

        try:
            client = airsim.MultirotorClient(ip=self.ip_address)
            client.confirmConnection()
            client.enableApiControl(True, self.vehicle_name)
            client.armDisarm(True, self.vehicle_name)
            self._control_stream_client = client
        except Exception as exc:
            print("[CONTROL STREAM] init failed:", repr(exc))
            return

        last_error_print = 0.0

        while True:
            stop_event = getattr(self, "_control_stream_stop_event", None)
            if stop_event is None or stop_event.is_set():
                break

            now = time.perf_counter()

            with self._control_stream_lock:
                cmd = self._latest_stream_cmd.copy()
                cmd_age = now - float(self._latest_stream_cmd_time)

            # Если основной PPO/YOLO step завис — не летим по старой команде.
            # Продолжаем слать API-команды, но с нулевой скоростью.
            if cmd_age > float(self.control_stream_stale_sec):
                cmd[:] = 0.0

            send_ok = False

            try:
                client.moveByVelocityAsync(
                    vx=float(cmd[0]),
                    vy=float(cmd[1]),
                    vz=float(cmd[2]),
                    duration=float(stream_duration),
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(cmd[3])),
                    vehicle_name=self.vehicle_name,
                )
                send_ok = True

            except Exception as exc:
                self._control_stream_error_count = int(getattr(self, "_control_stream_error_count", 0)) + 1

                if now - last_error_print > 1.0:
                    print("[CONTROL STREAM] moveByVelocityAsync failed:", repr(exc))
                    last_error_print = now

                try:
                    client.enableApiControl(True, self.vehicle_name)
                    client.armDisarm(True, self.vehicle_name)
                except Exception:
                    pass

            if send_ok:
                self._control_stream_last_send_time = time.perf_counter()
                self._control_stream_sent_count = int(getattr(self, "_control_stream_sent_count", 0)) + 1
                self._control_stream_last_cmd_age = float(cmd_age)

            time.sleep(period)

    def _mask_final_goal_from_base_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """
        Скрываем от policy финальную цель маршрута.
        В base nav obs это индексы:
          13,14,15 -> final_dx/final_dy/final_dz
          17       -> final_distance_norm
        """
        if not getattr(self, "hide_final_goal_in_obs", False):
            return base_obs

        masked = np.asarray(base_obs, dtype=np.float32).copy()

        if masked.shape[0] > 17:
            masked[13] = 0.0
            masked[14] = 0.0
            masked[15] = 0.0
            masked[17] = 0.0

        return masked
