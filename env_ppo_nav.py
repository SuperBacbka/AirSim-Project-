import time
from typing import Optional, Dict, Any, List
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
from deep_sort_realtime.deepsort_tracker import DeepSort

class AirSimDroneEnv(gym.Env):
    metadata = {"render_modes": []}
    _NAV_OBS_INDEX = {
        "vx": 3,
        "vy": 4,
        "vz": 5,
        "altitude_norm": 9,
        "local_dz": 12,
        "local_distance_norm": 16,
        "final_distance_norm": 17,
        "heading_local_cos": 19,
        "route_deviation_norm": 21,
        "planner_hint_x": 22,
        "planner_hint_y": 23,
        "planner_hint_z": 24,
        "depth_left": 25,
        "depth_center": 26,
        "depth_right": 27,
        "depth_up": 28,
        "risk_front": 31,
        "risk_left": 32,
        "risk_right": 33,
        "risk_up": 34,
        "risk_down": 35,
        "collision": 36,
    }

    def __init__(
        self,
        vehicle_name: str = "SimpleFlight",
        nav_visual_obs_mode: str = "safety_only",
        start_position=None,
        start_points=None,
        fixed_goal=None,
        goal_points=None,
        route_points=None,
        route_sets=None,
        ip_address: str = "127.0.0.1",
        step_duration: float = 0.03,
        action_repeat: int = 1,
        control_command_duration: Optional[float] = None,
        max_episode_steps: int = 400,
        max_velocity: float = 3.0,
        max_yaw_rate: float = 35.0,
        max_sensor_distance: float = 5.0,
        depth_max_distance: float = 12.0,
        goal_tolerance: float = 0.3,
        waypoint_tolerance: float = 0.5,
        min_altitude: float = 1.0,
        max_altitude: float = 15.0,
        world_xy_limit: float = 60.0,
        min_start_goal_distance: float = 2.0,
        random_goal: bool = True,
        use_route_planner: bool = True,
        planner_grid_resolution: float = 1.5,
        planner_grid_half_size: int = 10,
        planner_vertical_resolution: Optional[float] = None,
        planner_vertical_half_size: Optional[int] = None,
        obstacle_memory_ttl_steps: int = 30,
        planner_replan_interval: int = 6,
        planner_lookahead_steps: int = 2,
        planner_shift_distance: float = 2.5,
        planner_shift_hold_steps: int = 14,
        soft_avoidance_fail_steps: int = 8,
        use_depth_image: bool = True,
        depth_camera_name: str = "0",
        reserve_yolo_features: bool = True,
        yolo_feature_dim: int = 8,
        inactive_speed_threshold: float = 0.05,
        inactive_action_threshold: float = 0.08,
        inactive_penalty_weight: float = 0.03,
        inactive_max_steps: int = 20,
        no_progress_threshold: float = 0.0008,
        no_progress_max_steps: int = 25,
        active_progress_reward_weight: float = 0.01,
        enable_nav_visual=False,
        enable_nav_seg_context = False,
        enable_nav_safety_detector = False,
        visual_hazard_replan_enabled = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        scene_camera_name: str = "0",
        scene_image_type=airsim.ImageType.Scene,
            action_smoothing_alpha: float = 0.40,
            min_xy_cmd_ratio: float = 0.03,
            cmd_filter_alpha_xy: float = 0.72,
            cmd_filter_alpha_z: float = 0.55,
            cmd_filter_alpha_yaw: float = 0.55,
            max_delta_vxy_cmd: float = 0.45,
            max_delta_vz_cmd: float = 0.18,
            max_delta_yaw_rate_cmd: float = 3.5,
            altitude_kp: float = 0.8,
            altitude_kd: float = 0.6,
            altitude_deadband_m: float = 0.08,
            altitude_action_z_offset: float = 0.08,
            altitude_assist_mix: float = 0.40,
            max_vertical_speed_ratio: float = 0.25,
            yaw_rate_limit_deg: float = 6.0,
            nav_use_body_frame_control: bool = True,
            nav_fixed_yaw_enabled: bool = True,
            nav_fixed_yaw_deadband_xy_m: float = 1.2,
            nav_fixed_yaw_max_step_deg: float = 10.0,
            nav_apply_controller_gains: bool = True,

            nav_velocity_kp_xy: float = 0.16,
            nav_velocity_kd_xy: float = 0.02,
            nav_velocity_kp_z: float = 2.0,
            nav_velocity_ki_z: float = 0.0,
            nav_velocity_kd_z: float = 0.10,

            nav_angle_level_kp_rp: float = 2.2,
            nav_angle_level_kd_rp: float = 0.20,
            nav_angle_level_kp_yaw: float = 1.2,
            nav_angle_level_kd_yaw: float = 0.10,

            nav_angle_rate_kp_rp: float = 0.20,
            nav_angle_rate_kd_rp: float = 0.02,
            nav_angle_rate_kp_yaw: float = 0.12,
            nav_angle_rate_kd_yaw: float = 0.01,

        safety_detector_model_path: Optional[str] = None,
        safety_conf_threshold: float = 0.30,
        safety_target_class_names=None,

        infra_detector_model_path: Optional[str] = None,
        infra_model_type: str = "segmentation",
        infra_conf_threshold: float = 0.25,
        infra_target_class_names=None,
        paired_start_route: bool = False,
        nav_visual_input_size: int = 640,
        nav_visual_iou_threshold: float = 0.45,
        world_xy_oob_margin: float = 1.0,
        visual_hazard_threshold: float = 0.42,
        visual_hazard_clear_steps: int = 8,
        visual_replan_hold_steps: int = 18,
        visual_replan_shift_distance: float = 2.5,
        visual_hazard_xy_scale: float = 0.75,
        enable_goal_progress_reward: bool = False,
        goal_progress_weight: float = 0.02,
        goal_reached_bonus: float = 8.0,
        nav_use_deep_sort: bool = True,
        nav_deep_sort_max_age: int = 8,
        nav_deep_sort_n_init: int = 2,
        nav_deep_sort_max_iou_distance: float = 0.7,
        nav_track_keep_max_missed: int = 12,

        **kwargs,
    ):
        super().__init__()
        self.world_xy_oob_margin = float(world_xy_oob_margin)
        self.vehicle_name = vehicle_name
        self.ip_address = ip_address
        self.step_duration = step_duration
        self.control_command_duration = (
            None if control_command_duration is None else float(control_command_duration)
        )
        self.last_command_duration = 0.0
        self._last_move_task = None
        self.max_episode_steps = max_episode_steps
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.max_sensor_distance = max_sensor_distance
        self.depth_max_distance = depth_max_distance
        self.goal_tolerance = goal_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.world_xy_limit = world_xy_limit
        self.min_start_goal_distance = min_start_goal_distance
        self.random_goal = random_goal
        self.use_route_planner = use_route_planner
        self.planner_grid_resolution = planner_grid_resolution
        self.planner_grid_half_size = planner_grid_half_size
        self.planner_replan_interval = planner_replan_interval
        self.planner_lookahead_steps = planner_lookahead_steps
        self.planner_shift_distance = planner_shift_distance
        self.disable_soft_replan_for_testing = False
        self.current_route_idx = -1
        # Backward-compatible aliases for older training scripts.
        self.planner_vertical_resolution = (
            planner_grid_resolution if planner_vertical_resolution is None else float(planner_vertical_resolution)
        )
        self.planner_vertical_half_size = (
            planner_grid_half_size if planner_vertical_half_size is None else int(planner_vertical_half_size)
        )

        self.scene_camera_name = scene_camera_name
        self.scene_image_type = scene_image_type
        self.nav_visual_obs_mode = str(nav_visual_obs_mode).lower()
        self.safety_detector_model_path = safety_detector_model_path
        self.safety_conf_threshold = float(safety_conf_threshold)
        self.safety_target_class_names = (
            None if safety_target_class_names is None
            else {str(x).lower() for x in safety_target_class_names}
        )

        self.infra_detector_model_path = infra_detector_model_path
        self.infra_model_type = str(infra_model_type)
        self.infra_conf_threshold = float(infra_conf_threshold)
        self.infra_target_class_names = (
            None if infra_target_class_names is None
            else {str(x).lower() for x in infra_target_class_names}
        )

        self.nav_visual_input_size = int(nav_visual_input_size)
        self.nav_visual_iou_threshold = float(nav_visual_iou_threshold)

        self.visual_hazard_threshold = float(visual_hazard_threshold)
        self.visual_hazard_clear_steps = int(visual_hazard_clear_steps)
        self.visual_replan_hold_steps = int(visual_replan_hold_steps)
        self.visual_replan_shift_distance = float(visual_replan_shift_distance)
        self.visual_hazard_xy_scale = float(visual_hazard_xy_scale)

        self.nav_use_deep_sort = bool(nav_use_deep_sort)
        self.nav_deep_sort_max_age = int(nav_deep_sort_max_age)
        self.nav_deep_sort_n_init = int(nav_deep_sort_n_init)
        self.nav_deep_sort_max_iou_distance = float(nav_deep_sort_max_iou_distance)
        self.nav_track_keep_max_missed = int(nav_track_keep_max_missed)

        self.nav_deep_sort_tracker = None
        self.nav_last_tracks = []
        self.nav_locked_track_id = None
        self.nav_locked_track_missed = 0
        self.nav_locked_waypoint_idx = -1


        self._safety_model = None
        self._infra_model = None
        self._nav_visual_backend_ready = False
        self._last_nav_detections = []
        self._last_nav_selected = None
        self.safety_features = np.zeros(8, dtype=np.float32)
        self.infra_features = np.zeros(8, dtype=np.float32)
        self.paired_start_route = bool(paired_start_route)
        self.visual_hazard_active = False
        self.visual_hazard_score = 0.0
        self.visual_hazard_confirm_steps = 0
        self.visual_hazard_clear_counter = 0
        self.visual_replan_active = False
        self.visual_replan_steps_left = 0

        self.obstacle_memory_ttl_steps = int(obstacle_memory_ttl_steps)
        self.extra_compat_kwargs = dict(kwargs) if kwargs else {}
        self.debug_print_reset = bool(kwargs.pop("debug_print_reset", False))
        self.planner_shift_hold_steps = planner_shift_hold_steps
        self.soft_avoidance_fail_steps = soft_avoidance_fail_steps
        self.use_depth_image = use_depth_image
        self.depth_camera_name = depth_camera_name
        self.depth_image_interval = int(kwargs.pop("depth_image_interval", 3))
        self._last_depth_features = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._last_depth_dists = np.ones(5, dtype=np.float32)
        self.last_depth_read_time = 0.0
        self.last_depth_used_cache = 0.0
        self.reserve_yolo_features = reserve_yolo_features
        self.yolo_feature_dim = yolo_feature_dim if reserve_yolo_features else 0
        self.inactive_speed_threshold = float(inactive_speed_threshold)
        self.inactive_action_threshold = float(inactive_action_threshold)
        self.inactive_penalty_weight = float(inactive_penalty_weight)
        self.inactive_max_steps = int(inactive_max_steps)
        self.no_progress_threshold = float(no_progress_threshold)
        self.no_progress_max_steps = int(no_progress_max_steps)
        self.active_progress_reward_weight = float(active_progress_reward_weight)
        self.verbose = verbose

        self.xy_deadzone = 0.06
        self.start_position = self._normalize_point(start_position, "start_position")
        self.start_points = self._normalize_points(start_points, "start_points")
        self.fixed_goal = self._normalize_point(fixed_goal, "fixed_goal")
        self.goal_points = self._normalize_points(goal_points, "goal_points")
        self.route_points = self._normalize_points(route_points, "route_points")
        self.route_sets = None
        if route_sets is not None:
            self.route_sets = []
            for i, route in enumerate(route_sets):
                arr = np.asarray(route, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(f"route_sets[{i}] must be shape (N, 3), got {arr.shape}")
                self.route_sets.append(arr.copy())

        self.current_route_points = None

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.client = airsim.MultirotorClient(ip=self.ip_address)
        self.client.confirmConnection()

        self.current_step = 0
        self.goal_position = self.fixed_goal.copy() if self.fixed_goal is not None else np.array([0.0, 0.0, -3.0], dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_prev_action = np.zeros(4, dtype=np.float32)

        # ===== CAPS / ASAP-like smoothness weights =====
        # CAPS-like: штраф за delta action
        self.caps_w_xy = 0.015
        self.caps_w_z = 0.004
        self.caps_w_yaw = 0.016

        # ASAP-like: штраф за second difference / jerk
        self.asap_w_xy = 0.008
        self.asap_w_z = 0.00
        self.asap_w_yaw = 0.010
        self.prev_local_distance = None
        self.prev_final_distance = None
        self.prev_min_obstacle_dist = None
        self.episode_reward = 0.0
        self.episode_index = 0

        self.action_smoothing_alpha = float(action_smoothing_alpha)
        self.smoothed_action = np.zeros(4, dtype=np.float32)

        # ===== Altitude assist PID/PD =====
        self.altitude_pd_enabled = True

        # ===== command filter: light EMA + slew-rate limiter =====
        self.min_xy_cmd_ratio = float(min_xy_cmd_ratio)

        self.cmd_filter_alpha_xy = float(cmd_filter_alpha_xy)
        self.cmd_filter_alpha_z = float(cmd_filter_alpha_z)
        self.cmd_filter_alpha_yaw = float(cmd_filter_alpha_yaw)

        # максимум изменения команды за 1 env-step
        self.max_delta_vxy_cmd = float(max_delta_vxy_cmd)
        self.max_delta_vz_cmd = float(max_delta_vz_cmd)
        self.max_delta_yaw_rate_cmd = float(max_delta_yaw_rate_cmd)

        # внутренние состояния фильтра
        self.filtered_vx_cmd = 0.0
        self.filtered_vy_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.filtered_yaw_rate_cmd = 0.0

        # Базовые коэффициенты — стартовые, безопасные
        self.altitude_kp = float(altitude_kp)
        self.altitude_kd = float(altitude_kd)

        self.altitude_action_z_offset = float(altitude_action_z_offset)

        # Ограничения и сглаживание
        self.altitude_int_error = 0.0
        self.altitude_int_clip = 1.0
        self.altitude_deadband_m = float(altitude_deadband_m)
        self.altitude_assist_mix = float(altitude_assist_mix)
        self.max_vertical_speed = self.max_velocity * float(max_vertical_speed_ratio)

        self.yaw_rate_limit_deg = float(yaw_rate_limit_deg)

        self.heading_cos_sum = 0.0
        self.min_front_dist = 1.0
        self.min_left_dist = 1.0
        self.min_right_dist = 1.0
        self.min_up_dist = 1.0
        self.min_down_dist = 1.0
        self.speed_sum = 0.0

        self.front_avoid_enabled = False
        self.front_avoid_soft = 0.45
        self.front_avoid_hard = 0.20
        self.front_avoid_speed_scale = 0.55
        self.front_avoid_counter = 0

        self.collision_latched = False
        self.ground_contact_steps = 0
        self.ground_contact_down_threshold = 0.80
        self.ground_contact_speed_threshold = 1.0
        self.ground_contact_steps_required = 2


        self.mission_route: List[np.ndarray] = []
        self.current_waypoint_idx = 0
        self.active_local_waypoint = None
        self.local_path_world: List[np.ndarray] = []
        self.local_path_index = 0
        self.route_total_length = 1.0
        self.failed_progress_steps = 0
        self.shift_mode_steps_left = 0
        self.shift_vector = np.zeros(3, dtype=np.float32)
        self.last_progress_delta = 0.0
        self.last_action_magnitude = 0.0
        self.last_speed_norm = 0.0
        self.inactive_steps = 0
        self.no_progress_steps = 0
        self.yolo_features = np.zeros(self.yolo_feature_dim, dtype=np.float32)


        self.episode_stats = {}

        self.enable_goal_progress_reward = bool(enable_goal_progress_reward)
        self.goal_progress_weight = float(goal_progress_weight)
        self.goal_reached_bonus = float(goal_reached_bonus)

        # ===== Debug applied command trace =====
        self.last_raw_action                = np.zeros(4, dtype=np.float32)
        self.last_smoothed_action           = np.zeros(4, dtype=np.float32)
        self.enable_nav_visual              = bool(enable_nav_visual)
        self.enable_nav_seg_context         = bool(enable_nav_seg_context)
        self.enable_nav_safety_detector     = bool(enable_nav_safety_detector)
        self.visual_hazard_replan_enabled   = bool(visual_hazard_replan_enabled)

        self.nav_use_body_frame_control = bool(nav_use_body_frame_control)
        self.nav_fixed_yaw_enabled = bool(nav_fixed_yaw_enabled)
        self.nav_fixed_yaw_deadband_xy_m = float(nav_fixed_yaw_deadband_xy_m)
        self.nav_fixed_yaw_max_step_deg = float(nav_fixed_yaw_max_step_deg)
        self.nav_apply_controller_gains = bool(nav_apply_controller_gains)

        self.nav_velocity_kp_xy = float(nav_velocity_kp_xy)
        self.nav_velocity_kd_xy = float(nav_velocity_kd_xy)
        self.nav_velocity_kp_z = float(nav_velocity_kp_z)
        self.nav_velocity_ki_z = float(nav_velocity_ki_z)
        self.nav_velocity_kd_z = float(nav_velocity_kd_z)

        self.nav_angle_level_kp_rp = float(nav_angle_level_kp_rp)
        self.nav_angle_level_kd_rp = float(nav_angle_level_kd_rp)
        self.nav_angle_level_kp_yaw = float(nav_angle_level_kp_yaw)
        self.nav_angle_level_kd_yaw = float(nav_angle_level_kd_yaw)

        self.nav_angle_rate_kp_rp = float(nav_angle_rate_kp_rp)
        self.nav_angle_rate_kd_rp = float(nav_angle_rate_kd_rp)
        self.nav_angle_rate_kp_yaw = float(nav_angle_rate_kp_yaw)
        self.nav_angle_rate_kd_yaw = float(nav_angle_rate_kd_yaw)

        self.filtered_yaw_target_deg = 0.0

        self.last_applied_vx_cmd            = 0.0
        self.last_applied_vy_cmd            = 0.0
        self.last_applied_vz_cmd            = 0.0
        self.last_applied_yaw_rate_cmd      = 0.0
        self.prev_applied_cmd               = np.zeros(4, dtype=np.float32)
        self.prev_prev_applied_cmd          = np.zeros(4, dtype=np.float32)
        self.action_repeat                  = max(1, int(action_repeat))
        self.last_current_z                 = 0.0
        self.last_current_vz                = 0.0
        self.last_target_z = 0.0
        self.last_z_error = 0.0
        self._dbg_local_target_changed_this_step = False
        self._dbg_local_path_rebuilt_this_step = False
        self._dbg_progress_guard_active_this_step = False
        self._dbg_planner_replan_step_this_step = False
        self._dbg_prev_local_distance_before_reward = 0.0
        self._dbg_local_distance_raw_before_reward = 0.0
        self._dbg_progress_delta_raw_before_reward = 0.0

        self._cached_multirotor_state = None
        self._cached_position_vec = None
        self.last_current_roll = 0.0
        self.last_current_pitch = 0.0
        self.last_current_yaw = 0.0
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self._setup_nav_visual_pipeline()
        base_obs_dim = 41
        self.obs_dim = base_obs_dim + self.yolo_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        print("DEBUG ENV:",
              "lookahead=", self.planner_lookahead_steps,
              "max_delta_vxy_cmd=", self.max_delta_vxy_cmd,
              "max_delta_yaw_rate_cmd=", self.max_delta_yaw_rate_cmd)
    # =========================================================
    # Input normalization
    # =========================================================

    def _normalize_point(self, point, name: str):
        if point is None:
            return None
        arr = np.asarray(point, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != 3:
            raise ValueError(f"{name} must be shape (3,), got shape {arr.shape}")
        return arr.copy()

    def _normalize_points(self, points, name: str):
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(f"{name} must be shape (N, 3) or (3,), got shape {arr.shape}")
            arr = arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must be shape (N, 3), got shape {arr.shape}")
        return arr.copy()

    # =========================================================
    # YOLO placeholders
    # =========================================================

    def reset_yolo_features(self):
        if self.yolo_feature_dim > 0:
            self.yolo_features[:] = 0.0

        if hasattr(self, "safety_features"):
            self.safety_features[:] = 0.0

        if hasattr(self, "infra_features"):
            self.infra_features[:] = 0.0

    def update_yolo_features(
        self,
        target_visible: float = 0.0,
        target_class_id_norm: float = 0.0,
        target_conf: float = 0.0,
        target_center_x: float = 0.0,
        target_center_y: float = 0.0,
        target_area: float = 0.0,
        target_dx_from_center: float = 0.0,
        target_dy_from_center: float = 0.0,
    ):
        if self.yolo_feature_dim == 0:
            return
        vals = np.array([
            target_visible,
            target_class_id_norm,
            target_conf,
            target_center_x,
            target_center_y,
            target_area,
            target_dx_from_center,
            target_dy_from_center,
        ], dtype=np.float32)
        self.yolo_features[: min(self.yolo_feature_dim, vals.shape[0])] = vals[: self.yolo_feature_dim]

    # =========================================================
    # Gymnasium API
    # =========================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.prev_prev_action = np.zeros(4, dtype=np.float32)

        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self.prev_local_distance = None
        self.prev_final_distance = None
        self.prev_min_obstacle_dist = None
        self.episode_reward = 0.0
        self.failed_progress_steps = 0
        self.shift_mode_steps_left = 0
        self.shift_vector[:] = 0.0
        self.last_progress_delta = 0.0
        self.last_action_magnitude = 0.0
        self.last_speed_norm = 0.0
        self.inactive_steps = 0
        self.no_progress_steps = 0
        self.reset_yolo_features()
        self.altitude_int_error = 0.0
        self.heading_cos_sum = 0.0
        self.min_front_dist = 1.0
        self.min_left_dist = 1.0
        self.min_right_dist = 1.0
        self.min_up_dist = 1.0
        self.min_down_dist = 1.0
        self.speed_sum = 0.0
        self.last_raw_action = np.zeros(4, dtype=np.float32)
        self.last_smoothed_action = np.zeros(4, dtype=np.float32)
        self.nav_locked_track_id = None
        self.nav_locked_track_missed = 0
        self.nav_locked_waypoint_idx = -1
        self.nav_last_tracks = []
        self.visual_hazard_active = False
        self.visual_hazard_score = 0.0
        self.visual_hazard_confirm_steps = 0
        self.visual_hazard_clear_counter = 0
        self.visual_replan_active = False
        self.visual_replan_steps_left = 0
        self.safety_features[:] = 0.0
        self.infra_features[:] = 0.0
        self.yolo_features[:] = 0.0
        self._last_nav_detections = []
        self._last_nav_selected = None

        self.last_applied_vx_cmd = 0.0
        self.last_applied_vy_cmd = 0.0
        self.last_applied_vz_cmd = 0.0
        self.last_applied_yaw_rate_cmd = 0.0
        self.front_avoid_counter = 0
        self.filtered_vx_cmd = 0.0
        self.filtered_vy_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.filtered_yaw_rate_cmd = 0.0

        self.last_current_z = 0.0
        self.last_current_vz = 0.0
        self.last_target_z = 0.0
        self.last_z_error = 0.0

        self.episode_stats = {
            "progress_reward_sum": 0.0,
            "progress_reward_abs_sum": 0.0,
            "goal_reward_sum": 0.0,
            "goal_reward_abs_sum": 0.0,
            "waypoint_reward_sum": 0.0,
            "waypoint_reward_abs_sum": 0.0,
            "collision_penalty_sum": 0.0,
            "collision_penalty_abs_sum": 0.0,
            "obstacle_penalty_sum": 0.0,
            "obstacle_penalty_abs_sum": 0.0,
            "obstacle_clearance_reward_sum": 0.0,
            "obstacle_clearance_reward_abs_sum": 0.0,
            "frontal_block_penalty_sum": 0.0,
            "frontal_block_penalty_abs_sum": 0.0,
            "action_penalty_sum": 0.0,
            "action_penalty_abs_sum": 0.0,
            "altitude_penalty_sum": 0.0,
            "altitude_penalty_abs_sum": 0.0,
            "local_z_error_penalty_sum": 0.0,
            "local_z_error_penalty_abs_sum": 0.0,
            "local_z_hold_reward_sum": 0.0,
            "local_z_hold_reward_abs_sum": 0.0,
            "forward_facing_reward_sum": 0.0,
            "forward_facing_reward_abs_sum": 0.0,
            "route_deviation_penalty_sum": 0.0,
            "route_deviation_penalty_abs_sum": 0.0,
            "planner_alignment_reward_sum": 0.0,
            "planner_alignment_reward_abs_sum": 0.0,
            "risk_penalty_sum": 0.0,
            "risk_penalty_abs_sum": 0.0,
            "inactive_penalty_sum": 0.0,
            "inactive_penalty_abs_sum": 0.0,
            "active_progress_reward_sum": 0.0,
            "active_progress_reward_abs_sum": 0.0,
            "step_penalty_sum": 0.0,
            "step_penalty_abs_sum": 0.0,
            "caps_smooth_penalty_sum": 0.0,
            "caps_smooth_penalty_abs_sum": 0.0,
            "asap_smooth_penalty_sum": 0.0,
            "asap_smooth_penalty_abs_sum": 0.0,

        }

        self._hard_reset_and_place_drone()

        if self.paired_start_route:
            self.start_position, self.current_route_idx, self.current_route_points = self._sample_start_and_route_pair()
            if self.current_route_points is not None:
                self.goal_position = self.current_route_points[-1].copy()
            else:
                self.goal_position = self._resolve_fixed_goal()
        else:
            self.start_position = self._sample_start_position()

            if self.route_sets is not None and len(self.route_sets) > 0:
                self.current_route_idx = int(self.np_random.integers(0, len(self.route_sets)))
                self.current_route_points = self.route_sets[self.current_route_idx].copy()
                self.goal_position = self.current_route_points[-1].copy()
            else:
                self.current_route_idx = -1
                self.current_route_points = self.route_points.copy() if self.route_points is not None else None
                self.goal_position = self._sample_goal_position() if self.random_goal else self._resolve_fixed_goal()
        if self.start_position is not None:
            attempts = 0
            while np.linalg.norm(
                    self.goal_position - self.start_position) < self.min_start_goal_distance and attempts < 30:
                if self.paired_start_route:
                    self.start_position, self.current_route_idx, self.current_route_points = self._sample_start_and_route_pair()
                    if self.current_route_points is not None:
                        self.goal_position = self.current_route_points[-1].copy()
                elif self.route_sets is not None and len(self.route_sets) > 0:
                    self.current_route_idx = int(self.np_random.integers(0, len(self.route_sets)))
                    self.current_route_points = self.route_sets[self.current_route_idx].copy()
                    self.goal_position = self.current_route_points[-1].copy()
                else:
                    self.goal_position = self._sample_goal_position()
                attempts += 1

        self._move_drone_to_position(self.start_position)
        self._apply_nav_controller_gains()
        self._sync_nav_fixed_yaw_target()
        self._invalidate_state_cache()
        self.start_z_ref = float(self._current_position_vec()[2])

        print("DEBUG start_position:", self.start_position)
        print("DEBUG current_route_points:", self.current_route_points)
        print("DEBUG goal_position:", self.goal_position)

        self._build_mission_route()
        self._rebuild_local_path(force=True)

        self._post_reset_route_sync()
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0
        self.smoothed_action[:] = 0.0

        self.last_applied_vx_cmd = 0.0
        self.last_applied_vy_cmd = 0.0
        self.last_applied_vz_cmd = 0.0
        self.last_applied_yaw_rate_cmd = 0.0

        self.collision_latched = False
        self.ground_contact_steps = 0



        self.prev_applied_cmd[:] = 0.0
        self.prev_prev_applied_cmd[:] = 0.0

        obs = self._get_obs()
        self.prev_local_distance = self._get_local_distance_raw()
        self.prev_final_distance = self._get_goal_distance_raw()

        info = {
            "goal_position": self.goal_position.copy(),
            "step": self.current_step,
            "planner_info": self._get_planner_info(),
            "route_idx": int(self.current_route_idx),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        self.last_raw_action = action.copy()
        self._dbg_local_target_changed_this_step = False
        self._dbg_local_path_rebuilt_this_step = False
        self._dbg_progress_guard_active_this_step = False
        self._dbg_planner_replan_step_this_step = False
        self._dbg_prev_local_distance_before_reward = 0.0
        self._dbg_local_distance_raw_before_reward = 0.0
        self._dbg_progress_delta_raw_before_reward = 0.0

        # ===== Action smoothing: один раз на policy-step =====
        self.smoothed_action = (
                (1.0 - self.action_smoothing_alpha) * self.smoothed_action
                + self.action_smoothing_alpha * action
        ).astype(np.float32)

        self.smoothed_action[3] = (
                0.8 * self.smoothed_action[3]
                + 0.2 * action[3]
        )
        self.last_smoothed_action = self.smoothed_action.copy()

        total_reward = 0.0
        terminated = False
        truncated = False
        term_reason = None

        obs = None
        reward_info = {}
        final_distance_raw = 0.0
        local_distance_raw = 0.0
        repeats_executed = 0

        for repeat_idx in range(self.action_repeat):
            self.current_step += 1
            repeats_executed += 1
            self._invalidate_state_cache()

            if self.enable_nav_visual and repeat_idx == 0:
                self._update_nav_visual_features()
                self._visual_hazard_local_replan_if_needed()

            action_to_apply = self.smoothed_action.copy()

            if self.front_avoid_enabled:
                front_dist = float(self.min_front_dist / max(self.max_sensor_distance, 1e-6))
                left_dist = float(self.min_left_dist / max(self.max_sensor_distance, 1e-6))
                right_dist = float(self.min_right_dist / max(self.max_sensor_distance, 1e-6))

                if front_dist < self.front_avoid_soft:
                    yaw_now = float(self.last_current_yaw)
                    forward_dir = np.array(
                        [np.cos(yaw_now), np.sin(yaw_now)],
                        dtype=np.float32,
                    )
                    lateral_dir = np.array(
                        [-np.sin(yaw_now), np.cos(yaw_now)],
                        dtype=np.float32,
                    )

                    vx = float(action_to_apply[0]) * self.max_velocity
                    vy = float(action_to_apply[1]) * self.max_velocity
                    v_xy = np.array([vx, vy], dtype=np.float32)

                    forward_speed = float(np.dot(v_xy, forward_dir))

                    if forward_speed > 0.0:
                        k = float(np.clip(
                            (front_dist - self.front_avoid_hard) / max(self.front_avoid_soft - self.front_avoid_hard,
                                                                       1e-6),
                            0.0,
                            1.0,
                        ))
                        allowed_forward = forward_speed * k
                        v_xy = v_xy - (forward_speed - allowed_forward) * forward_dir

                        if left_dist >= right_dist:
                            lateral_sign = 1.0
                        else:
                            lateral_sign = -1.0

                        if front_dist < self.front_avoid_hard:
                            v_xy += lateral_sign * lateral_dir * (0.25 * self.max_velocity)

                        action_to_apply[0] = float(np.clip(v_xy[0] / max(self.max_velocity, 1e-6), -1.0, 1.0))
                        action_to_apply[1] = float(np.clip(v_xy[1] / max(self.max_velocity, 1e-6), -1.0, 1.0))

                        self.front_avoid_counter = max(self.front_avoid_counter, 2)

            self._apply_action(action_to_apply)

            if self.front_avoid_counter > 0:
                self.front_avoid_counter -= 1
                self.last_applied_vx_cmd *= self.front_avoid_speed_scale
                self.last_applied_vy_cmd *= self.front_avoid_speed_scale

            self._invalidate_state_cache()
            waypoint_reached = self._update_waypoint_progress()

            if self.current_step % max(1, self.planner_replan_interval) == 0:
                self._dbg_planner_replan_step_this_step = True
                self._rebuild_local_path(force=False)

            near_final_goal = (
                    self.current_waypoint_idx >= len(self.mission_route) - 1
                    and self._get_goal_distance_raw() < max(self.goal_tolerance * 2.0, 2.0)
            )

            if not self.disable_soft_replan_for_testing and not near_final_goal:
                self._soft_local_replan_if_blocked()
            else:
                self.failed_progress_steps = 0
                self.shift_mode_steps_left = 0
                self.shift_vector[:] = 0.0

            obs = self._get_obs()
            reward, reward_info = self._compute_reward(
                obs,
                self.smoothed_action,
                waypoint_reached=waypoint_reached,
            )


            self._update_inactivity_trackers(obs, self.smoothed_action, reward_info)
            self.episode_reward += reward
            total_reward += reward

            self._accumulate_step_stats(reward_info, obs)

            terminated, term_reason = self._check_terminated(obs)
            truncated = self.current_step >= self.max_episode_steps

            # ===== update applied-cmd history НА КАЖДОМ internal substep =====
            current_applied_cmd = self._get_applied_command_vector_norm()
            self.prev_prev_applied_cmd = self.prev_applied_cmd.copy()
            self.prev_applied_cmd = current_applied_cmd.copy()

            if terminated or truncated:
                break

        final_distance_raw = self._get_goal_distance_raw()
        final_distance_norm = float(
            np.clip(
                final_distance_raw / max(
                    np.sqrt(self.world_xy_limit ** 2 + self.world_xy_limit ** 2 + self.max_altitude ** 2),
                    1e-6,
                ),
                0.0,
                1.0,
            )
        )

        local_distance_raw = self._get_local_distance_raw()

        info = {
            "goal_position": self.goal_position.copy(),
            "step": self.current_step,
            "episode_reward": float(self.episode_reward),
            "reward_info": reward_info,
            "planner_info": self._get_planner_info(),
            "terminated_reason": term_reason,
            "local_distance": float(local_distance_raw),
            "final_distance": float(final_distance_raw),
            "final_distance_raw": float(final_distance_raw),
            "action_repeat": int(self.action_repeat),
            "repeats_executed": int(repeats_executed),
        }
        timeout_penalty = 0.0
        if truncated and not terminated:
            timeout_penalty = -20.0
            total_reward += timeout_penalty
            self.episode_reward += timeout_penalty

            reward_info["timeout_penalty"] = float(timeout_penalty)
            info["reward_info"]["timeout_penalty"] = float(timeout_penalty)

            self.episode_stats.setdefault("timeout_penalty_sum", 0.0)
            self.episode_stats.setdefault("timeout_penalty_abs_sum", 0.0)
            self.episode_stats["timeout_penalty_sum"] += float(timeout_penalty)
            self.episode_stats["timeout_penalty_abs_sum"] += abs(float(timeout_penalty))
        else:
            reward_info["timeout_penalty"] = 0.0
            info["reward_info"]["timeout_penalty"] = 0.0

        info["control_debug"] = {
            "raw_action_0": float(self.last_raw_action[0]),
            "raw_action_1": float(self.last_raw_action[1]),
            "raw_action_2": float(self.last_raw_action[2]),
            "raw_action_3": float(self.last_raw_action[3]),
            "approach_turn_zone_dbg": float(getattr(self, "_debug_approach_turn_zone", 0.0)),
            "inspect_turn_zone_dbg": float(getattr(self, "_debug_inspect_turn_zone", 0.0)),
            "inspect_lock_zone_dbg": float(getattr(self, "_debug_inspect_lock_zone", 0.0)),
            "smoothed_action_0": float(self.last_smoothed_action[0]),
            "smoothed_action_1": float(self.last_smoothed_action[1]),
            "smoothed_action_2": float(self.last_smoothed_action[2]),
            "smoothed_action_3": float(self.last_smoothed_action[3]),
            "applied_vx_cmd": float(self.last_applied_vx_cmd),
            "applied_vy_cmd": float(self.last_applied_vy_cmd),
            "applied_vz_cmd": float(self.last_applied_vz_cmd),
            "applied_yaw_rate_cmd": float(self.last_applied_yaw_rate_cmd),
            "current_z": float(self.last_current_z),
            "current_vz": float(self.last_current_vz),
            "target_z": float(self.last_target_z),
            "z_error": float(self.last_z_error),
            "current_roll": float(self.last_current_roll),
            "current_pitch": float(self.last_current_pitch),
            "current_yaw": float(self.last_current_yaw),
            "down_distance": float(getattr(self, "last_down_distance", self.max_sensor_distance)),
            "landed_state": float(getattr(self, "last_landed_state", 0.0)),
            "tilt_abs_rad": float(getattr(self, "last_tilt_abs", 0.0)),
            "ground_speed": float(getattr(self, "last_ground_speed", 0.0)),
            "ground_contact_steps": float(getattr(self, "ground_contact_steps", 0)),
            "command_duration": float(getattr(self, "last_command_duration", 0.0)),
            "depth_read_time": float(getattr(self, "last_depth_read_time", 0.0)),
            "depth_used_cache": float(getattr(self, "last_depth_used_cache", 0.0)),
        }

        if terminated or truncated:
            self.episode_index += 1
            mean_heading_error_cos = self.heading_cos_sum / max(1, self.current_step)
            mean_speed = self.speed_sum / max(1, self.current_step)
            info["episode_summary"] = self._build_episode_summary(
                final_distance_norm=final_distance_norm,
                final_distance_raw=final_distance_raw,
                term_reason=term_reason,
                truncated=truncated,
                mean_heading_error_cos=mean_heading_error_cos,
                mean_speed=mean_speed,
                obs=obs,
            )

        # prev_action оставляем по внешнему policy-step
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = self.smoothed_action.copy()

        return obs, float(total_reward), terminated, truncated, info

    def close(self):
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        try:
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
        except Exception:
            pass

    # =========================================================
    # Reset helpers
    # =========================================================

    def _resolve_fixed_goal(self) -> np.ndarray:
        if self.route_points is not None and len(self.route_points) > 0:
            return self.route_points[-1].copy()
        if self.fixed_goal is not None:
            return self.fixed_goal.copy()
        return np.array([0.0, 0.0, -3.0], dtype=np.float32)

    def _hard_reset_and_place_drone(self):
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.6)
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(z=-3.0, velocity=1.5, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def _move_drone_to_position(self, target_position: np.ndarray):
        """
        Безопасный reset для inspect:
        1) после client.reset() переводим SimpleFlight в flying-state через takeoffAsync;
        2) телепортируем в стартовую точку;
        3) фиксируем высоту короткой velocityZ-командой;
        4) оставляем API control + armed.
        """
        if target_position is None:
            return

        point = np.asarray(target_position, dtype=np.float32).reshape(3).copy()

        reset_lift_m = float(getattr(self, "inspect_reset_lift_m", 0.0))
        point[2] -= reset_lift_m  # AirSim NED: меньше z => выше

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            time.sleep(0.15)

            # КРИТИЧНО: после reset/teleport SimpleFlight должен перейти в flying state.
            try:
                self.client.takeoffAsync(timeout_sec=3.0, vehicle_name=self.vehicle_name).join()
            except TypeError:
                self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

            time.sleep(0.15)

        except Exception as exc:
            print("[INSPECT RESET] takeoff/arm before teleport failed:", exc)

        pose = airsim.Pose(
            airsim.Vector3r(float(point[0]), float(point[1]), float(point[2])),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )

        for _ in range(3):
            self.client.simSetVehiclePose(
                pose,
                ignore_collision=True,
                vehicle_name=self.vehicle_name,
            )
            time.sleep(0.10)

        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            time.sleep(0.10)

            # Удерживаем именно текущий NED-z после телепорта.
            self.client.moveByVelocityZAsync(
                vx=0.0,
                vy=0.0,
                z=float(point[2]),
                duration=0.80,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.vehicle_name,
            ).join()

            self.client.armDisarm(True, self.vehicle_name)
            self.client.enableApiControl(True, self.vehicle_name)

        except Exception as exc:
            print("[INSPECT RESET] hold after teleport failed:", exc)

        self.start_position = point.copy()
        self._invalidate_state_cache()

        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            print(
                "[INSPECT RESET STATE]",
                "pos=",
                [
                    state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val,
                ],
                "landed_state=",
                int(state.landed_state),
            )
        except Exception:
            pass

        if self.debug_print_reset:
            print("[INSPECT RESET TELEPORT] start_position:", self.start_position)

    def _sample_start_position(self) -> np.ndarray:
        if self.start_points is not None and len(self.start_points) > 0:
            idx = int(self.np_random.integers(0, len(self.start_points)))
            point = np.asarray(self.start_points[idx], dtype=np.float32)
            if point.ndim != 1 or point.shape[0] != 3:
                raise ValueError(
                    f"Sampled start point must be shape (3,), got shape {point.shape}. "
                    f"Check START_POINTS format."
                )
            return point.copy()

        if self.start_position is not None:
            return self.start_position.copy()

        return np.array([0.0, 0.0, -3.0], dtype=np.float32)

    def _sample_goal_position(self) -> np.ndarray:
        if self.goal_points is not None and len(self.goal_points) > 0:
            idx = int(self.np_random.integers(0, len(self.goal_points)))
            return self.goal_points[idx].copy()

        if self.route_points is not None and len(self.route_points) > 0:
            return self.route_points[-1].copy()

        if self.fixed_goal is not None:
            return self.fixed_goal.copy()

        x = self.np_random.uniform(-8.0, 8.0)
        y = self.np_random.uniform(-8.0, 8.0)
        z = self.np_random.uniform(-4.0, -2.5)
        return np.array([x, y, z], dtype=np.float32)

    # =========================================================
    # Route planner
    # =========================================================

    def _build_mission_route(self):
        if self.current_route_points is not None and len(self.current_route_points) > 0:
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

        anchor = self.start_position.copy() if self.start_position is not None else np.array([0.0, 0.0, -3.0], dtype=np.float32)
        for p in self.mission_route:
            self.route_total_length += float(np.linalg.norm(p - anchor))
            anchor = p.copy()
        self.route_total_length = max(self.route_total_length, 1e-6)

    def _current_position_vec(self) -> np.ndarray:
        if self._cached_position_vec is None:
            state = self._get_multirotor_state_cached()
            pos = state.kinematics_estimated.position
            self._cached_position_vec = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        return self._cached_position_vec.copy()

    def _get_local_distance_raw(self) -> float:
        if self.active_local_waypoint is None:
            return self._get_goal_distance_raw()
        pos = self._current_position_vec()
        return float(np.linalg.norm(self.active_local_waypoint - pos))

    def _point_to_grid(self, point: np.ndarray, center: np.ndarray):
        rel = (point - center) / max(self.planner_grid_resolution, 1e-6)
        return np.round(rel).astype(int)

    def _grid_to_point(self, grid: np.ndarray, center: np.ndarray):
        return center + grid.astype(np.float32) * self.planner_grid_resolution

    def _make_local_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        # Lightweight local 3D planner: interpolate in XY and Z with small lookahead horizon.
        delta = goal - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return [goal.copy()]
        steps = max(1, int(np.ceil(dist / max(self.planner_grid_resolution, 0.5))))
        path = []
        for i in range(1, steps + 1):
            alpha = i / steps
            path.append((start + alpha * delta).astype(np.float32))
        return path

    def _rebuild_local_path(self, force: bool = False):
        old_active_local_waypoint = None
        if self.active_local_waypoint is not None:
            old_active_local_waypoint = np.asarray(self.active_local_waypoint, dtype=np.float32).copy()

        def _mark_rebuild_debug(new_active_local_waypoint):
            self._dbg_local_path_rebuilt_this_step = True

            if old_active_local_waypoint is None and new_active_local_waypoint is None:
                return
            if old_active_local_waypoint is None or new_active_local_waypoint is None:
                self._dbg_local_target_changed_this_step = True
                return

            try:
                old_target = np.asarray(old_active_local_waypoint, dtype=np.float32).reshape(-1)
                new_target = np.asarray(new_active_local_waypoint, dtype=np.float32).reshape(-1)
                if old_target.shape != new_target.shape:
                    self._dbg_local_target_changed_this_step = True
                    return
                if float(np.linalg.norm(new_target - old_target)) > 1e-6:
                    self._dbg_local_target_changed_this_step = True
            except Exception:
                self._dbg_local_target_changed_this_step = True

        if not self.use_route_planner or not self.mission_route:
            self.active_local_waypoint = self.goal_position.copy()
            self.local_path_world = [self.active_local_waypoint.copy()]
            self.local_path_index = 0
            _mark_rebuild_debug(self.active_local_waypoint)
            return

        if self.current_waypoint_idx >= len(self.mission_route):
            self.current_waypoint_idx = len(self.mission_route) - 1

        pos = self._current_position_vec()
        target = self.mission_route[self.current_waypoint_idx].copy()

        if self.shift_mode_steps_left > 0:
            target = (target + self.shift_vector).astype(np.float32)

        path = self._make_local_path(pos, target)
        self.local_path_world = path

        is_final_wp = (self.current_waypoint_idx >= len(self.mission_route) - 1)

        if is_final_wp:
            self.local_path_index = max(len(path) - 1, 0)
            self.active_local_waypoint = target.copy()  # ВАЖНО: прямо в финальный goal
        else:
            self.local_path_index = min(self.planner_lookahead_steps, max(len(path) - 1, 0))
            self.active_local_waypoint = self.local_path_world[self.local_path_index].copy()
        _mark_rebuild_debug(self.active_local_waypoint)

    def _update_waypoint_progress(self) -> bool:
        if not self.mission_route:
            return False

        pos = self._current_position_vec()
        current_wp = self.mission_route[self.current_waypoint_idx]
        dist = float(np.linalg.norm(current_wp - pos))

        if self.current_waypoint_idx == len(self.mission_route) - 1:
            self.active_local_waypoint = current_wp.copy()
            self.local_path_world = [current_wp.copy()]
            self.local_path_index = 0
            return False

        if dist <= self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            self.shift_mode_steps_left = 0
            self.shift_vector[:] = 0.0
            self._rebuild_local_path(force=True)
            return True

        return False

        # достигли последнего waypoint: отдельную награду не даём,
        # здесь сработает goal_reward в _compute_reward()
        if dist <= self.waypoint_tolerance and self.current_waypoint_idx == len(self.mission_route) - 1:
            self.active_local_waypoint = current_wp.copy()
            return False

        return False

    def _soft_local_replan_if_blocked(self):
        local_distance = self._get_local_distance_raw()
        if self.prev_local_distance is None:
            self.prev_local_distance = local_distance
            return

        progress_delta = self.prev_local_distance - local_distance
        self.last_progress_delta = float(progress_delta)

        obs = self._get_depth_and_risk_features()[0]
        depth_front = float(obs[1])
        risk_front = float(1.0 - depth_front)

        if progress_delta < 0.002 and risk_front > 0.60:
            self.failed_progress_steps += 1
        else:
            self.failed_progress_steps = max(0, self.failed_progress_steps - 1)

        if self.failed_progress_steps >= self.soft_avoidance_fail_steps and self.shift_mode_steps_left <= 0:
            sign = -1.0 if float(obs[0]) > float(obs[2]) else 1.0
            self.shift_vector = np.array([0.0, sign * self.planner_shift_distance, 0.0], dtype=np.float32)
            self.shift_mode_steps_left = self.planner_shift_hold_steps
            self.failed_progress_steps = 0
            self._rebuild_local_path(force=True)

        if self.shift_mode_steps_left > 0:
            self.shift_mode_steps_left -= 1
            if self.shift_mode_steps_left == 0:
                self.shift_vector[:] = 0.0
                self._rebuild_local_path(force=True)

    def _update_infra_context_features(self, image_bgr: Optional[np.ndarray]):
        self.infra_features[:] = 0.0

        if not self.enable_nav_seg_context:
            return
        if image_bgr is None:
            return
        if self._infra_model is None:
            return

        infra_dets = self._run_nav_detector(
            image_bgr=image_bgr,
            model=self._infra_model,
            target_class_names=self.infra_target_class_names,
            conf_threshold=self.infra_conf_threshold,
            iou_threshold=self.nav_visual_iou_threshold,
        )

        selected = None
        if infra_dets:
            selected = self._select_nav_detection(infra_dets)

        if selected is not None:
            self.infra_features[:] = self._detection_to_nav_features(selected)

    def _update_inactivity_trackers(self, obs: np.ndarray, action: np.ndarray, reward_info: Dict[str, float]):
        speed_norm = float(np.linalg.norm([obs[3], obs[4], obs[5]]))
        action_norm = float(np.linalg.norm(action))
        progress_delta = float(reward_info.get("progress_delta_raw", self.last_progress_delta))

        self.last_speed_norm = speed_norm
        self.last_action_magnitude = action_norm
        self.last_progress_delta = progress_delta

        if speed_norm < self.inactive_speed_threshold and action_norm < self.inactive_action_threshold:
            self.inactive_steps += 1
        else:
            self.inactive_steps = 0

        if progress_delta < self.no_progress_threshold:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

    def _get_route_progress_norm(self, pos: np.ndarray) -> float:
        if not self.mission_route:
            return 0.0
        passed = 0.0
        anchor = self.start_position.copy()
        for i, wp in enumerate(self.mission_route):
            segment = float(np.linalg.norm(wp - anchor))
            if i < self.current_waypoint_idx:
                passed += segment
            elif i == self.current_waypoint_idx:
                segment_done = float(np.linalg.norm(pos - anchor))
                passed += np.clip(segment_done, 0.0, segment)
                break
            anchor = wp.copy()
        return float(np.clip(passed / self.route_total_length, 0.0, 1.0))

    def _get_route_deviation_norm(self, pos: np.ndarray) -> float:
        if self.active_local_waypoint is not None:
            target = self.active_local_waypoint
        elif self.mission_route:
            idx = min(self.current_waypoint_idx, len(self.mission_route) - 1)
            target = self.mission_route[idx]
        else:
            return 0.0

        dev_xy = float(np.linalg.norm((pos - target)[:2]))
        return float(np.clip(dev_xy / max(self.world_xy_limit, 1e-6), 0.0, 1.0))

    def _get_planner_hint(self, pos: np.ndarray) -> np.ndarray:
        if self.active_local_waypoint is None:
            return np.zeros(3, dtype=np.float32)
        delta = self.active_local_waypoint - pos
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return (delta / norm).astype(np.float32)

    def _get_planner_info(self) -> Dict[str, Any]:
        return {
            "waypoint_idx": int(self.current_waypoint_idx),
            "local_path_len": int(len(self.local_path_world)),
            "local_path_index": int(self.local_path_index),
            "shift_mode_steps_left": int(self.shift_mode_steps_left),
            "failed_progress_steps": int(self.failed_progress_steps),
            "last_progress_delta": float(self.last_progress_delta),
            "inactive_steps": int(self.inactive_steps),
            "no_progress_steps": int(self.no_progress_steps),
            "last_speed_norm": float(self.last_speed_norm),
            "last_action_magnitude": float(self.last_action_magnitude),
            "visual_debug": self._get_visual_debug_info()
        }

    # =========================================================
    # Observation
    # =========================================================

    def _get_depth_from_image(self):
        if not self.use_depth_image:
            return None
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.depth_camera_name,
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True,
                    compress=False,
                )
            ], vehicle_name=self.vehicle_name)
            if not responses:
                return None
            resp = responses[0]
            if resp.width <= 0 or resp.height <= 0 or len(resp.image_data_float) == 0:
                return None
            depth = np.array(resp.image_data_float, dtype=np.float32).reshape(resp.height, resp.width)
            depth = np.nan_to_num(depth, nan=self.depth_max_distance, posinf=self.depth_max_distance, neginf=self.depth_max_distance)
            depth = np.clip(depth, 0.0, self.depth_max_distance)
            return depth
        except Exception:
            return None

    def _get_distance_sensor(self, sensor_name: str) -> float:
        try:
            data = self.client.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name=self.vehicle_name)
            dist = float(data.distance)
            if np.isnan(dist) or dist <= 0.0:
                return self.max_sensor_distance
            return min(dist, self.max_sensor_distance)
        except Exception:
            return self.max_sensor_distance

    def _get_depth_and_risk_features(self):
        # DepthPerspective через simGetImages тяжёлый.
        # Читаем depth не каждый env-step, а раз в depth_image_interval шагов.
        use_cache = (
                self.use_depth_image
                and self.depth_image_interval > 1
                and self.current_step > 0
                and (self.current_step % self.depth_image_interval) != 0
                and hasattr(self, "_last_depth_features")
        )

        if use_cache:
            self.last_depth_used_cache = 1.0
            return self._last_depth_features.copy(), self._last_depth_dists.copy()

        t0 = time.perf_counter()
        depth = self._get_depth_from_image()
        self.last_depth_read_time = float(time.perf_counter() - t0)
        self.last_depth_used_cache = 0.0

        if depth is not None:
            h, w = depth.shape
            left = np.min(depth[:, : max(1, w // 3)])
            center = np.min(depth[:, max(0, w // 3): max(1, 2 * w // 3)])
            right = np.min(depth[:, max(0, 2 * w // 3):])
            up = np.min(depth[: max(1, h // 3), :])
            down = np.min(depth[max(0, 2 * h // 3):, :])
            dists = np.array([left, center, right, up, down], dtype=np.float32)
            dists = np.clip(dists / max(self.depth_max_distance, 1e-6), 0.0, 1.0)
        else:
            front = self._get_distance_sensor("FrontDistance")
            left = self._get_distance_sensor("LeftDistance")
            right = self._get_distance_sensor("RightDistance")
            up = self._get_distance_sensor("UpDistance")
            down = self._get_distance_sensor("DownDistance")
            dists = np.array([left, front, right, up, down], dtype=np.float32)
            dists = np.clip(dists / max(self.max_sensor_distance, 1e-6), 0.0, 1.0)

        depth_left, depth_center, depth_right, depth_up, depth_down = [float(x) for x in dists]
        depth_min = float(np.min(dists))
        risk = 1.0 - dists

        features = np.array([
            depth_left, depth_center, depth_right, depth_up, depth_down, depth_min,
            float(risk[0]), float(risk[1]), float(risk[2]), float(risk[3]), float(risk[4]),
        ], dtype=np.float32)

        self._last_depth_features = features.copy()
        self._last_depth_dists = dists.copy()

        return features, dists

    def _get_obs(self) -> np.ndarray:
        state = self._get_multirotor_state_cached()
        kin = state.kinematics_estimated
        pos = np.array([kin.position.x_val, kin.position.y_val, kin.position.z_val], dtype=np.float32)

        pos_x_norm = np.clip(pos[0] / self.world_xy_limit, -1.0, 1.0)
        pos_y_norm = np.clip(pos[1] / self.world_xy_limit, -1.0, 1.0)
        pos_z_norm = np.clip(pos[2] / self.max_altitude, -1.0, 1.0)

        vx = np.clip(kin.linear_velocity.x_val / self.max_velocity, -1.0, 1.0)
        vy = np.clip(kin.linear_velocity.y_val / self.max_velocity, -1.0, 1.0)
        vz = np.clip(kin.linear_velocity.z_val / self.max_velocity, -1.0, 1.0)

        _, _, yaw = airsim.to_eularian_angles(kin.orientation)
        yaw_sin = np.float32(np.sin(yaw))
        yaw_cos = np.float32(np.cos(yaw))
        yaw_rate = kin.angular_velocity.z_val / np.deg2rad(self.max_yaw_rate)
        yaw_rate = np.clip(yaw_rate, -1.0, 1.0)

        altitude = -pos[2]
        altitude_norm = np.clip(altitude / self.max_altitude, 0.0, 1.0)

        local_goal = self.active_local_waypoint.copy() if self.active_local_waypoint is not None else self.goal_position.copy()
        final_goal = self.goal_position.copy()

        local_delta_raw = local_goal - pos
        final_delta_raw = final_goal - pos

        local_dx = np.clip(local_delta_raw[0] / self.world_xy_limit, -1.0, 1.0)
        local_dy = np.clip(local_delta_raw[1] / self.world_xy_limit, -1.0, 1.0)
        local_dz = np.clip(local_delta_raw[2] / self.max_altitude, -1.0, 1.0)

        final_dx = np.clip(final_delta_raw[0] / self.world_xy_limit, -1.0, 1.0)
        final_dy = np.clip(final_delta_raw[1] / self.world_xy_limit, -1.0, 1.0)
        final_dz = np.clip(final_delta_raw[2] / self.max_altitude, -1.0, 1.0)

        max_goal_dist = np.sqrt(self.world_xy_limit ** 2 + self.world_xy_limit ** 2 + self.max_altitude ** 2)
        local_distance_norm = np.clip(np.linalg.norm(local_delta_raw) / max_goal_dist, 0.0, 1.0)
        final_distance_norm = np.clip(np.linalg.norm(final_delta_raw) / max_goal_dist, 0.0, 1.0)

        desired_heading = np.arctan2(local_delta_raw[1], local_delta_raw[0])
        heading_error = self._wrap_angle(desired_heading - yaw)
        heading_local_sin = np.float32(np.sin(heading_error))
        heading_local_cos = np.float32(np.cos(heading_error))

        route_progress_norm = self._get_route_progress_norm(pos)
        route_deviation_norm = self._get_route_deviation_norm(pos)
        planner_hint = self._get_planner_hint(pos)

        depth_and_risk, _ = self._get_depth_and_risk_features()
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        collision_flag = 1.0 if collision_info.has_collided else 0.0

        obs = np.array([
            pos_x_norm, pos_y_norm, pos_z_norm,
            vx, vy, vz,
            yaw_sin, yaw_cos, yaw_rate,
            altitude_norm,
            local_dx, local_dy, local_dz,
            final_dx, final_dy, final_dz,
            local_distance_norm, final_distance_norm,
            heading_local_sin, heading_local_cos,
            route_progress_norm, route_deviation_norm,
            float(planner_hint[0]), float(planner_hint[1]), float(planner_hint[2]),
            float(depth_and_risk[0]), float(depth_and_risk[1]), float(depth_and_risk[2]), float(depth_and_risk[3]), float(depth_and_risk[4]), float(depth_and_risk[5]),
            float(depth_and_risk[6]), float(depth_and_risk[7]), float(depth_and_risk[8]), float(depth_and_risk[9]), float(depth_and_risk[10]),
            collision_flag,
            self.prev_action[0], self.prev_action[1], self.prev_action[2], self.prev_action[3],
        ], dtype=np.float32)

        if self.yolo_feature_dim > 0:
            obs = np.concatenate([obs, self.yolo_features.astype(np.float32)], axis=0)

        return obs.astype(np.float32)

    # =========================================================
    # Action
    # =========================================================

    def _ema_rate_limit(self, prev_value: float, target_value: float, alpha: float, max_delta: float) -> float:
        # лёгкий EMA без большого лага
        ema_target = (1.0 - alpha) * float(prev_value) + alpha * float(target_value)

        # ограничение на скачок между соседними шагами
        delta = ema_target - float(prev_value)
        delta = float(np.clip(delta, -max_delta, max_delta))

        return float(prev_value + delta)

    def _apply_action(self, action: np.ndarray):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
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

        xy_speed_limit = min(self.max_velocity, 3.0)
        vx_cmd_raw = float(ax * xy_speed_limit)
        vy_cmd_raw = float(ay * xy_speed_limit)

        manual_yaw_limit_deg = 4.0
        auto_yaw_k = 8.0
        auto_yaw_limit_deg = 6.0

        manual_yaw_rate = float(action[3]) * manual_yaw_limit_deg
        auto_yaw_rate = 0.0
        desired_yaw_deg = float(np.rad2deg(yaw))

        if self.active_local_waypoint is not None:
            current_x = float(pos.x_val)
            current_y = float(pos.y_val)
            target_x = float(self.active_local_waypoint[0])
            target_y = float(self.active_local_waypoint[1])

            dx = target_x - current_x
            dy = target_y - current_y
            xy_dist = float(np.hypot(dx, dy))

            if xy_dist > self.nav_fixed_yaw_deadband_xy_m:
                desired_yaw_deg = float(np.rad2deg(np.arctan2(dy, dx)))
                yaw_error = self._wrap_angle(np.deg2rad(desired_yaw_deg) - float(yaw))
                if abs(yaw_error) < np.deg2rad(3.0):
                    yaw_error = 0.0
                auto_yaw_rate = float(np.clip(np.rad2deg(yaw_error) * auto_yaw_k,
                                              -auto_yaw_limit_deg, auto_yaw_limit_deg))

        yaw_rate_cmd_raw = 0.80 * auto_yaw_rate + 0.20 * manual_yaw_rate
        yaw_rate_cmd_raw = float(np.clip(yaw_rate_cmd_raw, -6.0, 6.0))

        if self.altitude_pd_enabled and self.active_local_waypoint is not None:
            z_offset = float(action[2]) * self.altitude_action_z_offset
            target_z = float(self.active_local_waypoint[2]) + z_offset

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

        xy_delta_limit = min(self.max_delta_vxy_cmd, 0.30)
        yaw_delta_limit = min(self.max_delta_yaw_rate_cmd, 2.0)

        if self.current_waypoint_idx >= len(self.mission_route) - 1:
            final_dist_raw = self._get_goal_distance_raw()
            if final_dist_raw < max(self.goal_tolerance * 2.0, 2.0):
                xy_delta_limit *= 0.75
                yaw_delta_limit *= 0.75

        vx_cmd = self._ema_rate_limit(
            self.filtered_vx_cmd, vx_cmd_raw, self.cmd_filter_alpha_xy, xy_delta_limit
        )
        vy_cmd = self._ema_rate_limit(
            self.filtered_vy_cmd, vy_cmd_raw, self.cmd_filter_alpha_xy, xy_delta_limit
        )
        vz_cmd = self._ema_rate_limit(
            self.filtered_vz_cmd, vz_cmd_raw, self.cmd_filter_alpha_z, self.max_delta_vz_cmd
        )
        yaw_rate_cmd = self._ema_rate_limit(
            self.filtered_yaw_rate_cmd, yaw_rate_cmd_raw, self.cmd_filter_alpha_yaw, yaw_delta_limit
        )

        self.filtered_vx_cmd = float(vx_cmd)
        self.filtered_vy_cmd = float(vy_cmd)
        self.filtered_vz_cmd = float(vz_cmd)
        self.filtered_yaw_rate_cmd = float(yaw_rate_cmd)

        self.last_applied_vx_cmd = float(vx_cmd)
        self.last_applied_vy_cmd = float(vy_cmd)
        self.last_applied_vz_cmd = float(vz_cmd)

        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate_cmd))
        self.last_applied_yaw_rate_cmd = float(yaw_rate_cmd)

        if self.nav_fixed_yaw_enabled:
            if abs(vx_cmd) + abs(vy_cmd) > 0.10:
                yaw_step_deg = float(np.clip(
                    self._wrap_angle_deg(desired_yaw_deg - self.filtered_yaw_target_deg),
                    -self.nav_fixed_yaw_max_step_deg,
                    self.nav_fixed_yaw_max_step_deg,
                ))
                self.filtered_yaw_target_deg = self._wrap_angle_deg(
                    self.filtered_yaw_target_deg + yaw_step_deg
                )

            yaw_mode = airsim.YawMode(
                is_rate=False,
                yaw_or_rate=float(self.filtered_yaw_target_deg),
            )
            self.last_applied_yaw_rate_cmd = 0.0

        command_duration = self.control_command_duration
        if command_duration is None:
            command_duration = max(float(self.step_duration) * 3.0, 0.25)
        else:
            command_duration = max(float(command_duration), float(self.step_duration) * 1.5)

        self.last_command_duration = float(command_duration)

        if self.nav_use_body_frame_control:
            cy = float(np.cos(yaw))
            sy = float(np.sin(yaw))

            vx_body = cy * float(vx_cmd) + sy * float(vy_cmd)
            vy_body = -sy * float(vx_cmd) + cy * float(vy_cmd)

            self._last_move_task = self.client.moveByVelocityBodyFrameAsync(
                vx=float(vx_body),
                vy=float(vy_body),
                vz=float(vz_cmd),
                duration=float(command_duration),
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=self.vehicle_name,
            )
        else:
            self._last_move_task = self.client.moveByVelocityAsync(
                vx=float(vx_cmd),
                vy=float(vy_cmd),
                vz=float(vz_cmd),
                duration=float(command_duration),
                yaw_mode=yaw_mode,
                vehicle_name=self.vehicle_name,
            )

        time.sleep(float(self.step_duration))
    # =========================================================
    # Reward
    # =========================================================
    def _get_remaining_route_distance(self) -> float:
        """
        Оставшаяся длина маршрута от текущей позиции до конца mission_route.
        Нужна для episode_summary.
        """
        if not self.mission_route:
            return float(self._get_goal_distance_raw())

        if self.current_waypoint_idx >= len(self.mission_route):
            return 0.0

        pos = self._current_position_vec()
        idx = int(np.clip(self.current_waypoint_idx, 0, len(self.mission_route) - 1))

        remaining = 0.0
        anchor = pos.astype(np.float32)

        for p in self.mission_route[idx:]:
            point = np.asarray(p, dtype=np.float32)
            remaining += float(np.linalg.norm(point - anchor))
            anchor = point

        return float(max(0.0, remaining))
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray, waypoint_reached: bool = False):
        reward = 0.0
        info: Dict[str, float] = {}

        nav = self._decode_base_obs(obs)

        local_distance = nav["local_distance_norm"]
        final_distance = nav["final_distance_norm"]
        heading_local_cos = nav["heading_local_cos"]

        depth_left = nav["depth_left"]
        depth_center = nav["depth_center"]
        depth_right = nav["depth_right"]
        depth_up = nav["depth_up"]
        collision_flag = nav["collision"]
        altitude_norm = nav["altitude_norm"]
        route_deviation_norm = nav["route_deviation_norm"]

        planner_hint = np.array(
            [nav["planner_hint_x"], nav["planner_hint_y"], nav["planner_hint_z"]],
            dtype=np.float32,
        )
        velocity_vec = np.array([nav["vx"], nav["vy"], nav["vz"]], dtype=np.float32)
        local_dz = nav["local_dz"]

        min_obstacle_dist = min(depth_center, depth_left, depth_right, depth_up)
        total_risk = float(np.mean([
            nav["risk_front"],
            nav["risk_left"],
            nav["risk_right"],
            nav["risk_up"],
            nav["risk_down"],
        ]))
        speed_norm = float(np.linalg.norm(velocity_vec))
        action_norm = float(np.linalg.norm(action))

        local_distance_raw = self._get_local_distance_raw()
        self._dbg_prev_local_distance_before_reward = (
            float(self.prev_local_distance) if self.prev_local_distance is not None else 0.0
        )
        self._dbg_local_distance_raw_before_reward = float(local_distance_raw)
        progress_delta_raw_before_guard = 0.0
        if self.prev_local_distance is not None:
            progress_delta_raw_before_guard = float(self.prev_local_distance - local_distance_raw)
        self._dbg_progress_delta_raw_before_reward = float(progress_delta_raw_before_guard)

        if abs(local_dz) < 0.05:
            local_z_error_penalty = 0.0
        else:
            local_z_error_penalty = -0.50 * abs(local_dz)
        reward += local_z_error_penalty
        info["local_z_error_penalty"] = local_z_error_penalty

        progress_delta_raw = 0.0
        progress_reward = 0.0
        if self.prev_local_distance is not None:
            if waypoint_reached:
                # При смене waypoint расстояние перескакивает с дистанции до старой точки
                # на дистанцию до новой точки. Это не должно считаться отрицательным progress.
                progress_delta_raw = 0.0
                progress_reward = 0.0
            else:
                progress_delta_raw = float(self.prev_local_distance - local_distance_raw)
                progress_delta_raw = float(np.clip(progress_delta_raw, -2.0, 2.0))
                progress_reward = progress_delta_raw * 15.0

            reward += progress_reward

        self._dbg_progress_guard_active_this_step = bool(
            self._dbg_progress_guard_active_this_step
            or (self._dbg_local_target_changed_this_step and not waypoint_reached)
        )
        info["progress_delta_raw"] = progress_delta_raw
        info["progress_reward"] = progress_reward

        goal_reward = 0.0
        goal_distance_threshold = self.goal_tolerance / max(
            np.sqrt(self.world_xy_limit ** 2 + self.world_xy_limit ** 2 + self.max_altitude ** 2),
            1e-6,
        )

        if (
                self.current_waypoint_idx >= len(self.mission_route) - 1
                and final_distance < goal_distance_threshold
        ):
            goal_reward = 100.0
            reward += goal_reward

        info["goal_reward"] = goal_reward

        waypoint_reward = 0.0
        if waypoint_reached:
            waypoint_reward = 8.0
            reward += waypoint_reward
        info["waypoint_reward"] = waypoint_reward

        collision_penalty = 0.0
        if collision_flag > 0.5:
            collision_penalty = -60.0
            reward += collision_penalty
        info["collision_penalty"] = collision_penalty

        obstacle_penalty = 0.0
        safe_threshold = 0.35
        if min_obstacle_dist < safe_threshold:
            proximity = 1.0 - (min_obstacle_dist / safe_threshold)
            obstacle_penalty = -0.6 * float(proximity ** 2)
            reward += obstacle_penalty
        info["obstacle_penalty"] = obstacle_penalty

        # ===== CAPS-like temporal smoothness =====
        # ===== CAPS/ASAP по applied command, а не по policy action =====
        current_applied_cmd = self._get_applied_command_vector_norm()

        delta_cmd = current_applied_cmd - self.prev_applied_cmd
        delta_xy = float(np.linalg.norm(delta_cmd[:2]))
        delta_z = float(abs(delta_cmd[2]))
        delta_yaw = float(abs(delta_cmd[3]))

        caps_smooth_penalty = -(
                self.caps_w_xy * (delta_xy ** 2) +
                self.caps_w_z * (delta_z ** 2) +
                self.caps_w_yaw * (delta_yaw ** 2)
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
                self.asap_w_xy * (jerk_xy ** 2) +
                self.asap_w_z * (jerk_z ** 2) +
                self.asap_w_yaw * (jerk_yaw ** 2)
        )
        reward += asap_smooth_penalty
        info["asap_smooth_penalty"] = asap_smooth_penalty
        info["jerk_xy"] = jerk_xy
        info["jerk_z"] = jerk_z
        info["jerk_yaw"] = jerk_yaw
        obstacle_clearance_reward = 0.0
        if self.prev_min_obstacle_dist is not None:
            obstacle_clearance_reward = 1.0 * (min_obstacle_dist - self.prev_min_obstacle_dist)
            obstacle_clearance_reward = float(np.clip(obstacle_clearance_reward, -0.1, 0.1))
            reward += obstacle_clearance_reward
        info["obstacle_clearance_reward"] = obstacle_clearance_reward

        frontal_block_penalty = 0.0
        if depth_center < 0.15:
            frontal_block_penalty = -0.2
            reward += frontal_block_penalty
        info["frontal_block_penalty"] = frontal_block_penalty

        action_penalty = -0.005 * action_norm
        reward += action_penalty
        info["action_penalty"] = action_penalty

        altitude_penalty = 0.0

        # мягкий штраф за опасно малую высоту
        if altitude_norm < 0.20:
            ratio = (0.20 - altitude_norm) / 0.20
            altitude_penalty -= 0.25 * float(np.clip(ratio, 0.0, 1.0))

        # дополнительная зона давления перед аварийным too_low
        if altitude_norm < 0.15:
            ratio = (0.15 - altitude_norm) / 0.15
            altitude_penalty -= 0.20 * float(np.clip(ratio, 0.0, 1.0))

        # сильный штраф совсем у земли
        if altitude_norm < 0.10:
            ratio = (0.10 - altitude_norm) / 0.10
            altitude_penalty -= 0.40 * float(np.clip(ratio, 0.0, 1.0))

        # небольшой штраф за слишком большую высоту
        if altitude_norm > 0.92:
            ratio = (altitude_norm - 0.92) / 0.08
            altitude_penalty -= 0.08 * float(np.clip(ratio, 0.0, 1.0))

        local_z_hold_reward = 0.0
        if abs(local_dz) < 0.08:
            local_z_hold_reward = 0.03 * (1.0 - abs(local_dz) / 0.08)
            reward += local_z_hold_reward
        info["local_z_hold_reward"] = local_z_hold_reward

        reward += altitude_penalty
        info["altitude_penalty"] = altitude_penalty

        forward_facing_reward = 0.02 * max(0.0, heading_local_cos)
        reward += forward_facing_reward
        info["forward_facing_reward"] = forward_facing_reward

        route_deviation_penalty = -0.10 * route_deviation_norm
        reward += route_deviation_penalty
        info["route_deviation_penalty"] = route_deviation_penalty

        planner_alignment_reward = 0.0
        if speed_norm > 1e-6:
            vel_dir = velocity_vec / speed_norm
            planner_alignment_reward = 0.03 * max(0.0, float(np.dot(vel_dir, planner_hint)))
            reward += planner_alignment_reward
        info["planner_alignment_reward"] = planner_alignment_reward

        risk_penalty = -0.08 * total_risk
        reward += risk_penalty
        info["risk_penalty"] = risk_penalty

        inactive_penalty = 0.0
        if speed_norm < self.inactive_speed_threshold and action_norm < self.inactive_action_threshold:
            speed_ratio = speed_norm / max(self.inactive_speed_threshold, 1e-6)
            inactive_penalty = -self.inactive_penalty_weight * float(1.0 - np.clip(speed_ratio, 0.0, 1.0))
            reward += inactive_penalty
        info["inactive_penalty"] = inactive_penalty

        active_progress_reward = 0.0
        if progress_delta_raw > self.no_progress_threshold and speed_norm >= self.inactive_speed_threshold:
            active_progress_reward = self.active_progress_reward_weight
            reward += active_progress_reward
        info["active_progress_reward"] = active_progress_reward

        # ===== goal progress shaping =====
        goal_progress_reward = 0.0
        goal_progress_delta_raw = 0.0
        goal_reached_bonus = 0.0

        goal_distance_raw = self._get_goal_distance_raw()

        if self.enable_goal_progress_reward:
            if self.prev_final_distance is not None:
                goal_progress_delta_raw = float(self.prev_final_distance - goal_distance_raw)
                goal_progress_reward = self.goal_progress_weight * float(
                    np.clip(goal_progress_delta_raw, -2.0, 2.0)
                )
                reward += goal_progress_reward

            if (
                    self.current_waypoint_idx >= len(self.mission_route) - 1
                    and goal_distance_raw < self.goal_tolerance
            ):
                goal_reached_bonus = self.goal_reached_bonus
                reward += goal_reached_bonus

        info["goal_progress_delta_raw"] = goal_progress_delta_raw
        info["goal_progress_reward"] = goal_progress_reward
        info["goal_reached_bonus"] = goal_reached_bonus

        step_penalty = -0.001
        reward += step_penalty
        info["step_penalty"] = step_penalty

        info["speed_norm"] = speed_norm
        info["action_norm"] = action_norm
        info["local_distance"] = local_distance
        info["final_distance"] = final_distance
        info["local_target_changed_this_step_dbg"] = 1.0 if self._dbg_local_target_changed_this_step else 0.0
        info["local_path_rebuilt_this_step_dbg"] = 1.0 if self._dbg_local_path_rebuilt_this_step else 0.0
        info["progress_guard_active_dbg"] = 1.0 if self._dbg_progress_guard_active_this_step else 0.0
        info["prev_local_distance_dbg"] = float(self._dbg_prev_local_distance_before_reward)
        info["local_distance_raw_dbg"] = float(self._dbg_local_distance_raw_before_reward)
        info["progress_delta_raw_dbg"] = float(self._dbg_progress_delta_raw_before_reward)
        info["planner_replan_step_dbg"] = 1.0 if self._dbg_planner_replan_step_this_step else 0.0

        self.prev_local_distance = local_distance_raw
        self.prev_final_distance = goal_distance_raw
        self.prev_min_obstacle_dist = min_obstacle_dist
        return float(reward), info

    def _accumulate_step_stats(self, reward_info, obs):
        keys = [
            "progress_reward",
            "goal_reward",
            "waypoint_reward",
            "collision_penalty",
            "obstacle_penalty",
            "obstacle_clearance_reward",
            "frontal_block_penalty",
            "action_penalty",
            "altitude_penalty",
            "local_z_error_penalty",
            "local_z_hold_reward",
            "forward_facing_reward",
            "route_deviation_penalty",
            "planner_alignment_reward",
            "risk_penalty",
            "inactive_penalty",
            "active_progress_reward",
            "step_penalty",
            "timeout_penalty",
            "caps_smooth_penalty",
            "asap_smooth_penalty",
            "goal_progress_reward",
            "goal_reached_bonus",
        ]
        for key in keys:
            val = float(reward_info.get(key, 0.0))
            self.episode_stats.setdefault(f"{key}_sum", 0.0)
            self.episode_stats.setdefault(f"{key}_abs_sum", 0.0)
            self.episode_stats[f"{key}_sum"] += val
            self.episode_stats[f"{key}_abs_sum"] += abs(val)

        self.heading_cos_sum += float(obs[19])
        self.min_front_dist = min(self.min_front_dist, float(obs[26]))
        self.min_left_dist = min(self.min_left_dist, float(obs[25]))
        self.min_right_dist = min(self.min_right_dist, float(obs[27]))
        self.min_up_dist = min(self.min_up_dist, float(obs[28]))
        self.min_down_dist = min(self.min_down_dist, float(obs[29]))
        current_speed = float(np.linalg.norm([obs[3], obs[4], obs[5]]))
        self.speed_sum += current_speed

    def _build_episode_summary(self, final_distance_norm, final_distance_raw, term_reason, truncated, mean_heading_error_cos, mean_speed, obs):
        nav = self._decode_base_obs(obs)

        route_progress_norm = 0.0
        if self.route_total_length > 1e-6:
            traveled = max(0.0, self.route_total_length - self._get_remaining_route_distance())
            route_progress_norm = float(np.clip(traveled / self.route_total_length, 0.0, 1.0))

        summary = {
            "episode": self.episode_index,
            "total_reward": float(self.episode_reward),
            "episode_length": int(self.current_step),
            "final_distance": float(final_distance_norm),
            "final_distance_raw": float(final_distance_raw),
            "success": int(term_reason == "goal_reached"),
            "collision": int(term_reason == "collision"),
            "timeout": int(truncated),
            "terminated_reason": str(term_reason) if term_reason is not None else "truncated",
            "final_heading_error_cos": float(nav["heading_local_cos"]),
            "mean_heading_error_cos": float(mean_heading_error_cos),
            "min_front_dist": float(self.min_front_dist),
            "min_left_dist": float(self.min_left_dist),
            "min_right_dist": float(self.min_right_dist),
            "min_up_dist": float(self.min_up_dist),
            "min_down_dist": float(self.min_down_dist),
            "mean_speed": float(mean_speed),
            "goal_reached_step": int(self.current_step) if term_reason == "goal_reached" else -1,
            "route_progress_norm": float(route_progress_norm),
            "route_deviation_norm": float(nav["route_deviation_norm"]),
            "waypoint_idx": int(self.current_waypoint_idx),
            "route_idx": int(self.current_route_idx),
            "inactive_steps": int(self.inactive_steps),
            "no_progress_steps": int(self.no_progress_steps),
            "last_progress_delta": float(self.last_progress_delta),
            "last_speed_norm": float(self.last_speed_norm),
        }
        abs_total = sum(v for k, v in self.episode_stats.items() if k.endswith("_abs_sum"))
        abs_total = max(abs_total, 1e-8)
        for key, value in self.episode_stats.items():
            summary[key] = float(value)
            if key.endswith("_abs_sum"):
                dominance_key = key.replace("_abs_sum", "_dominance")
                summary[dominance_key] = float(value / abs_total)
        return summary

    # =========================================================
    # Termination
    # =========================================================

    def _check_terminated(self, obs: np.ndarray):
        collision_flag = self._get_collision_flag_raw()
        final_distance_raw = self._get_goal_distance_raw()
        altitude_norm = self._get_altitude_norm_raw()

        final_goal_reached = (
                self.current_waypoint_idx >= len(self.mission_route) - 1
                and final_distance_raw < self.goal_tolerance
        )

        if collision_flag > 0.5:
            print(f"[TERM] step={int(self.current_step)} reason=collision")
            return True, "collision"

        if self._is_ground_contact_crash():
            print(f"[TERM] step={int(self.current_step)} reason=ground_contact")
            return True, "ground_contact"

        if final_goal_reached:
            return True, "goal_reached"

        if self._is_out_of_bounds():
            return True, "out_of_bounds"

        if altitude_norm < 0.08:
            return True, "too_low"

        if self.inactive_steps >= self.inactive_max_steps:
            return True, "inactive_hover"

        if self.no_progress_steps >= self.no_progress_max_steps:
            return True, "no_progress"

        return False, None

    def _is_out_of_bounds(self) -> bool:
        pos = self._current_position_vec()
        limit = float(self.world_xy_limit + self.world_xy_oob_margin)
        return bool(abs(pos[0]) > limit or abs(pos[1]) > limit)

    def _get_altitude_norm_raw(self) -> float:
        pos = self._current_position_vec()
        altitude = float(-pos[2])  # AirSim NED: z < 0 значит выше
        return float(np.clip(altitude / max(float(self.max_altitude), 1e-6), 0.0, 1.0))

    def _get_goal_distance_raw(self) -> float:
        pos = self._current_position_vec()
        return float(np.linalg.norm(self.goal_position - pos))

    def _post_reset_route_sync(self) -> None:
        """
        Hook для потомков.
        База ничего не делает.
        Потомки могут восстановить свой route/goal state
        после базового reset без ручной 'починки' в reset().
        """
        return



    def _get_collision_flag_raw(self) -> float:
        try:
            collided = bool(self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided)
            if collided:
                self.collision_latched = True
            return float(self.collision_latched)
        except Exception:
            return float(self.collision_latched)

    def _is_ground_contact_crash(self) -> bool:
        # Небольшая защита после reset/takeoff:
        # AirSim иногда ещё держит landed_state=True, хотя дрон уже перенесён/висит в воздухе.
        if int(getattr(self, "current_step", 0)) < 8:
            self.ground_contact_steps = 0
            self.last_down_distance = float(self.max_sensor_distance)
            self.last_landed_state = 0.0
            self.last_tilt_abs = 0.0
            self.last_ground_speed = 0.0
            return False

        try:
            down = self.client.getDistanceSensorData(
                distance_sensor_name="DownDistance",
                vehicle_name=self.vehicle_name,
            )
            down_dist = float(down.distance)
            if not np.isfinite(down_dist) or down_dist <= 0.0:
                down_dist = self.max_sensor_distance
        except Exception:
            down_dist = self.max_sensor_distance

        self.last_down_distance = float(down_dist)

        state = self._get_multirotor_state_cached()
        kin = state.kinematics_estimated

        try:
            landed = int(state.landed_state) != 0
        except Exception:
            landed = False

        vel = kin.linear_velocity
        speed = float(np.sqrt(
            vel.x_val ** 2 +
            vel.y_val ** 2 +
            vel.z_val ** 2
        ))

        roll, pitch, _ = airsim.to_eularian_angles(kin.orientation)
        tilt_abs = float(max(abs(roll), abs(pitch)))
        altitude = float(-kin.position.z_val)

        near_ground_sensor = down_dist < float(self.ground_contact_down_threshold)
        too_low_global = altitude < max(0.35, 0.65 * float(self.min_altitude))
        lying_attitude = tilt_abs > np.deg2rad(45.0)
        almost_static = speed < float(self.ground_contact_speed_threshold)

        self.last_landed_state = float(1.0 if landed else 0.0)
        self.last_tilt_abs = float(tilt_abs)
        self.last_ground_speed = float(speed)

        # ВАЖНО:
        # landed_state сам по себе НЕ считается падением.
        # Он учитывается только если дрон реально низко или датчик вниз видит землю.
        landed_near_ground = landed and (near_ground_sensor or too_low_global)

        ground_like = (
            (self.collision_latched and (near_ground_sensor or too_low_global))
            or landed_near_ground
            or (near_ground_sensor and almost_static)
            or (too_low_global and almost_static)
            or (lying_attitude and almost_static and (near_ground_sensor or too_low_global))
        )

        if ground_like:
            self.ground_contact_steps += 1
        else:
            self.ground_contact_steps = 0

        return self.ground_contact_steps >= int(self.ground_contact_steps_required)
    def _get_remaining_route_distance(self) -> float:
        """
        Оставшаяся длина маршрута от текущей позиции до конца mission_route.
        Нужна для episode_summary.
        """
        if not self.mission_route:
            return float(self._get_goal_distance_raw())

        if self.current_waypoint_idx >= len(self.mission_route):
            return 0.0

        pos = self._current_position_vec()
        idx = int(np.clip(self.current_waypoint_idx, 0, len(self.mission_route) - 1))

        remaining = 0.0
        anchor = pos.astype(np.float32)

        for p in self.mission_route[idx:]:
            point = np.asarray(p, dtype=np.float32)
            remaining += float(np.linalg.norm(point - anchor))
            anchor = point

        return float(max(0.0, remaining))

    def _sample_start_and_route_pair(self):
        if self.start_points is None or len(self.start_points) == 0:
            return self._sample_start_position(), None, None

        if self.route_sets is None or len(self.route_sets) == 0:
            return self._sample_start_position(), None, None

        if len(self.start_points) != len(self.route_sets):
            raise ValueError(
                f"paired_start_route=True requires len(start_points) == len(route_sets), "
                f"got {len(self.start_points)} != {len(self.route_sets)}"
            )

        idx = int(self.np_random.integers(0, len(self.start_points)))
        start = np.asarray(self.start_points[idx], dtype=np.float32).copy()
        route = self.route_sets[idx].copy()
        return start, idx, route
    def _invalidate_state_cache(self):
        self._cached_multirotor_state = None
        self._cached_position_vec = None

    def _get_multirotor_state_cached(self):
        if self._cached_multirotor_state is None:
            self._cached_multirotor_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        return self._cached_multirotor_state

    def _compose_nav_visual_features(self):
        if self.yolo_feature_dim <= 0:
            self.yolo_features = np.zeros(0, dtype=np.float32)
            return

        vals = np.zeros(self.yolo_feature_dim, dtype=np.float32)

        mode = getattr(self, "nav_visual_obs_mode", "safety_only")

        if mode == "safety_only":
            n = min(8, self.yolo_feature_dim, self.safety_features.shape[0])
            vals[:n] = self.safety_features[:n]

        elif mode == "dual16":
            n1 = min(8, self.yolo_feature_dim, self.safety_features.shape[0])
            vals[:n1] = self.safety_features[:n1]

            if self.yolo_feature_dim > 8:
                n2 = min(8, self.yolo_feature_dim - 8, self.infra_features.shape[0])
                vals[8:8 + n2] = self.infra_features[:n2]

        elif mode == "blend8":
            if self.yolo_feature_dim >= 8:
                safety_visible = float(self.safety_features[0]) if self.safety_features.shape[0] >= 1 else 0.0
                infra_visible = float(self.infra_features[0]) if self.infra_features.shape[0] >= 1 else 0.0

                if safety_visible > 0.0 and infra_visible > 0.0:
                    vals[:] = np.array([
                        max(float(self.safety_features[0]), float(self.infra_features[0])),
                        float(self.safety_features[1]),
                        max(float(self.safety_features[2]), float(self.infra_features[2])),
                        float(self.safety_features[3]),
                        float(self.safety_features[4]),
                        max(float(self.safety_features[5]), float(self.infra_features[5])),
                        0.5 * (float(self.safety_features[6]) + float(self.infra_features[6])),
                        0.5 * (float(self.safety_features[7]) + float(self.infra_features[7])),
                    ], dtype=np.float32)
                elif safety_visible > 0.0:
                    vals[:8] = self.safety_features[:8]
                elif infra_visible > 0.0:
                    vals[:8] = self.infra_features[:8]

        else:
            n = min(8, self.yolo_feature_dim, self.safety_features.shape[0])
            vals[:n] = self.safety_features[:n]

        self.yolo_features = vals.astype(np.float32)

    def _setup_nav_visual_pipeline(self):
        self._safety_model = None
        self._infra_model = None
        self._nav_visual_backend_ready = False

        if not self.enable_nav_visual:
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:
            print(f"[NAV_VISUAL] ultralytics import failed: {exc}")
            return

        if self.enable_nav_safety_detector and self.safety_detector_model_path:
            try:
                self._safety_model = YOLO(self.safety_detector_model_path)
                print(f"[NAV_VISUAL] bird/safety model loaded: {self.safety_detector_model_path}")

                if self.nav_use_deep_sort:
                    try:
                        self.nav_deep_sort_tracker = DeepSort(
                            max_age=self.nav_deep_sort_max_age,
                            n_init=self.nav_deep_sort_n_init,
                            max_iou_distance=self.nav_deep_sort_max_iou_distance,
                        )
                        print("[NAV_VISUAL] Deep SORT initialized")
                    except Exception as exc:
                        self.nav_deep_sort_tracker = None
                        print(f"[NAV_VISUAL] Deep SORT init failed: {exc}")
            except Exception as exc:
                self._safety_model = None
                print(f"[NAV_VISUAL] failed to load safety model: {exc}")

        if self.enable_nav_seg_context and self.infra_detector_model_path:
            try:
                self._infra_model = YOLO(self.infra_detector_model_path)
                print(f"[NAV_VISUAL] infra model loaded: {self.infra_detector_model_path}")
            except Exception as exc:
                self._infra_model = None
                print(f"[NAV_VISUAL] failed to load infra model: {exc}")

        self._nav_visual_backend_ready = (self._safety_model is not None) or (self._infra_model is not None)

    def _get_scene_image_bgr(self):
        if not self.enable_nav_visual:
            return None

        try:
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        self.scene_camera_name,
                        self.scene_image_type,
                        pixels_as_float=False,
                        compress=False,
                    )
                ],
                vehicle_name=self.vehicle_name,
            )
            if not responses:
                return None

            resp = responses[0]
            if resp.width <= 0 or resp.height <= 0 or len(resp.image_data_uint8) == 0:
                return None

            img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            if img.size != resp.height * resp.width * 3:
                return None

            img = img.reshape(resp.height, resp.width, 3)
            return img
        except Exception:
            return None

    def _run_nav_detector(self, image_bgr, model, target_class_names, conf_threshold, iou_threshold):
        if model is None or image_bgr is None:
            return []

        try:
            small = cv2.resize(image_bgr, (self.nav_visual_input_size, self.nav_visual_input_size))
            results = model.predict(
                source=small,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=self.nav_visual_input_size,
                verbose=False,
            )
        except Exception:
            return []

        detections = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None) or {}

        if boxes is None:
            return detections

        h, w = image_bgr.shape[:2]
        infer_h, infer_w = small.shape[:2]
        scale_x = float(w) / max(float(infer_w), 1.0)
        scale_y = float(h) / max(float(infer_h), 1.0)

        xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else []
        clss = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else []

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            if bw <= 1e-6 or bh <= 1e-6:
                continue

            cls_id_int = int(cls_id)
            class_name = str(names.get(cls_id_int, cls_id_int)).lower()
            if target_class_names is not None and class_name not in target_class_names:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            detections.append({
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "conf": float(conf),
                "class_id": cls_id_int,
                "class_name": class_name,
                "center_x": float(cx / max(w, 1.0)),
                "center_y": float(cy / max(h, 1.0)),
                "area_norm": float((bw * bh) / max(float(w * h), 1e-6)),
                "dx_from_center": float((cx / max(w, 1.0)) - 0.5),
                "dy_from_center": float((cy / max(h, 1.0)) - 0.5),
            })

        return detections

    def _select_nav_detection(self, detections):
        if not detections:
            return None

        prev = getattr(self, "_last_nav_selected", None)

        def score(det):
            center_dist = float(np.sqrt(det["dx_from_center"] ** 2 + det["dy_from_center"] ** 2))
            score_val = 2.0 * float(det["conf"]) + 0.5 * float(det["area_norm"])
            score_val -= 0.75 * center_dist

            if prev is not None:
                prev_dx = float(det["center_x"] - prev.get("center_x", 0.5))
                prev_dy = float(det["center_y"] - prev.get("center_y", 0.5))
                temporal_dist = float(np.sqrt(prev_dx ** 2 + prev_dy ** 2))
                score_val -= 0.50 * temporal_dist

            return score_val

        selected = max(detections, key=score)
        self._last_nav_selected = dict(selected)
        return selected

    def _detection_to_nav_features(self, detection):
        vals = np.zeros(8, dtype=np.float32)
        if detection is None:
            return vals

        vals[:] = np.array([
            1.0,
            float(np.clip(detection.get("class_id", 0.0) / 10.0, 0.0, 1.0)),
            float(np.clip(detection.get("conf", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("center_x", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("center_y", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("area_norm", 0.0), 0.0, 1.0)),
            float(np.clip(detection.get("dx_from_center", 0.0), -1.0, 1.0)),
            float(np.clip(detection.get("dy_from_center", 0.0), -1.0, 1.0)),
        ], dtype=np.float32)
        return vals

    def _update_nav_visual_features(self):
        if self.yolo_feature_dim <= 0:
            return

        self.safety_features[:] = 0.0
        self.infra_features[:] = 0.0
        self.visual_hazard_active = False
        self.visual_hazard_score = 0.0

        if not self.enable_nav_visual:
            self._compose_nav_visual_features()
            return

        image = self._get_scene_image_bgr()
        if image is None:
            self._compose_nav_visual_features()
            return

        safety_dets = []
        if self.enable_nav_safety_detector:
            safety_dets = self._run_nav_detector(
                image_bgr=image,
                model=self._safety_model,
                target_class_names=self.safety_target_class_names,
                conf_threshold=self.safety_conf_threshold,
                iou_threshold=self.nav_visual_iou_threshold,
            )

        selected = None
        if self.nav_use_deep_sort and self.nav_deep_sort_tracker is not None:
            tracks = self._update_nav_deep_sort_tracks(image, safety_dets)
            self.nav_last_tracks = [dict(t) for t in tracks]
            selected = self._select_nav_track(tracks)

        if selected is None and safety_dets:
            selected = self._select_nav_detection(safety_dets)

        if selected is not None:
            self.safety_features[:] = self._detection_to_nav_features(selected)
            self.visual_hazard_score = float(selected.get("conf", 0.0))

            if self.visual_hazard_score >= self.visual_hazard_threshold:
                self.visual_hazard_active = True
                self.visual_hazard_confirm_steps += 1
                self.visual_hazard_clear_counter = 0
            else:
                self.visual_hazard_confirm_steps = 0
        else:
            self.visual_hazard_confirm_steps = 0

        self._update_infra_context_features(image)

        self.visual_replan_active = bool(self.visual_hazard_active and self.shift_mode_steps_left > 0)
        self._compose_nav_visual_features()

    def _visual_hazard_local_replan_if_needed(self):
        if not self.visual_hazard_replan_enabled:
            return

        if not self.visual_hazard_active:
            return

        if self.shift_mode_steps_left > 0:
            self.visual_replan_active = True
            self.visual_replan_steps_left = self.shift_mode_steps_left
            return

        # safety_features:
        # [0]=visible, [1]=class_id_norm, [2]=conf, [3]=cx, [4]=cy,
        # [5]=area, [6]=dx_from_center, [7]=dy_from_center
        visible = float(self.safety_features[0]) if self.yolo_feature_dim >= 1 else 0.0
        conf = float(self.safety_features[2]) if self.yolo_feature_dim >= 3 else 0.0
        area = float(self.safety_features[5]) if self.yolo_feature_dim >= 6 else 0.0
        dx = float(self.safety_features[6]) if self.yolo_feature_dim >= 7 else 0.0

        # усиливаем реакцию на реально крупный / уверенный объект
        hazard_score = conf * (1.0 + 2.0 * area) * (1.0 - min(abs(dx), 1.0))
        if visible <= 0.0 or hazard_score < max(0.15, 0.5 * self.visual_hazard_threshold):
            return

        # приоритет — выбрать сторону с большим запасом по боковым датчикам
        # если датчики не доступны, fallback на знак dx_from_center
        if self.min_left_dist >= self.min_right_dist:
            lateral_sign = 1.0
        else:
            lateral_sign = -1.0

        # если объект сильно смещён в кадре, дополнительно подталкиваем уход в противоположную сторону
        if abs(dx) > 0.12:
            lateral_sign = -1.0 if dx > 0.0 else 1.0

        # если спереди уже тесно, делаем более резкий обход
        front_norm = float(self.min_front_dist / max(self.max_sensor_distance, 1e-6))
        shift_distance = float(self.visual_replan_shift_distance)
        if front_norm < 0.18:
            shift_distance *= 1.35

        self.shift_vector = np.array(
            [0.0, lateral_sign * shift_distance, 0.0],
            dtype=np.float32,
        )

        self.shift_mode_steps_left = int(self.visual_replan_hold_steps)
        self.visual_replan_active = True
        self.visual_replan_steps_left = int(self.visual_replan_hold_steps)

        # мягкий дополнительный brake, чтобы не пролетать сквозь стаю
        if hasattr(self, "front_avoid_counter"):
            self.front_avoid_counter = max(self.front_avoid_counter, int(self.visual_replan_hold_steps))

    def _get_applied_command_vector_norm(self) -> np.ndarray:
        vx = float(np.clip(
            self.last_applied_vx_cmd / max(self.max_velocity, 1e-6),
            -1.0, 1.0
        ))
        vy = float(np.clip(
            self.last_applied_vy_cmd / max(self.max_velocity, 1e-6),
            -1.0, 1.0
        ))
        vz = float(np.clip(
            self.last_applied_vz_cmd / max(self.max_vertical_speed, 1e-6),
            -1.0, 1.0
        ))
        yaw = float(np.clip(
            self.last_applied_yaw_rate_cmd / max(self.max_yaw_rate, 1e-6),
            -1.0, 1.0
        ))
        return np.array([vx, vy, vz, yaw], dtype=np.float32)

    def _get_visual_debug_info(self) -> dict:

        return {
            "enable_nav_visual": float(getattr(self, "enable_nav_visual", False)),
        "enable_nav_seg_context": float(getattr(self, "enable_nav_seg_context", False)),
        "enable_nav_safety_detector": float(getattr(self, "enable_nav_safety_detector", False)),

        "visual_hazard_active": float(getattr(self, "visual_hazard_active", False)),
        "visual_hazard_score": float(getattr(self, "visual_hazard_score", 0.0)),
        "visual_hazard_confirm_steps": int(getattr(self, "visual_hazard_confirm_steps", 0)),
        "visual_hazard_clear_counter": int(getattr(self, "visual_hazard_clear_counter", 0)),

        "visual_replan_active": float(getattr(self, "visual_replan_active", False)),
        "visual_replan_steps_left": int(getattr(self, "visual_replan_steps_left", 0)),

        "safety_visible": float(getattr(self, "safety_features", np.zeros(8))[0]),
        "safety_conf": float(getattr(self, "safety_features", np.zeros(8))[2]),
        "safety_area": float(getattr(self, "safety_features", np.zeros(8))[5]),
        "safety_dx": float(getattr(self, "safety_features", np.zeros(8))[6]),
        "safety_hazard_score": float(getattr(self, "safety_features", np.zeros(8))[7]),

        "infra_visible": float(getattr(self, "infra_features", np.zeros(8))[0]),
        "infra_class_norm": float(getattr(self, "infra_features", np.zeros(8))[1]),
        "infra_conf": float(getattr(self, "infra_features", np.zeros(8))[2]),
        "infra_cx": float(getattr(self, "infra_features", np.zeros(8))[3]),
        "infra_cy": float(getattr(self, "infra_features", np.zeros(8))[4]),
        "infra_area": float(getattr(self, "infra_features", np.zeros(8))[5]),
        "infra_dx": float(getattr(self, "infra_features", np.zeros(8))[6]),
        "infra_dy": float(getattr(self, "infra_features", np.zeros(8))[7]),

        }

    def _update_nav_deep_sort_tracks(self, image_bgr, detections):
        if self.nav_deep_sort_tracker is None or image_bgr is None:
            return []

        ds_inputs = []
        for det in detections or []:
            x1 = float(det.get("x1", 0.0))
            y1 = float(det.get("y1", 0.0))
            x2 = float(det.get("x2", 0.0))
            y2 = float(det.get("y2", 0.0))
            conf = float(det.get("conf", 0.0))
            cls_name = str(det.get("class_name", "bird")).lower()

            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            bbox_ltrb = [x1, y1, x2, y2]

            # deep_sort_realtime ожидает: (bbox, confidence, class_name)
            ds_inputs.append((bbox_ltrb, conf, cls_name))

        try:
            tracks = self.nav_deep_sort_tracker.update_tracks(ds_inputs, frame=image_bgr)
        except Exception:
            return []

        out = []
        for trk in tracks or []:
            try:
                if not trk.is_confirmed():
                    continue
            except Exception:
                continue

            try:
                ltrb = trk.to_ltrb()
            except Exception:
                continue

            x1, y1, x2, y2 = [float(v) for v in ltrb]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            img_h, img_w = image_bgr.shape[:2]
            out.append({
                "track_id": int(trk.track_id),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": float(cx / max(img_w, 1.0)),
                "center_y": float(cy / max(img_h, 1.0)),
                "area": float((w * h) / max(float(img_w * img_h), 1e-6)),
                "time_since_update": int(getattr(trk, "time_since_update", 0)),
                "hits": int(getattr(trk, "hits", 0)),
                "age": int(getattr(trk, "age", 0)),
                "class_name": str(
                    trk.det_class if hasattr(trk, "det_class") and trk.det_class is not None else "bird").lower(),
            })
        return out

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        angle_deg = (float(angle_deg) + 180.0) % 360.0 - 180.0
        return float(angle_deg)

    def _sync_nav_fixed_yaw_target(self):
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            self.filtered_yaw_target_deg = float(np.rad2deg(yaw))
        except Exception:
            self.filtered_yaw_target_deg = 0.0

    def _apply_nav_controller_gains(self):
        if not self.nav_apply_controller_gains:
            return
        try:
            self.client.setVelocityControllerGains(
                airsim.VelocityControllerGains(
                    x_gains=airsim.PIDGains(self.nav_velocity_kp_xy, 0.0, self.nav_velocity_kd_xy),
                    y_gains=airsim.PIDGains(self.nav_velocity_kp_xy, 0.0, self.nav_velocity_kd_xy),
                    z_gains=airsim.PIDGains(self.nav_velocity_kp_z, self.nav_velocity_ki_z, self.nav_velocity_kd_z),
                ),
                vehicle_name=self.vehicle_name,
            )

            self.client.setAngleLevelControllerGains(
                airsim.AngleLevelControllerGains(
                    roll_gains=airsim.PIDGains(self.nav_angle_level_kp_rp, 0.0, self.nav_angle_level_kd_rp),
                    pitch_gains=airsim.PIDGains(self.nav_angle_level_kp_rp, 0.0, self.nav_angle_level_kd_rp),
                    yaw_gains=airsim.PIDGains(self.nav_angle_level_kp_yaw, 0.0, self.nav_angle_level_kd_yaw),
                ),
                vehicle_name=self.vehicle_name,
            )

            self.client.setAngleRateControllerGains(
                airsim.AngleRateControllerGains(
                    roll_gains=airsim.PIDGains(self.nav_angle_rate_kp_rp, 0.0, self.nav_angle_rate_kd_rp),
                    pitch_gains=airsim.PIDGains(self.nav_angle_rate_kp_rp, 0.0, self.nav_angle_rate_kd_rp),
                    yaw_gains=airsim.PIDGains(self.nav_angle_rate_kp_yaw, 0.0, self.nav_angle_rate_kd_yaw),
                ),
                vehicle_name=self.vehicle_name,
            )
        except Exception as exc:
            print(f"[NAV GAINS] apply failed: {exc}")




    def _select_nav_track(self, tracks):
        if not tracks:
            self.nav_locked_track_missed += 1
            if self.nav_locked_track_missed > self.nav_track_keep_max_missed:
                self.nav_locked_track_id = None
                self.nav_locked_waypoint_idx = -1
            return None

        # если уже держим трек — пытаемся удержать его
        if self.nav_locked_track_id is not None:
            for trk in tracks:
                if int(trk["track_id"]) == int(self.nav_locked_track_id):
                    self.nav_locked_track_missed = 0
                    return trk

        # иначе берём самый "опасный": ближе к центру и крупнее
        best = None
        best_score = -1e9
        for trk in tracks:
            cx = float(trk["center_x"])
            cy = float(trk["center_y"])
            area = float(trk["area"])
            conf = float(trk.get("conf", 0.0))

            dx = cx - 0.5
            dy = cy - 0.5
            center_dist = float(np.sqrt(dx * dx + dy * dy))

            score = 2.0 * conf + 1.5 * area - 1.0 * center_dist
            if score > best_score:
                best_score = score
                best = trk

        if best is not None:
            self.nav_locked_track_id = int(best["track_id"])
            self.nav_locked_track_missed = 0

        return best

    def _decode_base_obs(self, obs: np.ndarray) -> Dict[str, float]:
        return {k: float(obs[i]) for k, i in self._NAV_OBS_INDEX.items()}
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
