import os
import re
from typing import Optional
import json
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import FloatSchedule
from ppo_train_nav import (
    EpisodeCsvLoggerCallback,
    PPOWithTrainCsv,
    StepRewardCsvLoggerCallback,
    LOG_ROOT as BASE_LOG_ROOT,
    MODEL_ROOT as BASE_MODEL_ROOT,
    make_env,
)




# Если None, будет найден самый свежий checkpoint в BASE_MODEL_ROOT/checkpoints.
LOAD_MODEL_PATH: Optional[str] = "models/ppo_nav/train21_last/ppo_nav_final_continued.zip"


TOTAL_TIMESTEPS_TO_ADD = 20480
LOG_ROOT = "logs/ppo_nav/train22_last"
MODEL_ROOT = "models/ppo_nav/train22_last"


def _extract_steps(filename: str) -> int:
    match = re.search(r"_(\d+)_steps\\.zip$", filename)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Папка checkpoints не найдена: {checkpoint_dir}")

    candidates = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".zip")
    ]

    if not candidates:
        raise FileNotFoundError(f"В папке нет checkpoint-файлов: {checkpoint_dir}")

    candidates.sort(key=lambda p: (_extract_steps(os.path.basename(p)), os.path.getmtime(p)))
    return candidates[-1]


def resolve_load_path() -> str:
    if LOAD_MODEL_PATH:
        if not os.path.exists(LOAD_MODEL_PATH):
            raise FileNotFoundError(f"Checkpoint не найден: {LOAD_MODEL_PATH}")
        return LOAD_MODEL_PATH
    return find_latest_checkpoint(os.path.join(BASE_MODEL_ROOT, "checkpoints"))


if __name__ == "__main__":
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(os.path.join(MODEL_ROOT, "checkpoints"), exist_ok=True)

    env = make_env()
    load_path = resolve_load_path()
    print(f"Loading checkpoint: {load_path}")

    model = PPOWithTrainCsv.load(
        load_path,
        env=env,
        device="cpu",

    )

    model.lr_schedule = FloatSchedule(1e-4)
    model.clip_range = FloatSchedule(0.15)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = 1e-4

    model.n_steps = 1024
    model.batch_size = 64
    model.n_epochs = 10

    model.gamma = 0.99
    model.gae_lambda = 0.95



    model.ent_coef = 0.0005
    model.vf_coef = 0.5
    model.max_grad_norm = 0.5

    model.train_metrics_csv_path = os.path.join(LOG_ROOT, "train_metrics.csv")

    sb3_logger = configure(os.path.join(LOG_ROOT, "sb3"), ["stdout", "csv", "tensorboard"])
    model.set_logger(sb3_logger)

    # ========= Обновить optimizer под новый learning rate =========
    model._update_learning_rate(model.policy.optimizer)

    # ========= Обновить rollout buffer =========
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )
    sb3_logger = configure(os.path.join(LOG_ROOT, "sb3"), ["stdout", "csv", "tensorboard"])
    model.set_logger(sb3_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path=os.path.join(MODEL_ROOT, "checkpoints"),
        name_prefix="ppo_nav_continue",
    )

    callback = CallbackList([
        checkpoint_callback,
        EpisodeCsvLoggerCallback(os.path.join(LOG_ROOT, "episode_metrics.csv")),
        StepRewardCsvLoggerCallback(os.path.join(LOG_ROOT, "reward_trace.csv")),
    ])

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_TO_ADD,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(os.path.join(MODEL_ROOT, "ppo_nav_final_continued"))
    finally:
        env.close()
