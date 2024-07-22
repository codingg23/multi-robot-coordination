"""
train.py

Train the coordination policy using PPO from stable-baselines3.

The policy is shared across all robots. During training, each episode
randomises the room layout and obstacle positions so the policy has to
generalise rather than memorise.

I trained for 500k timesteps on an M1 MacBook (no GPU).
Took about 6 hours. The policy converges but it's clearly not optimal -
it explores somewhat randomly rather than systematically.

Usage:
    python -m rl.train --timesteps 500000 --n-robots 3
"""

import argparse
import os
import time
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    print("stable-baselines3 not installed. pip install stable-baselines3")
    SB3_AVAILABLE = False

from .sim_env import MultiRobotEnv


def make_env(n_robots: int = 3, seed: int = 0):
    def _init():
        env = MultiRobotEnv(n_robots=n_robots, seed=seed)
        return env
    return _init


def train(
    timesteps: int = 500_000,
    n_robots: int = 3,
    n_envs: int = 4,
    save_dir: str = "rl/checkpoints",
    eval_freq: int = 10_000,
):
    if not SB3_AVAILABLE:
        return

    os.makedirs(save_dir, exist_ok=True)

    print(f"Training: {timesteps} timesteps, {n_robots} robots, {n_envs} parallel envs")

    # parallel envs for faster data collection
    env = make_vec_env(make_env(n_robots), n_envs=n_envs)
    eval_env = MultiRobotEnv(n_robots=n_robots, seed=9999)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,   # small entropy bonus - helps with exploration
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(eval_freq // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_swarm",
        ),
    ]

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/3600:.1f}h")

    model.save(os.path.join(save_dir, "final_policy"))
    print(f"Saved to {save_dir}/final_policy")


def evaluate(model_path: str, n_robots: int = 3, n_episodes: int = 10):
    """Quick evaluation of a trained policy."""
    if not SB3_AVAILABLE:
        return

    model = PPO.load(model_path)
    env = MultiRobotEnv(n_robots=n_robots)

    episode_rewards = []
    episode_lengths = []
    found_counts = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        found_counts.append(len(env.target_found_by))

    print(f"Evaluation ({n_episodes} episodes):")
    print(f"  Mean reward: {np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}")
    print(f"  Mean length: {np.mean(episode_lengths):.0f} steps")
    print(f"  Robots that found target: {np.mean(found_counts):.1f}/{n_robots}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-robots", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", default="rl/checkpoints")
    parser.add_argument("--eval", metavar="MODEL_PATH", help="evaluate a saved model")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, n_robots=args.n_robots)
    else:
        train(args.timesteps, args.n_robots, args.n_envs, args.save_dir)
