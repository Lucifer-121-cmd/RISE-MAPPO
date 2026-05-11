"""MAPPO training loop runner.

A minimal, single-process training loop that drives an iterable of
:class:`MultiRobotSearchEnv` instances (one per "vec env"). The runner
deliberately does *not* use multiprocessing: per the master prompt we
get the single-env path right first.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from envs.multi_robot_search_env import EnvConfig, MultiRobotSearchEnv
from marl.mappo.actor import Actor, ActorConfig
from marl.mappo.algorithm import MAPPO, MAPPOConfig
from marl.mappo.buffer import BufferConfig, RolloutBuffer
from marl.mappo.critic import Critic, CriticConfig
from marl.utils import set_global_seed


_LOG = logging.getLogger("paper3.runner")


@dataclass
class RunnerConfig:
    rollout_length: int = 256
    num_envs: int = 4
    n_training_updates: int = 200
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 25
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "results/checkpoints"
    use_wandb: bool = False
    wandb_project: str = "paper3-marl-search"
    wandb_run_name: Optional[str] = None
    extra_meta: Dict[str, str] = field(default_factory=dict)


class Runner:
    """Drive collection → buffer fill → MAPPO update."""

    def __init__(
        self,
        env_factory: Callable[[int], MultiRobotSearchEnv],
        actor: Actor,
        critic: Critic,
        algo: MAPPO,
        runner_cfg: RunnerConfig,
    ) -> None:
        self.runner_cfg = runner_cfg
        self.actor = actor
        self.critic = critic
        self.algo = algo
        self.device = torch.device(runner_cfg.device if torch.cuda.is_available() or runner_cfg.device == "cpu" else "cpu")
        # Build envs.
        self.envs: List[MultiRobotSearchEnv] = [
            env_factory(runner_cfg.seed + i) for i in range(runner_cfg.num_envs)
        ]
        self.num_agents = self.envs[0].cfg.num_robots
        self.local_shapes = self.envs[0].observation_shapes()
        self.global_shapes = self.envs[0].global_state_shapes()
        self.use_rise = bool(getattr(self.algo.cfg, "use_rise", False))
        self.buffer = RolloutBuffer(
            BufferConfig(
                rollout_length=runner_cfg.rollout_length,
                num_envs=runner_cfg.num_envs,
                num_agents=self.num_agents,
                device=str(self.device),
                use_rise=self.use_rise,
            ),
            self.local_shapes,
            self.global_shapes,
        )
        Path(runner_cfg.save_dir).mkdir(parents=True, exist_ok=True)
        self._wandb = None
        if runner_cfg.use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb = wandb
                wandb.init(
                    project=runner_cfg.wandb_project,
                    name=runner_cfg.wandb_run_name,
                    config={**runner_cfg.__dict__, **runner_cfg.extra_meta},
                )
            except Exception as exc:  # pragma: no cover
                _LOG.warning("wandb init failed: %s", exc)
                self._wandb = None
        self._reset_envs()

    # ------------------------------------------------------------------
    def _reset_envs(self) -> None:
        self._cur_local: List[Dict[str, Dict[str, np.ndarray]]] = []
        self._cur_global: List[Dict[str, np.ndarray]] = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.runner_cfg.seed + i * 1009)
            self._cur_local.append(obs)
            self._cur_global.append(env.global_state())

    @staticmethod
    def _stack_local(
        per_env: List[Dict[str, Dict[str, np.ndarray]]],
        agents: List[str],
        keys: List[str],
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            arrays = []
            for env_obs in per_env:
                arrays.append(np.stack([env_obs[a][k] for a in agents], axis=0))
            out[k] = np.stack(arrays, axis=0)
        return out

    @staticmethod
    def _stack_global(
        per_env: List[Dict[str, np.ndarray]],
        keys: List[str],
    ) -> Dict[str, np.ndarray]:
        out = {}
        for k in keys:
            out[k] = np.stack([env_state[k] for env_state in per_env], axis=0)
        return out

    # ------------------------------------------------------------------
    def collect_rollout(self) -> Dict[str, float]:
        """Fill ``self.buffer`` with one rollout; return episode stats."""
        E = self.runner_cfg.num_envs
        A = self.num_agents
        agents = self.envs[0].agents
        local_keys = list(self.local_shapes.keys())
        global_keys = list(self.global_shapes.keys())
        episode_rewards: List[float] = []
        episode_coverages: List[float] = []
        episode_detected: List[float] = []
        # Per-env running ep stats.
        ep_r = np.zeros(E, dtype=np.float32)
        for t in range(self.runner_cfg.rollout_length):
            local_stack = self._stack_local(self._cur_local, agents, local_keys)
            global_stack = self._stack_global(self._cur_global, global_keys)
            actions, log_probs = self._sample_actions(local_stack, E, A)
            if self.use_rise:
                values, values_cvar, agent_sigmas = self._compute_values_rise(global_stack)
            else:
                values = self._compute_values(global_stack)
                values_cvar = None
                agent_sigmas = None
            # Step each env with corresponding actions.
            rewards = np.zeros(E, dtype=np.float32)
            risk_costs = np.zeros(E, dtype=np.float32) if self.use_rise else None
            dones = np.zeros(E, dtype=np.float32)
            new_local: List[Dict[str, Dict[str, np.ndarray]]] = []
            new_global: List[Dict[str, np.ndarray]] = []
            for i, env in enumerate(self.envs):
                act_dict = {a: int(actions[i, j]) for j, a in enumerate(agents)}
                obs, rew_dict, term_dict, trunc_dict, info = env.step(act_dict)
                rewards[i] = float(next(iter(rew_dict.values())))
                if self.use_rise:
                    risk_costs[i] = float(info.get("risk_cost", 0.0))
                done = any(term_dict.values()) or any(trunc_dict.values())
                ep_r[i] += rewards[i]
                if done:
                    dones[i] = 1.0
                    episode_rewards.append(float(ep_r[i]))
                    episode_coverages.append(float(info["coverage"]))
                    episode_detected.append(float(info["detected"]))
                    ep_r[i] = 0.0
                    obs, _ = env.reset()
                new_local.append(obs)
                new_global.append(env.global_state())
            self.buffer.insert(
                t=t,
                local_obs=local_stack,
                global_state=global_stack,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
                risk_costs=risk_costs,
                values_cvar=values_cvar,
                agent_sigmas=agent_sigmas,
            )
            self._cur_local = new_local
            self._cur_global = new_global
        # Bootstrap at horizon.
        global_stack = self._stack_global(self._cur_global, global_keys)
        if self.use_rise:
            last_values, last_values_cvar, _ = self._compute_values_rise(global_stack)
        else:
            last_values = self._compute_values(global_stack)
            last_values_cvar = None
        last_dones = np.zeros(E, dtype=np.float32)
        self.buffer.compute_returns_and_advantages(
            last_values, last_dones, last_values_cvar=last_values_cvar,
        )
        return {
            "ep_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "ep_coverage_mean": float(np.mean(episode_coverages)) if episode_coverages else 0.0,
            "ep_detected_mean": float(np.mean(episode_detected)) if episode_detected else 0.0,
            "n_episodes": float(len(episode_rewards)),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_actions(
        self,
        local_stack: Dict[str, np.ndarray],
        E: int,
        A: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # local_stack[k] has shape (E, A, ...).
        flat: Dict[str, torch.Tensor] = {}
        for k, arr in local_stack.items():
            t = torch.from_numpy(arr).to(self.device)
            flat[k] = t.reshape((E * A,) + arr.shape[2:])
        action, log_prob, _ = self.actor.get_action(flat)
        return (
            action.view(E, A).cpu().numpy(),
            log_prob.view(E, A).cpu().numpy(),
        )

    @torch.no_grad()
    def _compute_values(self, global_stack: Dict[str, np.ndarray]) -> np.ndarray:
        flat = {k: torch.from_numpy(v).to(self.device) for k, v in global_stack.items()}
        v = self.critic(flat).cpu().numpy()
        if self.critic.popart is not None and float(self.critic.popart.debias) > 1e-6:
            v_t = torch.from_numpy(v).to(self.device)
            v = self.critic.popart.denormalize(v_t).cpu().numpy()
        return v.astype(np.float32)

    @torch.no_grad()
    def _compute_values_rise(
        self, global_stack: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RISE-MAPPO: query dual-head critic with GP-uncertainty attention.

        Returns ``(v_mean, v_cvar, agent_sigmas)`` for the buffer; the
        agent_sigmas are stored so the PPO update reproduces the exact
        attention conditioning.
        """
        flat = {k: torch.from_numpy(v).to(self.device) for k, v in global_stack.items()}
        sigmas = self.critic.extract_agent_sigmas(flat)        # (E, A)
        v_mean, v_cvar = self.critic(flat, agent_sigmas=sigmas)
        if self.critic.popart_mean is not None and float(self.critic.popart_mean.debias) > 1e-6:
            v_mean = self.critic.popart_mean.denormalize(v_mean)
        if self.critic.popart_cvar is not None and float(self.critic.popart_cvar.debias) > 1e-6:
            v_cvar = self.critic.popart_cvar.denormalize(v_cvar)
        # NaN-safe: never poison the buffer with non-finite values.
        v_mean = torch.nan_to_num(v_mean, nan=0.0, posinf=0.0, neginf=0.0)
        v_cvar = torch.nan_to_num(v_cvar, nan=0.0, posinf=0.0, neginf=0.0)
        sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            v_mean.cpu().numpy().astype(np.float32),
            v_cvar.cpu().numpy().astype(np.float32),
            sigmas.cpu().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the rollout → update loop for ``n_training_updates`` rounds."""
        set_global_seed(self.runner_cfg.seed)
        for upd in range(1, self.runner_cfg.n_training_updates + 1):
            t0 = time.time()
            roll_stats = self.collect_rollout()
            algo_stats = self.algo.update(self.buffer)
            duration = time.time() - t0
            log_payload = {**roll_stats, **algo_stats, "duration_s": duration, "update": upd}
            if upd % self.runner_cfg.log_interval == 0:
                _LOG.info(
                    "update %d  R=%.3f  cov=%.3f  det=%.2f  pl=%.3f  vl=%.3f  H=%.3f  kl=%.4f  %.1fs",
                    upd,
                    roll_stats["ep_reward_mean"],
                    roll_stats["ep_coverage_mean"],
                    roll_stats["ep_detected_mean"],
                    algo_stats["policy_loss"],
                    algo_stats["value_loss"],
                    algo_stats["entropy"],
                    algo_stats["approx_kl"],
                    duration,
                )
            if self._wandb is not None:
                self._wandb.log(log_payload, step=upd)
            if upd % self.runner_cfg.save_interval == 0:
                self._save_checkpoint(upd)
        self._save_checkpoint(self.runner_cfg.n_training_updates)

    def _save_checkpoint(self, update: int) -> None:
        """Persist actor + critic state under ``save_dir``."""
        path = Path(self.runner_cfg.save_dir) / f"mappo_upd{update}.pt"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "update": update,
            },
            path,
        )
        _LOG.info("saved %s", path)


def build_default_pipeline(
    env_cfg: EnvConfig,
    runner_cfg: RunnerConfig,
    mappo_cfg: MAPPOConfig,
) -> Runner:
    """Construct a runner with default actor + critic shapes from env."""
    probe = MultiRobotSearchEnv(env_cfg)
    actor = Actor(ActorConfig(
        patch_size=probe.patch_size,
        num_actions=probe.num_subgoal_actions,
        num_robots=env_cfg.num_robots,
    ))
    critic = Critic(CriticConfig(
        grid_size=probe.world.grid_size,
        num_robots=env_cfg.num_robots,
        use_popart=mappo_cfg.use_popart,
        use_rise=mappo_cfg.use_rise,
        gp_attention_eta=mappo_cfg.gp_attention_eta,
        world_size=env_cfg.world_size,
    ))
    device = torch.device(runner_cfg.device if torch.cuda.is_available() or runner_cfg.device == "cpu" else "cpu")
    algo = MAPPO(actor=actor, critic=critic, config=mappo_cfg, device=device)
    runner = Runner(
        env_factory=lambda seed: MultiRobotSearchEnv(EnvConfig(**{**env_cfg.__dict__, "seed": seed})),
        actor=actor,
        critic=critic,
        algo=algo,
        runner_cfg=runner_cfg,
    )
    return runner
