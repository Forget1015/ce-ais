"""CALVIN 仿真环境 Gym 风格封装。

封装 CALVIN 环境的初始化、重置、步进和观测获取接口，
提供统一的 Gym 风格 API。

支持:
- OOD 干扰注入（物理动力常数变异 + 视觉灾难）
- ABC→D 零样本跨环境多任务链式评估协议
- pybullet 直接操控（mass/friction/lighting）
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger("ce-ais")


@dataclass
class Observation:
    """多模态观测数据。"""

    rgb: torch.Tensor       # [B, 3, H, W] RGB 图像 (float32, [0,1])
    depth: torch.Tensor     # [B, 1, H, W] 深度图
    pose: torch.Tensor      # [B, d_p] 末端执行器姿态


@dataclass
class OODPhysicsParams:
    """物理 OOD 摄动参数。"""

    mass_scale: float = 1.0
    friction_scale: float = 1.0


@dataclass
class OODVisualParams:
    """视觉 OOD 摄动参数。"""

    brightness_jitter: float = 0.0
    gaussian_noise_std: float = 0.0
    camera_pose_offset: float = 0.0


def _create_calvin_env(
    scene: str = "calvin_scene_D",
    cameras: str = "static_and_gripper",
    data_path: str = "data",
    use_egl: bool = True,
    seed: int = 0,
):
    """通过 hydra compose 创建真实 CALVIN 环境。

    Args:
        scene: 场景配置名 (calvin_scene_A/B/C/D)。
        cameras: 相机配置名。
        data_path: CALVIN 资源路径。
        use_egl: 使用 EGL headless 渲染。
        seed: 随机种子。

    Returns:
        PlayTableSimEnv 实例。
    """
    import hydra
    from omegaconf import OmegaConf

    import calvin_env
    calvin_conf_dir = str(Path(calvin_env.__file__).parents[1] / "conf")

    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=calvin_conf_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config_data_collection",
            overrides=[
                f"scene={scene}",
                f"cameras={cameras}",
                f"data_path={data_path}",
                f"seed={seed}",
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
            ],
        )

    env = hydra.utils.instantiate(
        cfg.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
        use_egl=use_egl,
    )
    return env, cfg


class CALVINWrapper:
    """CALVIN 仿真环境 Gym 风格封装。

    提供统一的 reset/step/inject_ood 接口，
    支持 ABC→D 零样本跨环境多任务链式评估协议。

    Args:
        config: 评估配置字典。
        env: 可选的外部环境实例（用于测试注入）。
    """

    def __init__(self, config: dict, env: Any = None):
        self.config = config
        self.protocol = config.get("protocol", "ABC_to_D")
        self.max_chain_length = config.get("max_chain_length", 5)

        # OOD 摄动配置
        ood_cfg = config.get("ood_perturbations", {})
        physics_cfg = ood_cfg.get("physics", {})
        visual_cfg = ood_cfg.get("visual", {})

        self.ood_physics = OODPhysicsParams(
            mass_scale=physics_cfg.get("mass_scale", [0.5, 2.0])[0]
            if isinstance(physics_cfg.get("mass_scale"), list)
            else physics_cfg.get("mass_scale", 1.0),
            friction_scale=physics_cfg.get("friction_scale", [0.3, 1.5])[0]
            if isinstance(physics_cfg.get("friction_scale"), list)
            else physics_cfg.get("friction_scale", 1.0),
        )
        self.ood_visual = OODVisualParams(
            brightness_jitter=visual_cfg.get("brightness_jitter", 0.0),
            gaussian_noise_std=visual_cfg.get("gaussian_noise_std", 0.0),
            camera_pose_offset=visual_cfg.get("camera_pose_offset", 0.0),
        )

        self._env = env
        self._tasks = None
        self._current_task: Optional[str] = None
        self._step_count: int = 0
        self._ood_active: bool = False
        self._start_info: Optional[Dict] = None

        if env is None and config.get("use_real_env", False):
            self._init_real_env()

    def _init_real_env(self):
        """初始化真实 CALVIN 仿真环境。"""
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        scene = self.config.get("scene", "calvin_scene_D")
        cameras = self.config.get("cameras", "static_and_gripper")
        data_path = self.config.get("data_path", "data")
        seed = self.config.get("seed", 0)
        use_egl = self.config.get("use_egl", True)

        logger.info("Initializing CALVIN env: scene=%s, cameras=%s, egl=%s", scene, cameras, use_egl)
        self._env, cfg = _create_calvin_env(
            scene=scene, cameras=cameras, data_path=data_path,
            use_egl=use_egl, seed=seed,
        )

        # 初始化任务检测 oracle
        import hydra as _hydra
        tasks_cfg = cfg.get("tasks", None)
        if tasks_cfg is not None:
            self._tasks = _hydra.utils.instantiate(tasks_cfg)
            logger.info("Task oracle loaded: %d tasks", self._tasks.num_tasks)

        logger.info("CALVIN env initialized (cid=%d)", self._env.cid)

    def reset(self, task: Optional[str] = None,
              robot_obs: Optional[np.ndarray] = None,
              scene_obs: Optional[np.ndarray] = None) -> Observation:
        """重置环境，返回初始观测。

        Args:
            task: 可选的任务名称。
            robot_obs: 可选的机器人状态（用于重放特定状态）。
            scene_obs: 可选的场景状态。

        Returns:
            初始观测 Observation。
        """
        self._current_task = task
        self._step_count = 0

        if self._env is not None:
            raw_obs = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            self._start_info = self._env.get_info()
            return self._wrap_observation(raw_obs)

        return self._make_placeholder_observation()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """执行动作，返回 (observation, reward, done, info)。

        Args:
            action: 动作张量 [d_a] 或 [B, d_a]。

        Returns:
            observation, reward, done, info。
        """
        self._step_count += 1

        if self._env is not None:
            action_np = action.detach().cpu().numpy().flatten()[:7].copy()
            # CALVIN 要求 gripper action 为 -1 (close) 或 1 (open)
            action_np[-1] = 1.0 if action_np[-1] >= 0 else -1.0
            raw_obs, _, done, info = self._env.step(action_np)
            obs = self._wrap_observation(raw_obs)

            if self._ood_active:
                obs = self._apply_visual_ood(obs)

            # 任务成功检测
            reward = 0.0
            if self._tasks is not None and self._current_task and self._start_info:
                end_info = self._env.get_info()
                achieved = self._tasks.get_task_info_for_set(
                    self._start_info, end_info, {self._current_task}
                )
                if self._current_task in achieved:
                    reward = 1.0
                    done = True

            return obs, reward, done, info

        obs = self._make_placeholder_observation()
        return obs, 0.0, False, {"step": self._step_count}

    def inject_ood(
        self, perturbation_type: str, **kwargs: Any
    ) -> None:
        """注入 OOD 干扰。

        支持通过 pybullet 直接操控物理参数（mass/friction）和视觉参数。

        Args:
            perturbation_type: 摄动类型，"physics" / "visual" / "camera"。
            **kwargs: 摄动参数。
        """
        self._ood_active = True

        if perturbation_type == "physics":
            mass_scale = kwargs.get("mass_scale", self.ood_physics.mass_scale)
            friction_scale = kwargs.get("friction_scale", self.ood_physics.friction_scale)
            self.ood_physics.mass_scale = mass_scale
            self.ood_physics.friction_scale = friction_scale

            if self._env is not None:
                self._apply_physics_ood(mass_scale, friction_scale)
            logger.info(
                "Injected physics OOD: mass=%.2f, friction=%.2f",
                mass_scale, friction_scale,
            )

        elif perturbation_type == "visual":
            self.ood_visual.brightness_jitter = kwargs.get(
                "brightness", self.ood_visual.brightness_jitter
            )
            self.ood_visual.gaussian_noise_std = kwargs.get(
                "noise_std", self.ood_visual.gaussian_noise_std
            )
            logger.info(
                "Injected visual OOD: brightness=%.2f, noise=%.3f",
                self.ood_visual.brightness_jitter,
                self.ood_visual.gaussian_noise_std,
            )

        elif perturbation_type == "camera":
            offset = kwargs.get("camera_offset", self.ood_visual.camera_pose_offset)
            self.ood_visual.camera_pose_offset = offset
            if self._env is not None:
                self._apply_camera_ood(offset)
            logger.info("Injected camera OOD: offset=%.3f", offset)

        else:
            raise ValueError(
                f"Unknown perturbation type: {perturbation_type}. "
                "Expected 'physics', 'visual', or 'camera'."
            )

    def _apply_physics_ood(self, mass_scale: float, friction_scale: float):
        """通过 pybullet.changeDynamics 修改物理参数。"""
        import pybullet as p

        cid = self._env.cid
        num_bodies = p.getNumBodies(physicsClientId=cid)

        for body_id in range(num_bodies):
            num_joints = p.getNumJoints(body_id, physicsClientId=cid)
            # base link
            dyn = p.getDynamicsInfo(body_id, -1, physicsClientId=cid)
            if dyn[0] > 0:  # mass > 0 (non-static)
                p.changeDynamics(
                    body_id, -1,
                    mass=dyn[0] * mass_scale,
                    lateralFriction=dyn[1] * friction_scale,
                    physicsClientId=cid,
                )
            # child links
            for joint_id in range(num_joints):
                dyn = p.getDynamicsInfo(body_id, joint_id, physicsClientId=cid)
                if dyn[0] > 0:
                    p.changeDynamics(
                        body_id, joint_id,
                        mass=dyn[0] * mass_scale,
                        lateralFriction=dyn[1] * friction_scale,
                        physicsClientId=cid,
                    )

        logger.info("Applied physics OOD to %d bodies", num_bodies)

    def _apply_camera_ood(self, offset: float):
        """偏移静态相机的位置。"""
        if not hasattr(self._env, "cameras"):
            return
        for cam in self._env.cameras:
            if hasattr(cam, "look_from"):
                cam.look_from = [c + np.random.uniform(-offset, offset) for c in cam.look_from]
                logger.info("Camera '%s' offset applied: %.3f", getattr(cam, "name", "?"), offset)

    def check_task_success(self, task_name: str) -> bool:
        """手动检查当前任务是否成功。"""
        if self._tasks is None or self._start_info is None or self._env is None:
            return False
        end_info = self._env.get_info()
        achieved = self._tasks.get_task_info_for_set(
            self._start_info, end_info, {task_name}
        )
        return task_name in achieved

    def run_chain_evaluation(
        self,
        policy_fn: Callable,
        task_chain: List[str],
        max_steps_per_task: int = 300,
    ) -> Dict[str, Any]:
        """运行 ABC→D 多任务链式评估。

        Args:
            policy_fn: 策略函数 (observation_dict, instruction) -> action。
            task_chain: 任务名称列表（按顺序执行）。
            max_steps_per_task: 每个任务的最大步数。

        Returns:
            评估结果字典。
        """
        chain_length = min(len(task_chain), self.max_chain_length)
        task_results: List[Dict[str, Any]] = []
        chain_success = True

        obs = self.reset(task=task_chain[0] if task_chain else None)

        for i, task_name in enumerate(task_chain[:chain_length]):
            self._current_task = task_name
            # 记录任务开始时的 scene info
            if self._env is not None:
                self._start_info = self._env.get_info()

            task_success = False
            task_steps = 0
            start_time = time.time()

            for step_idx in range(max_steps_per_task):
                obs_dict = {
                    "rgb": obs.rgb,
                    "depth": obs.depth,
                    "pose": obs.pose,
                }
                action = policy_fn(obs_dict, task_name)
                obs, reward, done, info = self.step(action)
                task_steps += 1

                if reward > 0:
                    task_success = True
                    break

            elapsed = time.time() - start_time
            task_results.append(
                {
                    "task": task_name,
                    "success": task_success,
                    "steps": task_steps,
                    "time_s": elapsed,
                }
            )

            if not task_success:
                chain_success = False
                break

        return {
            "protocol": self.protocol,
            "chain_length": chain_length,
            "chain_success": chain_success,
            "completed_tasks": sum(1 for r in task_results if r["success"]),
            "task_results": task_results,
        }

    def _wrap_observation(self, raw_obs: Any) -> Observation:
        """将 CALVIN 原始观测转换为 Observation。

        CALVIN get_obs() 返回:
            rgb_obs: {rgb_static: [H,W,3] uint8, rgb_gripper: [H,W,3] uint8}
            depth_obs: {depth_static: [H,W] float32, depth_gripper: ...}
            robot_obs: ndarray (15,)
            scene_obs: ndarray (24,)
        """
        if not isinstance(raw_obs, dict):
            return self._make_placeholder_observation()

        # RGB: 优先 rgb_static
        rgb = None
        if "rgb_obs" in raw_obs and isinstance(raw_obs["rgb_obs"], dict):
            rgb_raw = raw_obs["rgb_obs"].get("rgb_static")
            if rgb_raw is not None:
                if isinstance(rgb_raw, np.ndarray):
                    rgb_raw = rgb_raw.astype(np.float32) / 255.0
                    rgb = torch.from_numpy(rgb_raw).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
                elif isinstance(rgb_raw, torch.Tensor):
                    rgb = rgb_raw.float()
                    if rgb.max() > 1.0:
                        rgb = rgb / 255.0
                    if rgb.dim() == 3:
                        rgb = rgb.unsqueeze(0)
        elif "rgb" in raw_obs or "rgb_obs" in raw_obs:
            v = raw_obs.get("rgb", raw_obs.get("rgb_obs"))
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if isinstance(v, torch.Tensor):
                if v.max() > 1.0:
                    v = v / 255.0
                if v.dim() == 3:
                    v = v.unsqueeze(0)
                rgb = v

        # Depth: 优先 depth_static
        depth = None
        if "depth_obs" in raw_obs and isinstance(raw_obs["depth_obs"], dict):
            d_raw = raw_obs["depth_obs"].get("depth_static")
            if d_raw is not None:
                if isinstance(d_raw, np.ndarray):
                    d_raw = d_raw.astype(np.float32)
                    depth = torch.from_numpy(d_raw).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                elif isinstance(d_raw, torch.Tensor):
                    depth = d_raw.float()
                    if depth.dim() == 2:
                        depth = depth.unsqueeze(0).unsqueeze(0)
                    elif depth.dim() == 3:
                        depth = depth.unsqueeze(0)
        elif "depth" in raw_obs or "depth_obs" in raw_obs:
            v = raw_obs.get("depth", raw_obs.get("depth_obs"))
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if isinstance(v, torch.Tensor):
                if v.dim() == 2:
                    v = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 3:
                    v = v.unsqueeze(0)
                depth = v

        # Pose: robot_obs 前 7 维 (tcp_pos[3] + tcp_orn[4] 或 tcp_pos[3] + tcp_orn[3] + gripper)
        pose = None
        robot_obs = raw_obs.get("robot_obs")
        if robot_obs is not None:
            if isinstance(robot_obs, np.ndarray):
                pose = torch.from_numpy(robot_obs[:7].astype(np.float32)).unsqueeze(0)  # [1, 7]
            elif isinstance(robot_obs, torch.Tensor):
                pose = robot_obs[:7].float().unsqueeze(0)

        return Observation(
            rgb=rgb if rgb is not None else torch.zeros(1, 3, 200, 200),
            depth=depth if depth is not None else torch.zeros(1, 1, 200, 200),
            pose=pose if pose is not None else torch.zeros(1, 7),
        )

    def _make_placeholder_observation(self) -> Observation:
        """生成占位观测数据（无真实环境时使用）。"""
        return Observation(
            rgb=torch.zeros(1, 3, 200, 200),
            depth=torch.zeros(1, 1, 200, 200),
            pose=torch.zeros(1, 7),
        )

    def _apply_visual_ood(self, obs: Observation) -> Observation:
        """对观测应用视觉 OOD 摄动。"""
        rgb = obs.rgb.clone()

        if self.ood_visual.brightness_jitter > 0:
            jitter = torch.randn(1).item() * self.ood_visual.brightness_jitter
            rgb = torch.clamp(rgb + jitter, 0.0, 1.0)

        if self.ood_visual.gaussian_noise_std > 0:
            noise = torch.randn_like(rgb) * self.ood_visual.gaussian_noise_std
            rgb = torch.clamp(rgb + noise, 0.0, 1.0)

        return Observation(rgb=rgb, depth=obs.depth, pose=obs.pose)

    @property
    def available_tasks(self) -> List[str]:
        """列出所有可用任务名。"""
        if self._tasks is not None:
            return list(self._tasks.tasks.keys())
        return []

    def close(self):
        """释放环境资源。"""
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
