#!/usr/bin/env python3
"""CE-AIS 论文主实验脚本。

运行 CE-AIS vs 4 个 baseline 在 ABC→D 协议下的完整对比实验。

输出: results/main_experiment.json

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/run_paper_experiments.py \
        --data-dir data/task_ABC_D --n-chains 200
"""

import argparse
import ast
from copy import deepcopy
from itertools import product
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="CE-AIS Paper Main Experiments")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "task_ABC_D"))
    parser.add_argument("--n-chains", type=int, default=200)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy", "calvin"])
    parser.add_argument("--no-egl", action="store_true",
                        help="Disable PyBullet EGL plugin and use DIRECT/TinyRenderer")
    parser.add_argument("--device", type=str, default=None,
                        help="Evaluation device, e.g. cuda:0 or cpu. Defaults to project.device")
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Encoder checkpoint path. Defaults to latest checkpoints/encoder_epoch*.pt")
    parser.add_argument("--cewm-ckpt", type=str, default=None,
                        help="CE-WM checkpoint path. Defaults to latest checkpoints/cewm_epoch*.pt")
    parser.add_argument("--progress-steps", type=int, default=25,
                        help="Print per-task progress every N env steps; use 0 to print only task boundaries")
    parser.add_argument("--sequence-source", type=str, default="official", choices=["official", "random"],
                        help="Use official CALVIN reachable sequences or random task chains")
    parser.add_argument("--calvin-policy-ckpt", type=str, default=None,
                        help="CALVIN-native policy checkpoint for --vla-type calvin")
    parser.add_argument("--calvin-train-folder", type=str, default=None,
                        help="CALVIN policy training folder containing .hydra/config.yaml")
    parser.add_argument("--calvin-dataset-path", type=str, default=None,
                        help="Dataset path for official CALVIN policy loading. Defaults to --data-dir")
    parser.add_argument("--run-expert-replay-diagnostic", action="store_true",
                        help="Replay validation rel_actions to sanity-check env/oracle/action plumbing")
    parser.add_argument("--expert-replay-episodes", type=int, default=5,
                        help="Number of validation language segments to replay for diagnostics")
    parser.add_argument("--methods", type=str, nargs="*",
                        default=["ce_ais", "frozen_openvla", "pdf", "tt_vla", "ada_world_policy"])
    return parser.parse_args()


def load_validation_annotations(data_dir: str):
    ann_path = os.path.join(data_dir, "validation", "lang_annotations", "auto_lang_ann.npy")
    if os.path.exists(ann_path):
        return np.load(ann_path, allow_pickle=True).item()
    return None


def load_frame_state(data_dir: str, frame_id: int):
    frame_path = os.path.join(data_dir, "validation", f"episode_{frame_id:07d}.npz")
    if not os.path.exists(frame_path):
        return None, None
    data = np.load(frame_path)
    return data["robot_obs"].astype(np.float64), data["scene_obs"].astype(np.float64)


def _calvin_root() -> Path:
    import calvin_env
    return Path(calvin_env.__file__).resolve().parents[2]


def load_calvin_language_map() -> dict:
    import yaml
    ann_path = _calvin_root() / "calvin_models" / "conf" / "annotations" / "new_playtable_validation.yaml"
    annotations = yaml.safe_load(ann_path.read_text())
    return {task: texts[0] for task, texts in annotations.items()}


def load_calvin_sequence_metadata():
    seq_path = _calvin_root() / "calvin_models" / "calvin_agent" / "evaluation" / "multistep_sequences.py"
    tree = ast.parse(seq_path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"tasks", "task_categories"}:
                values[target.id] = ast.literal_eval(node.value)
    return values["tasks"], values["task_categories"]


def check_condition(state, condition):
    for key, value in condition.items():
        current = state[key]
        if isinstance(value, list):
            if current not in value:
                return False
        elif current != value:
            return False
    return True


def valid_next_states(state, task_rules):
    next_states = []
    for rule in task_rules:
        if check_condition(state, rule["condition"]):
            next_state = deepcopy(state)
            next_state.update(rule["effect"])
            next_states.append(next_state)
    return next_states


def get_env_state_for_initial_condition(initial_condition):
    from math import pi

    robot_obs = np.array([
        0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596,
        1.50013585, 0.07999963, -1.21779124, 1.03987629, 2.11978254,
        -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0,
    ])
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]

    seed = zlib.crc32(str(tuple(initial_condition.values())).encode("utf-8"))
    state = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(block_table)

    scene_obs = np.zeros(24)
    if initial_condition["slider"] == "left":
        scene_obs[0] = 0.28
    if initial_condition["drawer"] == "open":
        scene_obs[1] = 0.22
    if initial_condition["lightbulb"] == 1:
        scene_obs[3] = 0.088
    scene_obs[4] = initial_condition["lightbulb"]
    scene_obs[5] = initial_condition["led"]

    for offset, color in [(6, "red_block"), (12, "blue_block"), (18, "pink_block")]:
        loc = initial_condition[color]
        if loc == "slider_right":
            scene_obs[offset:offset + 3] = block_slider_right
        elif loc == "slider_left":
            scene_obs[offset:offset + 3] = block_slider_left
        elif color == "blue_block" and initial_condition["red_block"] != "table":
            scene_obs[offset:offset + 3] = block_table[0]
        else:
            scene_obs[offset:offset + 3] = block_table[1 if color == "pink_block" else 0]
        scene_obs[offset + 5] = np.random.uniform(*block_rot_z_range)

    np.random.set_state(state)
    return robot_obs, scene_obs


def sample_official_eval_specs(n_chains, chain_length, rng):
    task_rules, task_categories = load_calvin_sequence_metadata()
    language_map = load_calvin_language_map()
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    all_states = []
    for values in product(*possible_conditions.values()):
        block_locs = values[4:7]
        if block_locs.count("table") in [1, 2] and block_locs.count("slider_right") < 2 and block_locs.count("slider_left") < 2:
            all_states.append(dict(zip(possible_conditions.keys(), values)))

    specs = []
    task_names = list(task_rules.keys())
    while len(specs) < n_chains:
        initial_state = deepcopy(all_states[rng.randint(len(all_states))])
        state = deepcopy(initial_state)
        tasks = []
        used_categories = set()
        for _ in range(chain_length):
            candidates = []
            for task_name in task_names:
                category = task_categories[task_name]
                if category in used_categories:
                    continue
                next_states = valid_next_states(state, task_rules[task_name])
                if next_states:
                    candidates.append((task_name, next_states))
            if not candidates:
                break
            task_name, next_states = candidates[rng.randint(len(candidates))]
            state = deepcopy(next_states[rng.randint(len(next_states))])
            tasks.append(task_name)
            used_categories.add(task_categories[task_name])
        if len(tasks) == chain_length:
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            specs.append({
                "initial_state": initial_state,
                "robot_obs": robot_obs,
                "scene_obs": scene_obs,
                "tasks": tasks,
                "instructions": [language_map.get(task, task.replace("_", " ")) for task in tasks],
            })
    return specs


def sample_task_chains(available_tasks, n_chains, chain_length, rng):
    chains = []
    task_list = list(available_tasks)
    for _ in range(n_chains):
        chain = []
        for _ in range(chain_length):
            candidates = [t for t in task_list if not chain or t != chain[-1]]
            chain.append(rng.choice(candidates))
        chains.append(chain)
    return chains


def make_random_eval_specs(available_tasks, n_chains, chain_length, rng):
    return [{"tasks": chain, "instructions": [str(t).replace("_", " ") for t in chain]}
            for chain in sample_task_chains(available_tasks, n_chains, chain_length, rng)]


def run_expert_replay_diagnostic(wrapper, data_dir: str, n_episodes: int):
    annotations = load_validation_annotations(data_dir)
    if annotations is None:
        raise FileNotFoundError(
            os.path.join(data_dir, "validation", "lang_annotations", "auto_lang_ann.npy")
        )

    tasks = annotations["language"]["task"]
    anns = annotations["language"]["ann"]
    indices = annotations["info"]["indx"]
    total = min(n_episodes, len(indices))
    per_task = {}
    episodes = []

    for i in range(total):
        start, end = [int(x) for x in indices[i]]
        task = str(tasks[i])
        instruction = str(anns[i])
        start_path = os.path.join(data_dir, "validation", f"episode_{start:07d}.npz")
        if not os.path.exists(start_path):
            episodes.append({"task": task, "start": start, "end": end, "success": False, "error": "missing_start_frame"})
            continue

        start_data = np.load(start_path)
        obs = wrapper.reset(
            task=task,
            robot_obs=start_data["robot_obs"].astype(np.float64),
            scene_obs=start_data["scene_obs"].astype(np.float64),
        )
        success = False
        steps = 0
        error = None
        for frame_id in range(start, end + 1):
            frame_path = os.path.join(data_dir, "validation", f"episode_{frame_id:07d}.npz")
            if not os.path.exists(frame_path):
                error = f"missing_frame_{frame_id}"
                break
            frame = np.load(frame_path)
            action = torch.from_numpy(frame["rel_actions"].astype(np.float32))
            obs, reward, done, info = wrapper.step(action)
            steps += 1
            if reward > 0:
                success = True
                break

        bucket = per_task.setdefault(task, {"success": 0, "total": 0})
        bucket["total"] += 1
        bucket["success"] += int(success)
        episodes.append({
            "task": task,
            "instruction": instruction,
            "start": start,
            "end": end,
            "steps": steps,
            "success": success,
            "error": error,
        })
        print(f"  [expert_replay] {i + 1}/{total} {task}: {'success' if success else 'failed'} ({steps} steps)", flush=True)

    successes = sum(int(e["success"]) for e in episodes)
    return {
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "successes": successes,
        "total": len(episodes),
        "per_task": per_task,
        "episodes": episodes,
    }


def build_ce_ais_policy(config_dict, vla_config, device, encoder_ckpt=None, cewm_ckpt=None):
    """构建完整 CE-AIS 推理拓扑。"""
    from src.config.config_manager import ConfigManager
    from src.config.schema import (CEAISConfig, EncoderConfig, CEWMConfig,
                                    SteeringConfig, BilateralGatingConfig)
    from src.dual_stream.vla_adapter import build_vla_adapter
    from src.dual_stream.topology import DualStreamTopology
    from src.encoders.contrastive_encoder import ContrastiveEncoder
    from src.world_model.ce_wm import CausalEnergyWorldModel
    from src.steering.efe_steering import EFESteering
    from src.steering.bilateral_gating import BilateralGating

    enc_cfg = config_dict.get("encoder", {})
    cewm_cfg = config_dict.get("ce_wm", {})
    steer_cfg = config_dict.get("steering", {})
    gate_cfg = config_dict.get("bilateral_gating", {})

    encoder_config = EncoderConfig(**{k: v for k, v in enc_cfg.items()
                                      if k in EncoderConfig.__dataclass_fields__})
    cewm_config = CEWMConfig(**{k: v for k, v in cewm_cfg.items()
                                 if k in CEWMConfig.__dataclass_fields__})
    steering_config = SteeringConfig(**{k: v for k, v in steer_cfg.items()
                                        if k in SteeringConfig.__dataclass_fields__})
    gating_config = BilateralGatingConfig(**{k: v for k, v in gate_cfg.items()
                                             if k in BilateralGatingConfig.__dataclass_fields__})

    vla = build_vla_adapter(vla_config)
    encoder = ContrastiveEncoder(encoder_config).to(device).eval()
    ce_wm = CausalEnergyWorldModel(cewm_config).to(device).eval()
    steering = EFESteering(steering_config)
    gating = BilateralGating(gating_config)

    def resolve_ckpt(path):
        if path is None:
            return None
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    from src.utils.checkpoint import CheckpointManager

    enc_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="encoder")
    enc_ckpt = resolve_ckpt(encoder_ckpt) if encoder_ckpt else enc_mgr.find_latest()
    if enc_ckpt:
        enc_mgr.load(filepath=enc_ckpt, model=encoder, map_location=device)
        print(f"  Loaded encoder checkpoint: {enc_ckpt}")

    cewm_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="cewm")
    cewm_ckpt = resolve_ckpt(cewm_ckpt) if cewm_ckpt else cewm_mgr.find_latest()
    if cewm_ckpt:
        cewm_mgr.load(filepath=cewm_ckpt, model=ce_wm, map_location=device)
        print(f"  Loaded CE-WM checkpoint: {cewm_ckpt}")

    topology = DualStreamTopology(
        vla_adapter=vla, encoder=encoder, ce_wm=ce_wm,
        steering=steering, gating=gating,
        mc_samples=gate_cfg.get("mc_samples", 5),
    )
    return topology


def evaluate_method(method_name, policy_fn, wrapper, eval_specs, data_dir, args, rng, policy_reset_fn=None):
    """对单个方法运行完整评估。"""
    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}
    all_results = []
    latencies = []

    for ci, spec in enumerate(eval_specs):
        chain = spec["tasks"]
        instructions = spec.get("instructions")
        robot_obs = spec.get("robot_obs")
        scene_obs = spec.get("scene_obs")

        print(f"    [{method_name}] chain {ci + 1}/{len(eval_specs)} start: {chain}", flush=True)
        if instructions:
            print(f"    [{method_name}] chain {ci + 1}/{len(eval_specs)} instructions: {instructions}", flush=True)

        def progress(message, _ci=ci):
            print(f"    [{method_name}] chain {_ci + 1}/{len(eval_specs)} {message}", flush=True)

        t_start = time.time()
        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn,
            task_chain=chain,
            max_steps_per_task=args.max_steps,
            progress_fn=progress,
            progress_steps=args.progress_steps,
            initial_robot_obs=robot_obs,
            initial_scene_obs=scene_obs,
            instructions=instructions,
            policy_reset_fn=policy_reset_fn,
        )
        latencies.append((time.time() - t_start) * 1000 / max(sum(
            r["steps"] for r in result["task_results"]), 1))

        all_results.append(result)
        completed = result["completed_tasks"]
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

        if (ci + 1) % 10 == 0:
            avg = sum(chain_successes[1]) / len(chain_successes[1]) * 100
            print(f"    [{method_name}] {ci+1}/{len(eval_specs)} chains, "
                  f"L1 success: {avg:.1f}%")

    results_summary = {}
    for length in range(1, args.chain_length + 1):
        s = chain_successes[length]
        results_summary[f"chain_{length}"] = sum(s) / len(s) if s else 0.0

    results_summary["avg_completed_tasks"] = float(
        np.mean([r["completed_tasks"] for r in all_results]))
    results_summary["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0

    return results_summary


def main():
    args = parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.device and str(args.device).startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        hidden = os.environ.pop("CUDA_VISIBLE_DEVICES")
        print(
            f"[INFO] Ignoring CUDA_VISIBLE_DEVICES={hidden}; "
            f"--device {args.device} selects the physical GPU index."
        )
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    from src.config.config_manager import ConfigManager
    cm = ConfigManager(config_path=args.config)
    config_dict = cm.config

    requested_device = str(args.device or config_dict.get("project", {}).get("device", "cuda:0"))
    device_index = None
    if not torch.cuda.is_available():
        device = "cpu"
    elif requested_device.startswith("cuda"):
        try:
            device_index = int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested_device}") from exc
        if device_index >= torch.cuda.device_count():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            raise ValueError(
                f"Requested {requested_device}, but PyTorch sees only {torch.cuda.device_count()} CUDA device(s). "
                f"CUDA_VISIBLE_DEVICES={visible}. Unset CUDA_VISIBLE_DEVICES if you want --device cuda:N to mean physical GPU N."
            )
        device = requested_device
        torch.cuda.set_device(device)
    else:
        device = requested_device

    if not args.no_egl and device_index is not None:
        os.environ["EGL_VISIBLE_DEVICES"] = str(device_index)

    print("=" * 70)
    print("CE-AIS Paper Main Experiments")
    print(f"Methods: {args.methods}")
    print(f"Chains: {args.n_chains} x {args.chain_length}")
    print(f"Device: {device} (requested={requested_device}, visible={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"EGL: {'disabled' if args.no_egl else 'enabled'} (EGL_VISIBLE_DEVICES={os.environ.get('EGL_VISIBLE_DEVICES')})")
    print("=" * 70)

    # 初始化 CALVIN 环境
    print("\n[1] Initializing CALVIN environment...")
    from src.evaluation.calvin_integration import CALVINWrapper

    env_config = {
        "use_real_env": True,
        "scene": "calvin_scene_D",
        "cameras": "static_and_gripper",
        "use_egl": not args.no_egl,
        "seed": args.seed,
        "max_chain_length": args.chain_length,
    }
    wrapper = CALVINWrapper(config=env_config)
    print(f"  Available tasks: {len(wrapper.available_tasks)}")

    if args.sequence_source == "official":
        eval_specs = sample_official_eval_specs(args.n_chains, args.chain_length, rng)
    else:
        eval_specs = make_random_eval_specs(wrapper.available_tasks, args.n_chains, args.chain_length, rng)
    print(f"  Sequence source: {args.sequence_source}")

    diagnostics = {}
    if args.run_expert_replay_diagnostic:
        print("\n[Diagnostic] Running expert rel_actions replay...")
        diagnostics["expert_replay"] = run_expert_replay_diagnostic(
            wrapper, args.data_dir, args.expert_replay_episodes
        )
        diag = diagnostics["expert_replay"]
        print(f"  Expert replay: {diag['successes']}/{diag['total']} ({diag['success_rate']:.1%})")

    # VLA 共享配置
    vla_config = {
        "type": args.vla_type,
        "device": device,
        "action_dim": 7,
        "chunk_size": 1,
        "calvin_policy_ckpt": args.calvin_policy_ckpt,
        "calvin_train_folder": args.calvin_train_folder,
        "calvin_dataset_path": args.calvin_dataset_path or args.data_dir,
    }

    all_method_results = {}

    for method_name in args.methods:
        print(f"\n[Eval] Running: {method_name}")
        method_rng = np.random.RandomState(args.seed)

        if method_name == "ce_ais":
            topology = build_ce_ais_policy(
                config_dict, vla_config, device, args.encoder_ckpt, args.cewm_ckpt
            )

            def ce_ais_policy(obs_dict, instruction):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                action, info = topology.safe_step(obs_on_device, str(instruction))
                return action.squeeze(1).cpu()

            results = evaluate_method(
                method_name, ce_ais_policy, wrapper, eval_specs, args.data_dir, args, method_rng,
                policy_reset_fn=getattr(topology, "reset", None))

        elif method_name == "frozen_openvla":
            from src.evaluation.baseline_framework import FrozenOpenVLABaseline
            bl = FrozenOpenVLABaseline({**vla_config, "vla_type": args.vla_type})
            bl.setup()

            def frozen_policy(obs_dict, instruction):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                return bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

            results = evaluate_method(
                method_name, frozen_policy, wrapper, eval_specs, args.data_dir, args, method_rng,
                policy_reset_fn=getattr(bl, "reset_task", None))
            bl.teardown()

        else:
            from src.evaluation.baseline_framework import BASELINE_REGISTRY
            bl_cls = BASELINE_REGISTRY[method_name]
            bl = bl_cls({**vla_config, "vla_type": args.vla_type})
            bl.setup()

            def baseline_policy(obs_dict, instruction, _bl=bl):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                return _bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

            results = evaluate_method(
                method_name, baseline_policy, wrapper, eval_specs, args.data_dir, args, method_rng,
                policy_reset_fn=getattr(bl, "reset_task", None))
            bl.teardown()

        all_method_results[method_name] = results
        print(f"  [{method_name}] L1={results['chain_1']:.1%}, "
              f"L3={results.get('chain_3', 0):.1%}, "
              f"L5={results.get('chain_5', 0):.1%}, "
              f"latency={results['avg_latency_ms']:.1f}ms")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "n_chains": args.n_chains,
            "chain_length": args.chain_length,
            "max_steps_per_task": args.max_steps,
            "seed": args.seed,
            "vla_type": args.vla_type,
            "device": device,
            "requested_device": requested_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_visible_devices": os.environ.get("EGL_VISIBLE_DEVICES"),
            "use_egl": not args.no_egl,
            "methods": args.methods,
            "sequence_source": args.sequence_source,
            "encoder_ckpt": args.encoder_ckpt,
            "cewm_ckpt": args.cewm_ckpt,
            "calvin_policy_ckpt": args.calvin_policy_ckpt,
            "calvin_train_folder": args.calvin_train_folder,
            "calvin_dataset_path": args.calvin_dataset_path or args.data_dir,
        },
        "results": all_method_results,
        "diagnostics": diagnostics,
    }
    output_path = os.path.join(args.output_dir, "main_experiment.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # 打印对比表
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5':>8} {'Lat(ms)':>8}")
    print("-" * 70)
    for name, res in all_method_results.items():
        print(f"{name:<20} {res.get('chain_1',0):>7.1%} {res.get('chain_2',0):>7.1%} "
              f"{res.get('chain_3',0):>7.1%} {res.get('chain_4',0):>7.1%} "
              f"{res.get('chain_5',0):>7.1%} {res.get('avg_latency_ms',0):>7.1f}")
    print("=" * 70)

    wrapper.close()


if __name__ == "__main__":
    main()
