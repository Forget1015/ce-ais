"""CE-AIS 预训练管线。

实现两阶段预训练:
1. Contrastive Encoder 预训练（InfoNCE 损失）
2. CE-WM 预训练（NCE 损失，正负样本能量区分）

支持:
- 混合精度训练（AMP）
- YAML 配置文件驱动
- 定期检查点保存与训练中断恢复
- 训练结束后输出能量分布统计报告
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.schema import CEAISConfig
from src.data.data_constructor import DataConstructor
from src.data.perturbation import PerturbationRegistry
from src.training.losses import InfoNCELoss, NCELoss
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Logger


class PretrainPipeline:
    """CE-AIS 预训练管线。

    Args:
        config: CEAISConfig 完整配置对象。
        encoder: ContrastiveEncoder 模型实例。
        ce_wm: CausalEnergyWorldModel 模型实例。
        log_dir: 日志输出目录。
        checkpoint_dir: 检查点保存目录。
    """

    def __init__(
        self,
        config: CEAISConfig,
        encoder: nn.Module,
        ce_wm: nn.Module,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        accelerator=None,
    ):
        self.config = config
        self.encoder = encoder
        self.ce_wm = ce_wm

        if accelerator is None:
            from accelerate import Accelerator
            accelerator = Accelerator()
        self.accelerator = accelerator
        self.device = self.accelerator.device

        # 损失函数
        self.infonce_loss = InfoNCELoss(
            temperature=config.encoder.temperature
        )
        self.nce_loss = NCELoss(temperature=1.0)

        # 日志与检查点
        self.logger = Logger(
            log_dir=log_dir,
            tensorboard=config.logging.tensorboard,
            wandb=config.logging.wandb,
            log_interval=config.logging.log_interval,
        )
        self.encoder_ckpt = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            prefix="encoder",
            max_keep=5,
        )
        self.cewm_ckpt = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            prefix="cewm",
            max_keep=5,
        )

    def pretrain_encoder(
        self,
        dataloader: DataLoader,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Contrastive Encoder 预训练（InfoNCE 损失）。

        Args:
            dataloader: 训练数据加载器，每个批次包含
                {"rgb", "depth", "pose"} 的时序相邻样本对。
            resume: 是否从检查点恢复。

        Returns:
            训练统计信息字典。
        """
        self.encoder.train()

        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        start_epoch = 0
        global_step = 0

        if resume:
            state = self.encoder_ckpt.resume_or_start(
                self.encoder, optimizer, map_location=self.device
            )
            start_epoch = state["epoch"]
            global_step = state["global_step"]

        encoder, optimizer, dataloader = self.accelerator.prepare(
            self.encoder, optimizer, dataloader
        )

        total_epochs = self.config.training.encoder_epochs
        grad_clip_norm = float(getattr(self.config.training, "grad_clip_norm", 0.0) or 0.0)
        losses_history = []

        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            n_batches = 0
            t_start = time.time()

            pbar = tqdm(
                dataloader,
                desc=f"Encoder [{epoch+1}/{total_epochs}]",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )
            for batch in pbar:
                # 移动到设备 (non_blocking 与 pin_memory 配合，重叠传输与计算)
                rgb = batch["rgb"].to(self.device, non_blocking=True)
                depth = batch["depth"].to(self.device, non_blocking=True)
                pose = batch["pose"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with self.accelerator.autocast():
                    if "rgb_pos" in batch:
                        rgb_p = batch["rgb_pos"].to(self.device, non_blocking=True)
                        depth_p = batch["depth_pos"].to(self.device, non_blocking=True)
                        pose_p = batch["pose_pos"].to(self.device, non_blocking=True)
                        # 合并 anchor+positive 做单次前向，避免 BatchNorm in-place 更新冲突
                        rgb_cat = torch.cat([rgb, rgb_p], dim=0)
                        depth_cat = torch.cat([depth, depth_p], dim=0)
                        pose_cat = torch.cat([pose, pose_p], dim=0)
                        z_cat = encoder(rgb_cat, depth_cat, pose_cat)
                        z_anchor, z_positive = z_cat.chunk(2, dim=0)
                    else:
                        z_anchor = encoder(rgb, depth, pose)
                        z_positive = torch.roll(z_anchor, 1, dims=0)

                    loss = self.infonce_loss(z_anchor, z_positive)

                if not torch.isfinite(loss):
                    if self.accelerator.is_main_process:
                        tqdm.write(
                            f"Skipping non-finite encoder batch at epoch={epoch}, "
                            f"step={global_step}: loss={loss.item()}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                self.accelerator.backward(loss)
                if grad_clip_norm > 0:
                    self.accelerator.clip_grad_norm_(encoder.parameters(), grad_clip_norm)
                grads_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in encoder.parameters()
                )
                if not grads_finite:
                    if self.accelerator.is_main_process:
                        tqdm.write(
                            f"Skipping encoder step with non-finite gradients at "
                            f"epoch={epoch}, step={global_step}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                if (global_step % self.config.logging.log_interval == 0
                        and self.accelerator.is_main_process):
                    self.logger.log_training(
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t_start
            losses_history.append(avg_loss)

            if self.accelerator.is_main_process:
                tqdm.write(
                    f"Encoder epoch {epoch+1}/{total_epochs} — "
                    f"loss={avg_loss:.4f}, {elapsed:.1f}s"
                )

            # 检查点保存
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.encoder_ckpt.save(
                        epoch=epoch + 1,
                        model=self.accelerator.unwrap_model(encoder),
                        optimizer=optimizer,
                        global_step=global_step,
                    )

        self.encoder = self.accelerator.unwrap_model(encoder)

        return {
            "final_loss": losses_history[-1] if losses_history else 0.0,
            "losses_history": losses_history,
            "total_epochs": total_epochs,
        }

    def pretrain_cewm(
        self,
        dataloader: DataLoader,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """CE-WM 预训练（NCE 损失）。

        支持两种 batch 格式:
        - 真数据：{"rgb_seq", "depth_seq", "pose_seq", "a_pos"}，
          由冻结的 encoder 在线编码 z_seq，并通过 PerturbationRegistry
          生成 K 个负样本动作。
        - 合成数据：{"z_pos", "a_pos", "a_neg"}，已编码与已扰动。

        Args:
            dataloader: 训练数据加载器。
            resume: 是否从检查点恢复。

        Returns:
            训练统计信息字典，包含能量分布报告。
        """
        self.ce_wm.train()

        # 冻结 encoder 用于在线编码（不用 Accelerate 包装，仅推理）
        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.ce_wm.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        perturbation_strategies = PerturbationRegistry.list_strategies()
        K = int(self.config.training.neg_sample_ratio)

        start_epoch = 0
        global_step = 0

        if resume:
            state = self.cewm_ckpt.resume_or_start(
                self.ce_wm, optimizer, map_location=self.device
            )
            start_epoch = state["epoch"]
            global_step = state["global_step"]

        ce_wm, optimizer, dataloader = self.accelerator.prepare(
            self.ce_wm, optimizer, dataloader
        )

        total_epochs = self.config.training.ce_wm_epochs
        grad_clip_norm = float(getattr(self.config.training, "grad_clip_norm", 0.0) or 0.0)
        losses_history = []
        energy_history: List[Dict[str, float]] = []  # 每 epoch 的能量统计
        energy_pos_accum = []  # 仅最后 epoch，用于最终详细报告
        energy_neg_accum = []

        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            n_batches = 0
            # 当前 epoch 的能量累加（用于绘制 margin 曲线）
            epoch_pos_sum = 0.0
            epoch_neg_sum = 0.0
            epoch_pos_count = 0
            epoch_neg_count = 0

            pbar = tqdm(
                dataloader,
                desc=f"CE-WM [{epoch+1}/{total_epochs}]",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )
            for batch in pbar:
                if "rgb_seq" in batch:
                    # 真数据路径：在线编码 + 在线生成负样本
                    rgb_seq = batch["rgb_seq"].to(self.device, non_blocking=True)
                    depth_seq = batch["depth_seq"].to(self.device, non_blocking=True)
                    pose_seq = batch["pose_seq"].to(self.device, non_blocking=True)
                    a_pos = batch["a_pos"].to(self.device, non_blocking=True)

                    B_, T_ = rgb_seq.shape[0], rgb_seq.shape[1]

                    rgb_flat = rgb_seq.view(B_ * T_, *rgb_seq.shape[2:])
                    depth_flat = depth_seq.view(B_ * T_, *depth_seq.shape[2:])
                    pose_flat = pose_seq.view(B_ * T_, -1)
                    with torch.no_grad():
                        z_flat = self.encoder(rgb_flat, depth_flat, pose_flat)
                    z_pos = z_flat.view(B_, T_, -1)  # [B, T, d_z]

                    # 生成 K 个负样本：按摄动策略轮转
                    a_neg_list = []
                    for k in range(K):
                        strat = perturbation_strategies[
                            k % len(perturbation_strategies)
                        ]
                        a_neg_k = PerturbationRegistry.apply(strat, a_pos)
                        a_neg_list.append(a_neg_k)
                    a_neg = torch.stack(a_neg_list, dim=1)  # [B, K, T, d_a]
                else:
                    # 合成数据回退路径
                    z_pos = batch["z_pos"].to(self.device, non_blocking=True)
                    a_pos = batch["a_pos"].to(self.device, non_blocking=True)
                    a_neg = batch["a_neg"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with self.accelerator.autocast():
                    # 确保序列维度存在
                    if z_pos.dim() == 2:
                        z_pos = z_pos.unsqueeze(1)
                    if a_pos.dim() == 2:
                        a_pos = a_pos.unsqueeze(1)

                    # 正样本能量
                    energy_pos = ce_wm(z_pos, a_pos)  # [B]

                    # 负样本能量: a_neg [B, K, ...] → 逐个计算
                    B, K_batch = a_neg.shape[0], a_neg.shape[1]
                    if a_neg.dim() == 3:
                        # [B, K, d_a] → [B*K, 1, d_a]
                        a_neg_flat = a_neg.view(B * K_batch, 1, -1)
                    else:
                        # [B, K, T, d_a] → [B*K, T, d_a]
                        a_neg_flat = a_neg.view(B * K_batch, *a_neg.shape[2:])

                    z_expanded = z_pos.repeat_interleave(K_batch, dim=0)
                    energy_neg_flat = ce_wm(z_expanded, a_neg_flat)
                    energy_neg = energy_neg_flat.view(B, K_batch)  # [B, K]

                    loss = self.nce_loss(energy_pos, energy_neg)

                if not torch.isfinite(loss):
                    if self.accelerator.is_main_process:
                        tqdm.write(
                            f"Skipping non-finite CE-WM batch at epoch={epoch}, "
                            f"step={global_step}: loss={loss.item()}, "
                            f"energy_pos_finite={torch.isfinite(energy_pos).all().item()}, "
                            f"energy_neg_finite={torch.isfinite(energy_neg).all().item()}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                self.accelerator.backward(loss)
                if grad_clip_norm > 0:
                    self.accelerator.clip_grad_norm_(ce_wm.parameters(), grad_clip_norm)
                grads_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in ce_wm.parameters()
                )
                if not grads_finite:
                    if self.accelerator.is_main_process:
                        tqdm.write(
                            f"Skipping CE-WM step with non-finite gradients at "
                            f"epoch={epoch}, step={global_step}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                # 累积当前 epoch 的能量统计（用于 margin 曲线）
                with torch.no_grad():
                    epoch_pos_sum += energy_pos.detach().float().sum().item()
                    epoch_pos_count += energy_pos.numel()
                    epoch_neg_sum += energy_neg.detach().float().sum().item()
                    epoch_neg_count += energy_neg.numel()

                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                # 记录能量分布（最后一个 epoch，用于详细报告）
                if epoch == total_epochs - 1:
                    energy_pos_accum.append(energy_pos.detach().cpu())
                    energy_neg_accum.append(energy_neg.detach().cpu())

                if (global_step % self.config.logging.log_interval == 0
                        and self.accelerator.is_main_process):
                    self.logger.log_training(
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            losses_history.append(avg_loss)

            # 单 epoch 末：记录能量统计到 Logger（含 TensorBoard）
            if epoch_pos_count > 0 and epoch_neg_count > 0:
                pos_mean = epoch_pos_sum / epoch_pos_count
                neg_mean = epoch_neg_sum / epoch_neg_count
                margin = neg_mean - pos_mean
                energy_history.append(
                    {
                        "epoch": epoch,
                        "pos_mean": pos_mean,
                        "neg_mean": neg_mean,
                        "margin": margin,
                    }
                )
                if self.accelerator.is_main_process:
                    self.logger.log_epoch_energy(
                        epoch=epoch,
                        pos_mean=pos_mean,
                        neg_mean=neg_mean,
                        margin=margin,
                    )
                    tqdm.write(
                        f"CE-WM epoch {epoch+1}/{total_epochs} — "
                        f"loss={avg_loss:.4f}, margin={margin:+.4f}"
                    )
            else:
                if self.accelerator.is_main_process:
                    tqdm.write(
                        f"CE-WM epoch {epoch+1}/{total_epochs} — "
                        f"loss={avg_loss:.4f}"
                    )

            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.cewm_ckpt.save(
                        epoch=epoch + 1,
                        model=self.accelerator.unwrap_model(ce_wm),
                        optimizer=optimizer,
                        global_step=global_step,
                    )

        # 能量分布统计报告
        report = self._compute_energy_report(
            energy_pos_accum, energy_neg_accum
        )

        self.ce_wm = self.accelerator.unwrap_model(ce_wm)

        return {
            "final_loss": losses_history[-1] if losses_history else 0.0,
            "losses_history": losses_history,
            "total_epochs": total_epochs,
            "energy_report": report,
            "energy_history": energy_history,
        }

    def _compute_energy_report(
        self,
        energy_pos_list: list,
        energy_neg_list: list,
    ) -> Dict[str, float]:
        """计算能量分布统计报告。

        Args:
            energy_pos_list: 正样本能量张量列表。
            energy_neg_list: 负样本能量张量列表。

        Returns:
            能量分布报告字典。
        """
        if not energy_pos_list or not energy_neg_list:
            return {
                "pos_energy_mean": 0.0,
                "neg_energy_mean": 0.0,
                "energy_margin": 0.0,
            }

        all_pos = torch.cat(energy_pos_list)
        all_neg = torch.cat(energy_neg_list).view(-1)

        pos_mean = all_pos.mean().item()
        neg_mean = all_neg.mean().item()

        return {
            "pos_energy_mean": pos_mean,
            "neg_energy_mean": neg_mean,
            "energy_margin": neg_mean - pos_mean,
        }
