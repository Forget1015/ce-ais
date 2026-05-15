# CE-AIS 项目快捷命令
# 用法: make <target>

PYTHON := uv run python
PYTEST := uv run pytest
GPU    ?= 4

# triton/mamba-ssm 编译需要 python3.12 头文件（系统无 python3.12-dev）
PYTHON_INCLUDE := /data0/yejinxuan/miniconda3/envs/py312_pb/include/python3.12
export CPATH := $(PYTHON_INCLUDE):$(CPATH)

# ============================================================
# 环境
# ============================================================

.PHONY: sync
sync:
	uv sync --extra mamba --extra vla

# ============================================================
# 测试
# ============================================================

.PHONY: test
test:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTEST) tests/ -q --tb=short

.PHONY: test-unit
test-unit:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTEST) tests/unit/ -q --tb=short

.PHONY: test-integration
test-integration:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTEST) tests/integration/ -v --tb=short

.PHONY: test-property
test-property:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTEST) tests/property/ -v --tb=short

# ============================================================
# Week 3 端到端验证（IMPLEMENTATION_PLAN.md §4）
# ============================================================

.PHONY: verify-wm
verify-wm:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -c "\
from src.world_model.ce_wm import CausalEnergyWorldModel; \
from src.config.config_manager import ConfigManager; \
from src.config.schema import CEWMConfig; \
import dataclasses, torch, time; \
cfg = ConfigManager('configs/base.yaml').config; \
c = cfg.get('ce_wm', cfg); \
vk = {f.name for f in dataclasses.fields(CEWMConfig)}; \
c = CEWMConfig(**{k: v for k, v in c.items() if k in vk}); \
m = CausalEnergyWorldModel(c).cuda().half(); \
n = sum(p.numel() for p in m.parameters()); \
print(f'params: {n/1e6:.2f}M'); \
z = torch.randn(1,10,128).cuda().half(); a = torch.randn(1,10,7).cuda().half(); \
_ = m(z,a); torch.cuda.synchronize(); \
t = time.time(); \
[m(z,a) for _ in range(100)]; torch.cuda.synchronize(); \
print(f'avg forward: {(time.time()-t)*10:.2f}ms')"

# ============================================================
# 训练
# ============================================================

.PHONY: pretrain
pretrain:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/pretrain.py --config configs/base.yaml

.PHONY: evaluate
evaluate:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/evaluate.py \
		--config configs/base.yaml \
		--baseline frozen_openvla \
		--task_chains 5

# ============================================================
# 检查
# ============================================================

.PHONY: check-env
check-env:
	@$(PYTHON) -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
	@$(PYTHON) -c "from mamba_ssm import Mamba2; print('mamba-ssm: OK')"
	@$(PYTHON) -c "import calvin_env; print('calvin_env: OK')"
	@$(PYTHON) -c "from src.dual_stream.vla_adapter import OpenVLAAdapter; print('OpenVLAAdapter: OK')"
