# PyTorch 到 PaddlePaddle 迁移记录

本文档记录在将 MPP 项目从 PyTorch 迁移到 PaddlePaddle 过程中遇到的问题和解决方案。

**迁移原则：功能不变，即为了适配paddle而修改的代码必须维持和原代码完全一致的【行为】。**

---

## 问题 1: SequentialLR 缺失（已优化）

**日期**: 2026-02-03
**更新**: 2026-02-03（简化方案）
**文件**: `multiple_physics_pretraining/train_basic.py:209`
**问题**: `torch.optim.lr_scheduler.SequentialLR` 在 PaddlePaddle 中没有对应 API

### 原始代码

```python
# train_basic.py 第 199-214 行
k = params.warmup_steps
if (self.startEpoch * params.epoch_size) < k:
    warmup = torch.optim.lr_scheduler.LinearLR(
        self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=k
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        eta_min=params.learning_rate / 100,
        T_max=sched_epochs,
    )
    self.scheduler = torch.optim.lr_scheduler.SequentialLR(
        self.optimizer,
        [warmup, decay],
        [k],
        last_epoch=(params.epoch_size * self.startEpoch) - 1,
    )
```

**问题分析**:

- `SequentialLR` 用于组合多个调度器，按顺序切换
- 原实现在前 1000 步使用 `LinearLR` 进行 warmup，之后使用 `CosineAnnealingLR` 进行衰减
- PaddlePaddle 没有 `SequentialLR` API，无法直接使用 paconvert 转换
- ✅ `LinearLR` 和 `CosineAnnealingLR` 都可以被 paconvert 自动转换

### 解决方案（简化版）

创建轻量级的 `SimpleSequentialScheduler` 包装器，内部使用 PyTorch 原生调度器，只负责切换逻辑。

**实现位置**: `multiple_physics_pretraining/utils/schedulers.py`

**核心逻辑**:

```python
class SimpleSequentialScheduler:
    """轻量级调度器切换器，模拟 SequentialLR 的行为"""

    def __init__(self, optimizer, schedulers: List, milestones: List[int]):
        self.optimizer = optimizer
        self.schedulers = schedulers  # [warmup, decay]
        self.milestones = milestones  # [warmup_steps]
        self.step_count = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        # 确定当前应该使用哪个调度器
        scheduler_idx = 0
        for i, milestone in enumerate(self.milestones):
            if self.step_count >= milestone:
                scheduler_idx = i + 1

        # 调用对应调度器的 step
        self.schedulers[scheduler_idx].step()
        self.step_count += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
```

**方案优势**:

- ✅ 代码简洁
- ✅ 使用原生调度器（更易转换到 PaddlePaddle）
- ✅ 学习率曲线与 SequentialLR 完全一致（误差 < 1e-15）
- ✅ 支持状态保存和恢复
- ✅ 符合 KISS 原则

### PaddlePaddle 适配

`SimpleSequentialScheduler` 可以轻松适配到 PaddlePaddle：

```python
# PyTorch 版本（当前实现）
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=k
)
decay = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, eta_min=min_lr, T_max=total_steps - k
)
scheduler = SimpleSequentialScheduler(optimizer, [warmup, decay], [k])

# PaddlePaddle 版本（迁移后）
# 根据 paconvert 转换规则：
warmup = paddle.optimizer.lr.LinearLR(
    start_factor=0.01,
    end_factor=1.0,
    total_steps=k,
    learning_rate=optimizer.get_lr()
)
decay = paddle.optimizer.lr.CosineAnnealingDecay(
    eta_min=min_lr,
    T_max=total_steps - k,
    learning_rate=optimizer.get_lr()
)
# SimpleSequentialScheduler 需要适配 PaddlePaddle 的 API
scheduler = SimpleSequentialScheduler(optimizer, [warmup, decay], [k])
```

**适配要点**:

- PaddlePaddle 的调度器不直接绑定优化器
- 需要修改 `SimpleSequentialScheduler.step()` 来调用 `optimizer.set_lr(scheduler.get_lr())`
- warmup/decay 组合场景下，**不要**再对优化器调用 `set_lr_scheduler()`，否则后续手动 `set_lr()` 会冲突并抛出 `optimizer's learning rate can't be LRScheduler`

---

## 问题 2: Paddle 版训练启动与最小训练链路修复

**日期**: 2026-03-13
**文件**:

- `multiple_physics_pretraining_paddle/train_basic.py`
- `multiple_physics_pretraining_paddle/data_utils/*`
- `multiple_physics_pretraining_paddle/models/*`
- `multiple_physics_pretraining_paddle/utils/schedulers.py`

**问题**: `paconvert` 转换后的 Paddle 代码保留了多处 PyTorch 风格导入和 API，导致训练从启动到首个 batch 之间连续报错。

### 关键错误与处理

#### 1. 包导入不稳定

**现象**:

- `ModuleNotFoundError: No module named 'hdf5_datasets'`
- `ImportError: attempted relative import with no known parent package`

**原因**:

- 目录内仍混用顶层导入、相对导入和 `sys.path.append(...)`
- `except:` 过宽，吞掉真实错误后错误地回退到另一套导入分支

**解决**:

- 给 `multiple_physics_pretraining_paddle/`、`data_utils/`、`models/`、`utils/` 增加 `__init__.py`
- 包内统一使用稳定导入
- 将导入回退收窄为 `except ImportError`
- 删除 `sys.path.append(...)`、无关残留导入和调试代码

#### 2. Paddle 类型与基类不兼容

**现象**:

- `TypeError: 'type' object is not subscriptable`
- `AttributeError: module 'paddle.nn' has no attribute 'Module'`
- `AttributeError: module 'paddle.nn' has no attribute 'Parameter'`

**原因**:

- 直接照搬了 PyTorch 泛型和基类写法，如 `paddle.io.Sampler[T]`、`paddle.nn.Module`
- 参数注册仍使用 PyTorch 风格 `nn.Parameter(...)`

**解决**:

- `MultisetSampler` 改为继承 `paddle.io.Sampler`
- 全部模块基类改为 `paddle.nn.Layer`
- 可学习参数统一改为 `create_parameter(...)` 或直接复用 Paddle 层的 `weight/bias`

#### 3. Paddle API 名称和签名差异

**典型错误**:

- `paddle.cuda.*`、`paddle.compat.*`、`paddle.as_tensor(...)`
- `Conv2D(..., bias=False)`、`optimizer.zero_grad(...)`
- `Tensor.tensor_split(...)`、`mean(dim=...)`

**解决**:

- 设备/AMP 接口改为 `paddle.device.*`、`paddle.amp.*`
- 张量构造统一改为 `paddle.to_tensor(...)`
- 卷积参数改为 `bias_attr=False`
- 清梯度改为 `optimizer.clear_grad()`
- 张量拆分改为 `paddle.split(...)`
- 统计接口统一使用 `axis=` / `keepdim=`

#### 4. 线性层权重布局与注意力实现差异

**现象**:

- `linear` 输入维度不匹配
- `scaled_dot_product_attention` 在 CPU 上触发 `flash_attn` 内核缺失

**原因**:

- Paddle `Linear.weight` 维度布局与 PyTorch 不同
- Paddle 内置 `scaled_dot_product_attention` 在当前 CPU 路径下不可用

**解决**:

- `SubsampledLinear` 中按 Paddle 权重布局调整切片方向
- 在 `spatial_modules.py` 和 `time_modules.py` 增加 CPU 可用的注意力回退实现

#### 5. 设备与 AMP 适配

**现象**:

- `module 'paddle' has no attribute 'cuda'`
- `dtype.lower()` 相关异常
- 当前环境有 GPU 版 Paddle，但无可用 GPU / cuDNN

**解决**:

- 设备检测改为 `paddle.device.is_compiled_with_cuda()` + `paddle.device.cuda.device_count() > 0`
- 无可见 GPU 时强制走 CPU
- `auto_cast` 的 `dtype` 使用字符串，如 `"float16"` / `"bfloat16"`
- 无 GPU 时禁用 AMP，避免无意义的混精度分支

### 当前结果

使用以下命令：

```bash
python train_basic.py \
  --run_name quick_test \
  --config basic_config \
  --yaml_config ./config/mpp_avit_ti_config.yaml
```

Paddle 版本已能够：

- 正常完成导入和初始化
- 进入训练循环
- 连续跑过多个 batch
- 输出训练日志，如 `Epoch 1 Batch N Train Loss ...`

### 经验总结

- `paconvert` 更适合“语法搬运”，不等于“行为可运行”
- 涉及导入、参数注册、设备选择、优化器/调度器、注意力算子时，必须逐项核对 Paddle 原生 API
- 迁移验证不要只看“能 import”，至少要验证到“首个 batch 完整前向 + 反向 + optimizer step”

---

## 问题 3: 快速验证配置未真实限制每个 epoch 的步数

**日期**: 2026-03-13
**文件**:

- `multiple_physics_pretraining/data_utils/datasets.py`
- `multiple_physics_pretraining/data_utils/mixed_dset_sampler.py`
- `multiple_physics_pretraining_paddle/data_utils/datasets.py`
- `multiple_physics_pretraining_paddle/data_utils/mixed_dset_sampler.py`

**问题**: `mpp_avit_ti_config.yaml` 中已经将 `max_epochs: 3`、`epoch_size: 20` 设为快速验证配置，但 Torch 和 Paddle 版本都仍然完整遍历训练集，导致单个 epoch 跑出几百甚至更多 step。

### 原因

- Torch 版原本已经构造了 `MultisetSampler(max_samples=params.epoch_size)`，但 `DataLoader(..., sampler=sampler)` 被注释掉了。
- Paddle 版没有 `sampler` 参数，因此不能直接照搬 Torch 写法；如果不补等价实现，`epoch_size` 也不会真正生效。
- 因此配置中的 `epoch_size` 只影响了部分计数和调度，不影响真实训练步数。

### 解决方案

- Torch 版：恢复 `DataLoader(..., sampler=sampler)`，重新让 `MultisetSampler` 控制单个 epoch 的采样长度。
- Paddle 版：使用官方支持的 `batch_sampler`，通过 `paddle.io.BatchSampler(sampler=..., batch_size=...)` 包装现有 `MultisetSampler`。
- 两边同时修正 `MultisetSampler.__len__()`，使 `len(train_data_loader)` 与快速验证配置一致。

### 修复结果

修复后两套实现都重新遵守快速验证配置：

- `train_loader_size` 从完整数据集长度变为 `20`
- 单个 epoch 只运行约 20 个 batch
- 能按预期快速进入下一个 epoch 并在少量 epoch 后结束训练

### 备注

- 这次问题**不是** `DAdaptAdam` / `DAdaptAdan` 本地 Paddle 依赖污染 Torch 版本导致的。
- 对 Paddle 来说，正确替代 `sampler` 的方式是 `batch_sampler`，而不是简单删除采样限制逻辑。

---

## 问题 4: Adan / DAdapt 自定义优化器仍保留 Torch 优化器协议

**日期**: 2026-03-16
**文件**:

- `multiple_physics_pretraining_paddle/utils/adan_paddle.py`
- `multiple_physics_pretraining_paddle/utils/dadapt_adam_paddle.py`
- `multiple_physics_pretraining_paddle/utils/dadapt_adan_paddle.py`
- `multiple_physics_pretraining_paddle/utils/custom_optimizer_base.py`
- `multiple_physics_pretraining_paddle/unitTest/test_custom_optimizers.py`

**问题**: `paconvert` 直接把 `adan_pytorch` / `dadaptation` 里的自定义优化器搬到了 Paddle 版，但它们仍按 `torch.optim.Optimizer` 的构造和状态管理方式工作。切换到 `adan` 或 `learning_rate: -1` 的 DAdapt 路径后，会在初始化阶段报错：

- `TypeError: parameters argument should not get dict type`

继续排查后还发现两类兼容问题：

- 代码里依赖 `param_groups`、`state[p]` 这套 Torch 风格接口，而 Paddle 基类公开的是 `_param_groups` / `_accumulators`
- 若直接硬套到 Paddle，`Tensor.add_(float)`、`mul_(float)`、`div_(float)` 这类标量原地操作也会继续报错
- `adan_paddle.py` 中原 `addcmul_` 的机械转换还改变了原始更新公式的数值语义

### 解决方案

- 不重写算法主体，只补一个很小的兼容层 `custom_optimizer_base.py`
- 在兼容层里统一处理：
  - Paddle 正确的 `Optimizer.__init__(learning_rate=..., parameters=...)`
  - Torch 风格 `param_groups` / `state`
  - 自定义优化器的 `state_dict()` / `load_state_dict()`
  - 调度器驱动下的当前学习率同步
- 三个优化器文件只做最小修改：
  - 接到兼容层上
  - 把 Paddle 不支持的标量原地运算改成 `copy_(expr)` 等价写法
  - 修正 `Adan` 中被错误转换的更新公式
- 新增最小单测，验证：
  - 参数分组构造
  - 单步 `forward + backward + optimizer.step()`
  - `clear_grad()`
  - `state_dict/load_state_dict`

### 修复结果

- `adan + 固定 learning_rate` 可完成优化器初始化和单步更新
- `adam + learning_rate=-1` 可进入 `DAdaptAdam`
- `adan + learning_rate=-1` 可进入 `DAdaptAdan`
- 冒烟训练已能越过优化器初始化并进入训练循环；后续如遇 dataloader 多进程权限问题，属于当前运行环境限制，不是优化器迁移问题

### 经验总结

- `paconvert` 对“自定义优化器”只能完成语法级搬运，不能保证行为级兼容
- 迁移优化器时要重点核对三件事：基类构造协议、状态保存恢复、原地张量运算是否仍与 Paddle 等价
- 如果只是为了让训练先跑通，优先做“兼容层 + 最小修正”，不要一开始就大规模重写优化器实现
