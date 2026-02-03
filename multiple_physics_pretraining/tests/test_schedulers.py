"""
SimpleSequentialScheduler 单元测试

测试简化的学习率调度器的正确性，包括：
1. 与 PyTorch SequentialLR 的完全一致性
2. 状态保存和恢复
3. 从检查点恢复训练的场景
"""

import sys
from pathlib import Path

import torch
import torch.optim as optim

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schedulers import SimpleSequentialScheduler


class DummyModel(torch.nn.Module):
    """用于测试的简单模型"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)


def test_comparison_with_sequential():
    """与 PyTorch SequentialLR 完全对比"""
    print("\n=== 核心测试：SimpleSequentialScheduler vs SequentialLR ===")

    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 1000
    total_steps = 10000
    min_lr = 0.001

    # 创建自定义调度器
    optimizer1 = optim.Adam(model.parameters(), lr=base_lr)
    warmup1 = torch.optim.lr_scheduler.LinearLR(
        optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    custom_scheduler = SimpleSequentialScheduler(
        optimizer1, schedulers=[warmup1, decay1], milestones=[warmup_steps]
    )

    # 创建原始 SequentialLR
    optimizer2 = optim.Adam(model.parameters(), lr=base_lr)
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer2, [warmup2, decay2], [warmup_steps]
    )

    # 对比学习率曲线
    max_diff = 0.0
    test_steps = [0, 100, 500, 999, 1000, 1500, 5000, 9999]

    print("\n步数对比:")
    print(f"{'步数':<10} {'SimpleSeq':<15} {'SequentialLR':<15} {'差异':<15}")
    print("-" * 60)

    for step in test_steps:
        # 重置调度器
        optimizer1 = optim.Adam(model.parameters(), lr=base_lr)
        warmup1 = torch.optim.lr_scheduler.LinearLR(
            optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, eta_min=min_lr, T_max=total_steps - warmup_steps
        )
        custom_scheduler = SimpleSequentialScheduler(
            optimizer1, schedulers=[warmup1, decay1], milestones=[warmup_steps]
        )

        optimizer2 = optim.Adam(model.parameters(), lr=base_lr)
        warmup2 = torch.optim.lr_scheduler.LinearLR(
            optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, eta_min=min_lr, T_max=total_steps - warmup_steps
        )
        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer2, [warmup2, decay2], [warmup_steps]
        )

        # 运行到指定步数
        for _ in range(step):
            custom_scheduler.step()
            sequential_scheduler.step()

        custom_lr = custom_scheduler.get_last_lr()[0]
        sequential_lr = sequential_scheduler.get_last_lr()[0]
        diff = abs(custom_lr - sequential_lr)
        max_diff = max(max_diff, diff)

        print(f"{step:<10} {custom_lr:<15.8f} {sequential_lr:<15.8f} {diff:<15.2e}")

    print(f"\n最大差异: {max_diff:.2e}")

    # 验证差异在可接受范围内
    assert max_diff < 1e-15, f"学习率差异过大: {max_diff}"

    print("✅ 对比测试通过：学习率曲线完全一致")


def test_full_curve():
    """测试完整的学习率曲线（10000 步）"""
    print("\n=== 测试完整学习率曲线 ===")

    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 1000
    total_steps = 10000
    min_lr = 0.001

    # 创建自定义调度器
    optimizer1 = optim.Adam(model.parameters(), lr=base_lr)
    warmup1 = torch.optim.lr_scheduler.LinearLR(
        optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    custom_scheduler = SimpleSequentialScheduler(
        optimizer1, schedulers=[warmup1, decay1], milestones=[warmup_steps]
    )

    # 创建原始 SequentialLR
    optimizer2 = optim.Adam(model.parameters(), lr=base_lr)
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer2, [warmup2, decay2], [warmup_steps]
    )

    # 运行完整曲线
    max_diff = 0.0
    for step in range(total_steps):
        custom_scheduler.step()
        sequential_scheduler.step()

        custom_lr = custom_scheduler.get_last_lr()[0]
        sequential_lr = sequential_scheduler.get_last_lr()[0]
        diff = abs(custom_lr - sequential_lr)
        max_diff = max(max_diff, diff)

    print(f"运行 {total_steps} 步后的最大差异: {max_diff:.2e}")

    assert max_diff < 1e-15, f"学习率差异过大: {max_diff}"

    print("✅ 完整曲线测试通过")


def test_state_dict():
    """测试状态保存和恢复"""
    print("\n=== 测试状态保存和恢复 ===")

    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 1000
    total_steps = 10000
    min_lr = 0.001

    # 创建第一个调度器并运行一段时间
    optimizer1 = optim.Adam(model.parameters(), lr=base_lr)
    warmup1 = torch.optim.lr_scheduler.LinearLR(
        optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    scheduler1 = SimpleSequentialScheduler(
        optimizer1, schedulers=[warmup1, decay1], milestones=[warmup_steps]
    )

    for _ in range(2500):
        scheduler1.step()

    lr_before_save = scheduler1.get_last_lr()[0]
    state = scheduler1.state_dict()

    print(f"保存时的学习率: {lr_before_save:.6f}")
    print(f"保存时的步数: {state['step_count']}")

    # 创建第二个调度器并加载状态
    optimizer2 = optim.Adam(model.parameters(), lr=base_lr)
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    scheduler2 = SimpleSequentialScheduler(
        optimizer2, schedulers=[warmup2, decay2], milestones=[warmup_steps]
    )

    scheduler2.load_state_dict(state)
    lr_after_load = scheduler2.get_last_lr()[0]

    assert abs(lr_before_save - lr_after_load) < 1e-15, (
        f"恢复后学习率不一致: {lr_before_save} != {lr_after_load}"
    )
    print(f"✓ 恢复后学习率一致: {lr_after_load:.6f}")

    # 继续运行，确保行为一致
    scheduler1.step()
    scheduler2.step()

    lr1 = scheduler1.get_last_lr()[0]
    lr2 = scheduler2.get_last_lr()[0]

    assert abs(lr1 - lr2) < 1e-15, f"继续运行后学习率不一致: {lr1} != {lr2}"
    print(f"✓ 继续运行后学习率一致: {lr1:.6f}")

    print("✅ 状态保存和恢复测试通过")


def test_resume_from_checkpoint():
    """测试从检查点恢复训练的场景"""
    print("\n=== 测试检查点恢复场景 ===")

    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 1000
    total_steps = 10000
    min_lr = 0.001

    # 模拟训练到第 5 个 epoch（假设每个 epoch 500 步）
    epoch_size = 500
    start_epoch = 5
    last_step = (start_epoch * epoch_size) - 1

    print(f"从第 {start_epoch} 个 epoch 恢复训练")
    print(f"上次训练到第 {last_step} 步")

    # 方法 1: 使用 state_dict 恢复（推荐）
    # 先创建一个参考调度器运行到指定步数
    optimizer_ref = optim.Adam(model.parameters(), lr=base_lr)
    warmup_ref = torch.optim.lr_scheduler.LinearLR(
        optimizer_ref, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay_ref = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ref, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    scheduler_ref = SimpleSequentialScheduler(
        optimizer_ref, schedulers=[warmup_ref, decay_ref], milestones=[warmup_steps]
    )

    for _ in range(last_step + 1):
        scheduler_ref.step()

    # 保存状态
    saved_state = scheduler_ref.state_dict()
    ref_lr = scheduler_ref.get_last_lr()[0]
    print(f"参考学习率: {ref_lr:.6f}")

    # 创建新的调度器并恢复状态
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    scheduler = SimpleSequentialScheduler(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )

    # 加载状态
    scheduler.load_state_dict(saved_state)

    # 验证初始学习率正确
    current_lr = scheduler.get_last_lr()[0]
    print(f"恢复后的初始学习率: {current_lr:.6f}")

    assert abs(current_lr - ref_lr) < 1e-15, (
        f"恢复后学习率不一致: {current_lr} != {ref_lr}"
    )
    print("✓ 恢复后学习率正确")

    # 继续训练几步，验证行为一致
    for _ in range(100):
        scheduler.step()
        scheduler_ref.step()

    final_lr = scheduler.get_last_lr()[0]
    final_ref_lr = scheduler_ref.get_last_lr()[0]

    assert abs(final_lr - final_ref_lr) < 1e-15, (
        f"继续训练后学习率不一致: {final_lr} != {final_ref_lr}"
    )
    print(f"✓ 继续训练后学习率一致: {final_lr:.6f}")

    print("✅ 检查点恢复场景测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("SimpleSequentialScheduler 单元测试")
    print("=" * 60)

    try:
        test_comparison_with_sequential()
        test_full_curve()
        test_state_dict()
        test_resume_from_checkpoint()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试出错: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)
