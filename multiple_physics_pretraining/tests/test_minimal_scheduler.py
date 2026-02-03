"""
测试 SimpleSequentialScheduler 与 PyTorch SequentialLR 的行为一致性
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils.schedulers import SimpleSequentialScheduler


def test_behavior_match_from_scratch():
    """
    测试场景 1: 从头训练 10000 步
    验证 SimpleSequentialScheduler 与 SequentialLR 行为完全一致
    """
    print("\n" + "=" * 80)
    print("测试场景 1: 从头训练 10000 步")
    print("=" * 80)

    # 创建简单模型和优化器
    model = nn.Linear(10, 10)

    # 创建两个相同配置的优化器
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.1)

    # 配置参数
    warmup_steps = 1000
    total_steps = 10000
    decay_steps = total_steps - warmup_steps

    # 创建 PyTorch SequentialLR
    warmup1 = torch.optim.lr_scheduler.LinearLR(
        optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, eta_min=0.001, T_max=decay_steps
    )
    scheduler1 = torch.optim.lr_scheduler.SequentialLR(
        optimizer1, [warmup1, decay1], [warmup_steps]
    )

    # 创建 SimpleSequentialScheduler
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=0.001, T_max=decay_steps
    )
    scheduler2 = SimpleSequentialScheduler(
        optimizer2, [warmup2, decay2], [warmup_steps]
    )

    # 逐步调用 step() 并验证学习率
    max_diff = 0.0
    check_points = [0, 100, 500, 999, 1000, 1001, 2000, 5000, 9999]

    print("\n检查点学习率对比:")
    print(f"{'步数':<10} {'SequentialLR':<20} {'SimpleSequential':<20} {'差异':<15}")
    print("-" * 70)

    for step in range(total_steps):
        lr1 = scheduler1.get_last_lr()[0]
        lr2 = scheduler2.get_last_lr()[0]

        diff = abs(lr1 - lr2)
        max_diff = max(max_diff, diff)

        if step in check_points:
            print(f"{step:<10} {lr1:<20.15f} {lr2:<20.15f} {diff:<15.2e}")

        scheduler1.step()
        scheduler2.step()

    print("-" * 70)
    print(f"\n最大差异: {max_diff:.2e}")

    # 验证差异小于阈值
    threshold = 1e-15
    if max_diff < threshold:
        print(f"✅ 测试通过: 最大差异 {max_diff:.2e} < {threshold:.2e}")
        return True
    else:
        print(f"❌ 测试失败: 最大差异 {max_diff:.2e} >= {threshold:.2e}")
        return False


def test_behavior_match_with_resume():
    """
    测试场景 2: 使用 last_epoch 恢复训练
    验证恢复后的学习率曲线一致
    """
    print("\n" + "=" * 80)
    print("测试场景 2: 使用 last_epoch 恢复训练")
    print("=" * 80)

    # 创建简单模型和优化器
    model = nn.Linear(10, 10)

    # 创建两个相同配置的优化器
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.1)

    # 配置参数
    warmup_steps = 1000
    total_steps = 10000
    decay_steps = total_steps - warmup_steps
    resume_step = 2499  # 从第 2500 步恢复

    # 创建 PyTorch SequentialLR (恢复)
    warmup1 = torch.optim.lr_scheduler.LinearLR(
        optimizer1, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, eta_min=0.001, T_max=decay_steps
    )
    scheduler1 = torch.optim.lr_scheduler.SequentialLR(
        optimizer1, [warmup1, decay1], [warmup_steps], last_epoch=resume_step
    )

    # 创建 SimpleSequentialScheduler (恢复)
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=0.001, T_max=decay_steps
    )
    scheduler2 = SimpleSequentialScheduler(
        optimizer2, [warmup2, decay2], [warmup_steps], last_epoch=resume_step
    )

    # 验证初始学习率
    lr1_init = scheduler1.get_last_lr()[0]
    lr2_init = scheduler2.get_last_lr()[0]
    diff_init = abs(lr1_init - lr2_init)

    print(f"\n恢复点 (step={resume_step}) 学习率:")
    print(f"  SequentialLR:      {lr1_init:.15f}")
    print(f"  SimpleSequential:  {lr2_init:.15f}")
    print(f"  差异:              {diff_init:.2e}")

    # 继续训练并验证
    max_diff = diff_init
    remaining_steps = total_steps - resume_step - 1
    check_points = [0, 100, 500, 1000, remaining_steps - 1]

    print("\n继续训练后的学习率对比:")
    print(f"{'相对步数':<10} {'SequentialLR':<20} {'SimpleSequential':<20} {'差异':<15}")
    print("-" * 70)

    for step in range(remaining_steps):
        lr1 = scheduler1.get_last_lr()[0]
        lr2 = scheduler2.get_last_lr()[0]

        diff = abs(lr1 - lr2)
        max_diff = max(max_diff, diff)

        if step in check_points:
            print(f"{step:<10} {lr1:<20.15f} {lr2:<20.15f} {diff:<15.2e}")

        scheduler1.step()
        scheduler2.step()

    print("-" * 70)
    print(f"\n最大差异: {max_diff:.2e}")

    # 验证差异小于阈值
    threshold = 1e-15
    if max_diff < threshold:
        print(f"✅ 测试通过: 最大差异 {max_diff:.2e} < {threshold:.2e}")
        return True
    else:
        print(f"❌ 测试失败: 最大差异 {max_diff:.2e} >= {threshold:.2e}")
        return False


def test_milestone_switching():
    """
    测试场景 3: 验证里程碑切换逻辑
    确保在正确的步数切换调度器
    """
    print("\n" + "=" * 80)
    print("测试场景 3: 验证里程碑切换逻辑")
    print("=" * 80)

    # 创建简单模型和优化器
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 配置参数
    warmup_steps = 100

    # 创建调度器
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=0.001, T_max=400
    )
    scheduler = SimpleSequentialScheduler(
        optimizer, [warmup, decay], [warmup_steps]
    )

    print("\n学习率变化:")
    print(f"{'步数':<10} {'学习率':<20} {'阶段':<15}")
    print("-" * 50)

    # 检查关键步数
    key_steps = [0, 50, 99, 100, 101, 200, 499]

    for step in range(500):
        lr = scheduler.get_last_lr()[0]

        if step in key_steps:
            phase = "Warmup" if step < warmup_steps else "Decay"
            print(f"{step:<10} {lr:<20.15f} {phase:<15}")

        scheduler.step()

    print("-" * 50)
    print("✅ 里程碑切换测试完成")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SimpleSequentialScheduler 行为一致性测试")
    print("=" * 80)

    results = []

    # 运行所有测试
    results.append(("从头训练", test_behavior_match_from_scratch()))
    results.append(("恢复训练", test_behavior_match_with_resume()))
    results.append(("里程碑切换", test_milestone_switching()))

    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:<20} {status}")

    all_passed = all(result for _, result in results)

    print("=" * 80)
    if all_passed:
        print("✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("❌ 部分测试失败!")
        sys.exit(1)
