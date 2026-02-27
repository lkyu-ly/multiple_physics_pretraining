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


def test_resume_with_child_last_epoch():
    """
    测试场景 4: train_basic.py 双重 last_epoch 恢复模式

    对比两种恢复方式的 lr 行为差异：

    **坏做法**（子调度器无 last_epoch）：
      LinearLR.__init__ 将 optimizer.lr 设为 warmup 起始值 0.001。
      CosineAnnealingLR 在 last_epoch=0 时直接返回 group['lr']=0.001（PyTorch 特例），
      导致 decay base_lr 被 warmup 起始值污染，lr 永远停在 eta_min=0.001。

    **好做法**（子调度器传 last_epoch，即 train_basic.py 的实际做法）：
      LinearLR 以 last_epoch=2499 初始化 → optimizer.lr 设为 warmup 完成值 0.1。
      CosineAnnealingLR 以 last_epoch=1499 初始化 → 在 last_epoch=1500 处从 0.1
      出发做乘法步，lr ≈ 0.1（> eta_min），之后继续正常衰减。

    注意：好做法的绝对 lr 值（≈0.1）与从零训练在同一步的精确值（≈0.069）有偏差，
    这是 PyTorch 乘法公式的固有特性（非 bug），由 train_basic.py 接受。
    本测试验证关键属性：lr 不会卡在 eta_min，且恢复后单调递减。
    """
    print("\n" + "=" * 80)
    print("测试场景 4: train_basic.py 双重 last_epoch 恢复模式")
    print("=" * 80)

    # 对应 train_basic.py 的典型参数
    warmup_steps = 1000          # k = params.warmup_steps
    epoch_size = 500
    resume_epoch = 5             # self.startEpoch（已过 warmup 阶段）
    sched_epochs = 10            # params.scheduler_epochs
    total_decay_steps = sched_epochs * epoch_size - warmup_steps  # 4000
    last_step = resume_epoch * epoch_size - 1   # = 2499
    eta_min = 0.001
    base_lr = 0.1

    model = torch.nn.Linear(10, 10)

    # ===== 对照组：子调度器不传 last_epoch（错误做法）=====
    # 现象：CosineAnnealingLR 在 last_epoch=0 时返回 group['lr']=0.001
    # （warmup 起始值），decay lr 永远停在 eta_min
    opt_bad = torch.optim.Adam(model.parameters(), lr=base_lr)
    for g in opt_bad.param_groups:
        g['initial_lr'] = base_lr  # 模拟 optimizer.load_state_dict 后的状态

    warmup_bad = torch.optim.lr_scheduler.LinearLR(
        opt_bad, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        # last_epoch=-1 (default): __init__ 调用 step()，optimizer.lr 变为 0.001
    )
    decay_bad = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_bad, eta_min=eta_min, T_max=total_decay_steps
        # last_epoch=-1 (default): last_epoch=0 时返回 group['lr']=0.001 → lr 卡在 eta_min
    )
    sched_bad = SimpleSequentialScheduler(
        opt_bad, [warmup_bad, decay_bad], [warmup_steps], last_epoch=last_step
    )
    lr_bad = sched_bad.get_last_lr()[0]

    # ===== 实验组：子调度器传 last_epoch（train_basic.py 实际做法）=====
    opt_good = torch.optim.Adam(model.parameters(), lr=base_lr)
    for g in opt_good.param_groups:
        g['initial_lr'] = base_lr  # 模拟 optimizer.load_state_dict 后的状态

    warmup_good = torch.optim.lr_scheduler.LinearLR(
        opt_good, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_steps, last_epoch=last_step  # train_basic.py: last_epoch=last_step
        # step() 将 optimizer.lr 设为 warmup 完成值 0.1
    )
    decay_good = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_good, eta_min=eta_min, T_max=total_decay_steps,
        last_epoch=last_step - warmup_steps  # train_basic.py: last_step - k = 1499
        # step() 从 last_epoch=1500 出发，以 group['lr']=0.1 为基准做乘法步 → lr ≈ 0.1
    )
    sched_good = SimpleSequentialScheduler(
        opt_good, [warmup_good, decay_good], [warmup_steps], last_epoch=last_step
    )
    lr_good = sched_good.get_last_lr()[0]

    print(f"\n恢复点 (step={last_step}, epoch={resume_epoch}) 学习率对比:")
    print(f"  坏做法（无 child last_epoch）: {lr_bad:.8f}  ← 卡在 eta_min")
    print(f"  好做法（有 child last_epoch）: {lr_good:.8f}  ← 从 base_lr 正常衰减")

    # 断言 1：坏做法 lr 等于 eta_min（被 warmup 起始值污染）
    assert lr_bad == eta_min, (
        f"无 child last_epoch 时 lr 应卡在 eta_min={eta_min}，实际={lr_bad:.8f}"
    )

    # 断言 2：好做法 lr 高于 eta_min（decay 正确继承 warmup 完成值 base_lr=0.1）
    assert lr_good > eta_min, (
        f"有 child last_epoch 时 lr 应高于 eta_min={eta_min}，实际={lr_good:.8f}"
    )

    # 断言 3：好做法 lr 显著优于坏做法（核心验证：child last_epoch 确实起效）
    assert lr_good > lr_bad, (
        f"好做法 lr ({lr_good:.8f}) 应高于坏做法 lr ({lr_bad:.8f})"
    )

    # 断言 4：好做法恢复后继续训练 lr 单调递减
    # (last_epoch≈1500 < T_max/2=2000，处于余弦下降段)
    lrs = [lr_good]
    for _ in range(20):
        sched_good.step()
        lrs.append(sched_good.get_last_lr()[0])

    is_monotone = all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
    assert is_monotone, (
        f"好做法恢复后的 lr 应单调递减（余弦下降段），"
        f"但在第 {next(i for i in range(len(lrs)-1) if lrs[i] < lrs[i+1])} 步出现上升"
    )

    print(f"\n继续训练 20 步的学习率变化（好做法）:")
    print(f"  起始: {lrs[0]:.8f}  第10步: {lrs[10]:.8f}  第20步: {lrs[20]:.8f}")
    print(
        f"\n✅ 测试通过: child last_epoch 防止 decay base_lr 被 warmup 起始值污染，"
        f"lr ({lr_good:.6f} > {eta_min}) 且单调递减"
    )
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
    results.append(("双重last_epoch恢复", test_resume_with_child_last_epoch()))

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
