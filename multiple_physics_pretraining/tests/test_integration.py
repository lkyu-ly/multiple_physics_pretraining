"""
集成测试：验证 WarmupCosineScheduler 在训练流程中的使用

测试场景：
1. 模拟训练循环，验证学习率正确更新
2. 模拟检查点保存和恢复
"""

import sys
from pathlib import Path

import torch
import torch.optim as optim

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.schedulers import WarmupCosineScheduler


class DummyModel(torch.nn.Module):
    """用于测试的简单模型"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def test_training_loop():
    """测试训练循环中的学习率更新"""
    print("\n=== 测试训练循环 ===")

    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    warmup_steps = 1000
    total_steps = 10000
    base_lr = 0.1
    min_lr = 0.001

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
    )

    # 模拟训练循环
    print("模拟训练 100 步...")
    for step in range(100):
        # 模拟前向传播和反向传播
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        # 调度器步骤
        scheduler.step()

        if step % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step}: LR = {lr:.6f}")

    print("✅ 训练循环测试通过")


def test_checkpoint_save_restore():
    """测试检查点保存和恢复"""
    print("\n=== 测试检查点保存和恢复 ===")

    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    warmup_steps = 1000
    total_steps = 10000
    base_lr = 0.1
    min_lr = 0.001
    epoch_size = 500

    # 模拟训练到第 5 个 epoch
    print("模拟训练到第 5 个 epoch...")
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
    )

    # 训练 5 个 epoch
    for epoch in range(5):
        for step in range(epoch_size):
            x = torch.randn(5, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    # 保存检查点
    checkpoint = {
        "epoch": 5,
        "iters": 5 * epoch_size,
        "model_state": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    lr_before_save = scheduler.get_last_lr()[0]
    print(f"保存时的学习率: {lr_before_save:.6f}")
    print(f"保存时的步数: {checkpoint['iters']}")

    # 模拟恢复训练
    print("\n恢复训练...")
    model2 = DummyModel()
    optimizer2 = optim.Adam(model2.parameters(), lr=base_lr)

    # 加载检查点
    model2.load_state_dict(checkpoint["model_state"])
    optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    last_step = (start_epoch * epoch_size) - 1

    # 创建调度器（使用 last_step）
    scheduler2 = WarmupCosineScheduler(
        optimizer=optimizer2,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
        last_step=last_step,
    )

    # 加载调度器状态
    scheduler2.load_state_dict(checkpoint["scheduler_state_dict"])

    lr_after_restore = scheduler2.get_last_lr()[0]
    print(f"恢复后的学习率: {lr_after_restore:.6f}")

    assert abs(lr_before_save - lr_after_restore) < 1e-6, (
        f"恢复后学习率不一致: {lr_before_save} != {lr_after_restore}"
    )
    print("✓ 恢复后学习率一致")

    # 继续训练
    print("\n继续训练 2 个 epoch...")
    for epoch in range(2):
        for step in range(epoch_size):
            x = torch.randn(5, 10)
            y = model2(x)
            loss = y.sum()
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()

    final_lr = scheduler2.get_last_lr()[0]
    print(f"继续训练后的学习率: {final_lr:.6f}")

    print("✅ 检查点保存和恢复测试通过")


def test_different_scenarios():
    """测试不同的训练场景"""
    print("\n=== 测试不同训练场景 ===")

    base_lr = 0.1
    min_lr = 0.001

    # 场景 1: 短训练（仅 warmup）
    print("\n场景 1: 短训练（仅 warmup）")
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=1000,
        total_steps=10000,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
    )

    for _ in range(500):
        scheduler.step()

    lr = scheduler.get_last_lr()[0]
    print(f"  500 步后学习率: {lr:.6f}")
    assert 0.001 < lr < 0.1, "学习率应该在 warmup 范围内"
    print("  ✓ 短训练场景正常")

    # 场景 2: 长训练（完整 decay）
    print("\n场景 2: 长训练（完整 decay）")
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=1000,
        total_steps=10000,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
    )

    for _ in range(10000):
        scheduler.step()

    lr = scheduler.get_last_lr()[0]
    print(f"  10000 步后学习率: {lr:.6f}")
    assert abs(lr - min_lr) < 1e-6, "学习率应该衰减到最小值"
    print("  ✓ 长训练场景正常")

    # 场景 3: 从中间恢复
    print("\n场景 3: 从中间恢复（第 3000 步）")
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=1000,
        total_steps=10000,
        base_lr=base_lr,
        min_lr=min_lr,
        start_factor=0.01,
        last_step=2999,  # 从第 3000 步恢复
    )

    lr = scheduler.get_last_lr()[0]
    print(f"  恢复后学习率: {lr:.6f}")
    assert 0.05 < lr < 0.1, "学习率应该在 decay 中期范围内"
    print("  ✓ 中间恢复场景正常")

    print("\n✅ 所有场景测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("WarmupCosineScheduler 集成测试")
    print("=" * 60)

    try:
        test_training_loop()
        test_checkpoint_save_restore()
        test_different_scenarios()

        print("\n" + "=" * 60)
        print("✅ 所有集成测试通过！")
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
