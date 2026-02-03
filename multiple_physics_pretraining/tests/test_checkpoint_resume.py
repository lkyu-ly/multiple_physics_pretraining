"""
验证检查点恢复时学习率的正确性

对比 SimpleSequentialScheduler 和 PyTorch SequentialLR 在恢复场景下的行为
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


def test_checkpoint_resume_comparison():
    """
    对比测试：验证从检查点恢复时，SimpleSequentialScheduler
    和 SequentialLR 的学习率是否完全一致
    """
    print("\n" + "=" * 70)
    print("检查点恢复对比测试")
    print("=" * 70)

    model = DummyModel()
    base_lr = 0.1
    warmup_steps = 1000
    total_steps = 10000
    min_lr = 0.001

    # 模拟训练场景：训练到第 2500 步后保存检查点
    checkpoint_step = 2500
    print(f"\n模拟场景：训练到第 {checkpoint_step} 步后保存检查点")

    # ========== 场景 1: 使用 SimpleSequentialScheduler ==========
    print("\n--- 场景 1: SimpleSequentialScheduler ---")

    # 从头训练到 checkpoint_step
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

    # 运行到 checkpoint_step
    for _ in range(checkpoint_step):
        scheduler1.step()

    lr_at_checkpoint_simple = scheduler1.get_last_lr()[0]
    print(f"在第 {checkpoint_step} 步的学习率: {lr_at_checkpoint_simple:.8f}")

    # 模拟保存检查点（记录 step_count）
    saved_step_count = scheduler1.step_count
    print(f"保存的 step_count: {saved_step_count}")

    # 模拟恢复：重新创建调度器，使用 last_epoch 参数
    # 关键：需要先恢复优化器状态，这样 param_groups 中才有 initial_lr
    last_step = checkpoint_step - 1  # last_epoch 是上一次执行的步数
    optimizer1_restored = optim.Adam(model.parameters(), lr=base_lr)

    # 模拟恢复优化器状态（这是关键步骤！）
    # 在实际代码中，这是通过 optimizer.load_state_dict() 完成的
    # 这里我们手动设置 initial_lr 来模拟恢复后的状态
    for group in optimizer1_restored.param_groups:
        group['initial_lr'] = base_lr

    warmup1_restored = torch.optim.lr_scheduler.LinearLR(
        optimizer1_restored,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
        last_epoch=last_step,
    )
    decay1_restored = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1_restored,
        eta_min=min_lr,
        T_max=total_steps - warmup_steps,
        last_epoch=last_step - warmup_steps if last_step >= warmup_steps else -1,
    )
    scheduler1_restored = SimpleSequentialScheduler(
        optimizer1_restored,
        schedulers=[warmup1_restored, decay1_restored],
        milestones=[warmup_steps],
    )
    scheduler1_restored.step_count = last_step + 1

    lr_after_restore_simple = scheduler1_restored.get_last_lr()[0]
    print(f"恢复后的学习率: {lr_after_restore_simple:.8f}")
    print(f"差异: {abs(lr_at_checkpoint_simple - lr_after_restore_simple):.2e}")

    # ========== 场景 2: 使用 PyTorch SequentialLR ==========
    print("\n--- 场景 2: PyTorch SequentialLR ---")

    # 从头训练到 checkpoint_step
    optimizer2 = optim.Adam(model.parameters(), lr=base_lr)
    warmup2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    decay2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, eta_min=min_lr, T_max=total_steps - warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.SequentialLR(
        optimizer2, [warmup2, decay2], [warmup_steps]
    )

    # 运行到 checkpoint_step
    for _ in range(checkpoint_step):
        scheduler2.step()

    lr_at_checkpoint_seq = scheduler2.get_last_lr()[0]
    print(f"在第 {checkpoint_step} 步的学习率: {lr_at_checkpoint_seq:.8f}")

    # 模拟恢复：重新创建调度器，使用 last_epoch 参数
    # 关键：需要先恢复优化器状态，这样 param_groups 中才有 initial_lr
    optimizer2_restored = optim.Adam(model.parameters(), lr=base_lr)

    # 模拟恢复优化器状态（这是关键步骤！）
    for group in optimizer2_restored.param_groups:
        group['initial_lr'] = base_lr

    warmup2_restored = torch.optim.lr_scheduler.LinearLR(
        optimizer2_restored,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
        last_epoch=last_step,
    )
    decay2_restored = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2_restored,
        eta_min=min_lr,
        T_max=total_steps - warmup_steps,
        last_epoch=last_step - warmup_steps if last_step >= warmup_steps else -1,
    )
    scheduler2_restored = torch.optim.lr_scheduler.SequentialLR(
        optimizer2_restored,
        [warmup2_restored, decay2_restored],
        [warmup_steps],
        last_epoch=last_step,
    )

    lr_after_restore_seq = scheduler2_restored.get_last_lr()[0]
    print(f"恢复后的学习率: {lr_after_restore_seq:.8f}")
    print(f"差异: {abs(lr_at_checkpoint_seq - lr_after_restore_seq):.2e}")

    # ========== 对比两种方案 ==========
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)

    print(f"\n恢复前学习率对比:")
    print(f"  SimpleSequentialScheduler: {lr_at_checkpoint_simple:.8f}")
    print(f"  SequentialLR:              {lr_at_checkpoint_seq:.8f}")
    print(f"  差异:                      {abs(lr_at_checkpoint_simple - lr_at_checkpoint_seq):.2e}")

    print(f"\n恢复后学习率对比:")
    print(f"  SimpleSequentialScheduler: {lr_after_restore_simple:.8f}")
    print(f"  SequentialLR:              {lr_after_restore_seq:.8f}")
    print(f"  差异:                      {abs(lr_after_restore_simple - lr_after_restore_seq):.2e}")

    # 验证恢复的正确性
    assert abs(lr_at_checkpoint_simple - lr_after_restore_simple) < 1e-15, (
        f"SimpleSequentialScheduler 恢复后学习率不一致: "
        f"{lr_at_checkpoint_simple} != {lr_after_restore_simple}"
    )

    assert abs(lr_at_checkpoint_seq - lr_after_restore_seq) < 1e-15, (
        f"SequentialLR 恢复后学习率不一致: "
        f"{lr_at_checkpoint_seq} != {lr_after_restore_seq}"
    )

    assert abs(lr_after_restore_simple - lr_after_restore_seq) < 1e-15, (
        f"两种方案恢复后学习率不一致: "
        f"{lr_after_restore_simple} != {lr_after_restore_seq}"
    )

    print("\n✅ 验证通过：两种方案恢复后的学习率完全一致！")

    # ========== 继续训练验证 ==========
    print("\n" + "=" * 70)
    print("继续训练验证（恢复后再训练 100 步）")
    print("=" * 70)

    for step in range(100):
        scheduler1_restored.step()
        scheduler2_restored.step()

        lr1 = scheduler1_restored.get_last_lr()[0]
        lr2 = scheduler2_restored.get_last_lr()[0]

        if step % 20 == 0:
            print(
                f"步数 {checkpoint_step + step + 1}: "
                f"Simple={lr1:.8f}, Seq={lr2:.8f}, "
                f"差异={abs(lr1 - lr2):.2e}"
            )

        assert abs(lr1 - lr2) < 1e-15, (
            f"继续训练后学习率不一致 (步数 {checkpoint_step + step + 1}): "
            f"{lr1} != {lr2}"
        )

    print("\n✅ 验证通过：继续训练后学习率曲线完全一致！")

    print("\n" + "=" * 70)
    print("✅ 所有检查点恢复测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_checkpoint_resume_comparison()
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"❌ 测试失败: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 测试出错: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        sys.exit(1)
