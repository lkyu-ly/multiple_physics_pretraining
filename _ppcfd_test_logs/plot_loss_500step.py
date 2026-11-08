"""一次性绘图脚本：从验收训练 log 绘制 loss 曲线（线性轴）+ [401,500] 窗口均值标注。

用法: python plot_loss_500step.py
输入: 同目录 baseline_off_loss_500step_2026-11-07.log / cinn_loss_500step_2026-11-07.log
输出: 同目录同名 .png 两张
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/lkyu/baidu/MPP/_ppcfd_test_logs"
DATE = "2026-11-07"
TITLE_PREFIX = "MPP AViT-Ti, SWE 128x128, batch=8, AdamW lr=1e-3, fp32"
WIN_LO, WIN_HI = 401, 500  # 全局 step，含端点，100 个点（Epoch 5 全部）

RUNS = [
    (
        f"baseline_off_loss_500step_{DATE}.log",
        f"baseline_off_loss_500step_{DATE}.png",
        "Dynamic graph",
        f"{TITLE_PREFIX} | Dynamic graph",
        "C0",
    ),
    (
        f"cinn_loss_500step_{DATE}.log",
        f"cinn_loss_500step_{DATE}.png",
        "Static graph + CINN",
        f"{TITLE_PREFIX} | Static graph + CINN",
        "C1",
    ),
]

PAT = re.compile(r"Train Loss ([0-9.eE+-]+)")


def parse(path):
    losses = [float(m) for m in PAT.findall(open(path).read())]
    if len(losses) != 500:
        sys.exit(f"{path}: 期望 500 条 Train Loss, 实得 {len(losses)}")
    return losses  # 全局 step 1..500


for log_name, png_name, legend, title, color in RUNS:
    losses = parse(f"{HERE}/{log_name}")
    steps = range(1, len(losses) + 1)
    win = losses[WIN_LO - 1 : WIN_HI]
    mean = sum(win) / len(win)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(steps, losses, color=color, linewidth=1.0, label=legend)
    ax.axvspan(WIN_LO, WIN_HI, color="gray", alpha=0.15, label=f"Steps {WIN_LO}-{WIN_HI}")
    ax.annotate(
        f"mean [{WIN_LO},{WIN_HI}] = {mean:.5f}",
        xy=((WIN_LO + WIN_HI) / 2, max(win)),
        xytext=(140, max(losses) * 0.88),  # 左移避开右上角图例，主数不被遮挡
        arrowprops=dict(arrowstyle="->", color="0.3"),
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{HERE}/{png_name}")
    plt.close(fig)
    print(f"{log_name}: mean[{WIN_LO},{WIN_HI}] = {mean:.6f}  -> {png_name}")
