import os
import sys

import paddle

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.adan_paddle import Adan
from utils.dadapt_adam_paddle import DAdaptAdam
from utils.dadapt_adan_paddle import DAdaptAdan


def _make_param_groups():
    layer = paddle.nn.Linear(4, 2)
    return layer, [
        {"params": [layer.bias], "weight_decay": 0.0},
        {"params": [layer.weight], "weight_decay": 1e-3},
    ]


def _run_one_step(optimizer_cls, lr):
    paddle.seed(2026)
    layer, params = _make_param_groups()
    optimizer = optimizer_cls(params, lr=lr)
    x = paddle.randn([3, 4])
    y = paddle.randn([3, 2])
    loss = paddle.nn.functional.mse_loss(layer(x), y)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return optimizer, layer


def _grad_cleared(param):
    return param.grad is None or float(paddle.abs(param.grad).sum()) == 0.0


def test_adan_accepts_param_groups_and_steps():
    optimizer, layer = _run_one_step(Adan, lr=1e-4)
    assert len(optimizer.param_groups) == 2
    assert _grad_cleared(layer.weight)


def test_dadapt_adam_accepts_param_groups_and_steps():
    optimizer, layer = _run_one_step(DAdaptAdam, lr=1.0)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["d"] > 0
    assert _grad_cleared(layer.weight)


def test_dadapt_adan_accepts_param_groups_and_steps():
    optimizer, layer = _run_one_step(DAdaptAdan, lr=1.0)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["d"] > 0
    assert _grad_cleared(layer.weight)


def test_custom_optimizer_state_dict_round_trip():
    optimizer, layer = _run_one_step(DAdaptAdan, lr=1.0)
    state = optimizer.state_dict()

    params = [
        {"params": [layer.bias], "weight_decay": 0.0},
        {"params": [layer.weight], "weight_decay": 1e-3},
    ]
    new_optimizer = DAdaptAdan(params, lr=1.0)
    new_optimizer.set_state_dict(state)

    assert new_optimizer.param_groups[0]["k"] == optimizer.param_groups[0]["k"]
    assert new_optimizer.state[layer.weight]["step"] == optimizer.state[layer.weight][
        "step"
    ]
