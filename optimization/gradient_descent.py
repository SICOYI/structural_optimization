import torch
import psutil


def optimizer(q, gradients, step):
    q.data -= gradients / torch.norm(gradients) * step

    return q


def check_available_memory():
    """返回当前可用CPU内存（MB）"""
    return psutil.virtual_memory().available / (1024 ** 2)