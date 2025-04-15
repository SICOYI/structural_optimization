import torch
from typing import Tuple


def initialize_fdm_matrices(
        connectivity: List[Tuple[int, int]],
        n_nodes: int,
        free_indices: torch.Tensor,
        fixed_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    初始化FDM所需的矩阵
    Args:
        connectivity: 连接关系列表 [(i,j), ...]
        n_nodes: 总节点数
        free_indices: 自由节点索引（0-based）
        fixed_indices: 固定节点索引（0-based）
    Returns:
        CF: 固定节点关联矩阵
        CN: 自由节点关联矩阵
    """
    n_elements = len(connectivity)
    C = torch.zeros(n_elements, n_nodes, dtype=torch.float32)

    for elem_idx, (i, j) in enumerate(connectivity):
        C[elem_idx, i - 1] = 1  # 注意转为0-based索引
        C[elem_idx, j - 1] = -1

    CF = C[:, fixed_indices]
    CN = C[:, free_indices]
    return CF, CN


def solve_fdm(
        Q: torch.Tensor,
        F_value: torch.Tensor,
        CN: torch.Tensor,
        CF: torch.Tensor,
        node_coords: torch.Tensor,
        fixed_indices: torch.Tensor,
        free_indices: torch.Tensor
) -> torch.Tensor:
    """
    FDM求解器核心
    Args:
        Q: 权重对角矩阵 (n_elements × n_elements)
        F_value: 外力值 (n_free_nodes × 1)
        CN: 自由节点关联矩阵
        CF: 固定节点关联矩阵
        node_coords: 初始节点坐标 (n_nodes × 3)
        fixed_indices: 固定节点索引 (0-based)
        free_indices: 自由节点索引 (0-based)
    Returns:
        新节点坐标 (n_nodes × 3)
    """
    # 初始化载荷向量
    px = torch.zeros(len(free_indices), 1, dtype=torch.float32)
    py = torch.zeros_like(px)
    pz = F_value.unsqueeze(1)  # 转为列向量

    # 计算刚度矩阵分量
    Dn = torch.matmul(CN.t(), torch.matmul(Q, CN))
    Df = torch.matmul(CN.t(), torch.matmul(Q, CF))

    # 固定节点坐标
    xF = node_coords[fixed_indices, 0].unsqueeze(1)
    yF = node_coords[fixed_indices, 1].unsqueeze(1)
    zF = node_coords[fixed_indices, 2].unsqueeze(1)

    # 求解位移
    xN = torch.linalg.solve(Dn, px - torch.matmul(Df, xF))
    yN = torch.linalg.solve(Dn, py - torch.matmul(Df, yF))
    zN = torch.linalg.solve(Dn, pz - torch.matmul(Df, zF))

    # 更新坐标
    new_coords = node_coords.clone()
    new_coords[free_indices, 0] = xN.squeeze()
    new_coords[free_indices, 1] = yN.squeeze()
    new_coords[free_indices, 2] = zN.squeeze()

    return new_coords


def force_distribution(
        Q: torch.Tensor,
        new_coords: torch.Tensor,
        CN: torch.Tensor,
        CF: torch.Tensor,
        fixed_indices: torch.Tensor
) -> torch.Tensor:
    """
    计算节点力分布
    Args:
        Q: 权重矩阵
        new_coords: 变形后坐标
        CN, CF: 关联矩阵
        fixed_indices: 固定节点索引
    Returns:
        节点力向量 (n_elements × 1)
    """
    delta_X = torch.matmul(CN, new_coords[:, 0].unsqueeze(1)) + \
              torch.matmul(CF, new_coords[fixed_indices, 0].unsqueeze(1))
    forces = torch.matmul(Q, delta_X)
    return forces