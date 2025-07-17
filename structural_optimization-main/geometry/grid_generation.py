import torch
from typing import List, Tuple

def generate_rectangular_grid(
    length: float,
    width: float,
    n1: int,
    n2: int = 2,
    judge: int = 0,
    z: float = 0,
    height: float = 0
) -> List[List[float]]:
    x_points = [i * (length / n1) for i in range(n1 + 1)]
    y_points = [j * (width / n2) for j in range(n2, -1, -1)]

    grid_points = []
    for x in x_points:
        for y in y_points:
            if abs(y - width/2) < 1e-6:
                grid_points.append([x, y, height])
            else:
                grid_points.append([x, y, z])

    if judge == 1:
        corners = {
            (x_points[0], y_points[0], z),
            (x_points[0], y_points[-1], z),
            (x_points[-1], y_points[0], z),
            (x_points[-1], y_points[-1], z)
        }
        grid_points = [p for p in grid_points if tuple(p) not in corners]

    return grid_points

def generate_connectivity_matrix(
    grid_points: List[List[float]]
) -> List[Tuple[int, int]]:
    """
    生成连接性矩阵（定义哪些点相互连接）
    Args:
        grid_points: 网格点坐标列表
    Returns:
        连接关系列表 [(节点1索引, 节点2索引), ...]
    """
    # 为每个坐标分配唯一索引
    indexed_points = {tuple(point): idx+1 for idx, point in enumerate(grid_points)}
    connectivity = []

    # X方向的连接（垂直线）
    x_values = sorted({p[0] for p in grid_points})
    for x in x_values:
        # 获取当前垂直线上的所有点（按Y从大到小排序）
        points_on_line = [p for p in grid_points if abs(p[0] - x) < 1e-6]
        points_on_line.sort(key=lambda p: -p[1])  # Y降序

        # 连接相邻点
        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append((node1, node2))

    # Y方向的连接（水平线）
    y_values = sorted({p[1] for p in grid_points})
    for y in y_values:
        # 获取当前水平线上的所有点（按X从小到大排序）
        points_on_line = [p for p in grid_points if abs(p[1] - y) < 1e-6]
        points_on_line.sort(key=lambda p: p[0])  # X升序

        # 连接相邻点
        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append((node1, node2))

    return connectivity


def Symmetry_shaper(grid_points, connectivity, free_nodes):

    connectivity = torch.tensor(connectivity, dtype=torch.long, device=device)
    free_nodes = torch.tensor(free_nodes, dtype=torch.long, device=device)

    result_indices_x = []
    result_indices_y = []
    prev_x_list = []
    prev_y_list = []

    for node in free_nodes:
        node_coord = grid_points[node - 1]
        x = node_coord[0]

        if x in prev_x_list:
            x_index = prev_x_list.index(x)
        else:
            result_indices_x.append([])
            prev_x_list.append(x)
            x_index = len(prev_x_list) - 1

        mask = (connectivity == node).any(dim=1)
        candidate_indices = torch.where(mask)[0]  # indexing connectivity

        for idx in candidate_indices:
            conn = connectivity[idx]
            coord1 = grid_points[conn[0] - 1]
            coord2 = grid_points[conn[1] - 1]

            if coord1[0] == coord2[0] and coord1[0] == x:
                result_indices_x[x_index].append(idx.item())

    for node in free_nodes:
        node_coord = grid_points[node - 1]
        y = node_coord[1]

        if y in prev_y_list:
            y_index = prev_y_list.index(y)
        else:
            result_indices_y.append([])
            prev_y_list.append(y)
            y_index = len(prev_y_list) - 1

        mask = (connectivity == node).any(dim=1)
        candidate_indices = torch.where(mask)[0]

        for idx in candidate_indices:
            conn = connectivity[idx]
            coord1 = grid_points[conn[0] - 1]
            coord2 = grid_points[conn[1] - 1]
            if coord1[1] == coord2[1] and coord1[1] == y:
                result_indices_y[y_index].append(idx.item())

    max_len_x = max(len(indices) for indices in result_indices_x) if result_indices_x else 0
    max_len_y = max(len(indices) for indices in result_indices_y) if result_indices_y else 0

    for indices in result_indices_x:
        indices += [-1] * (max_len_x - len(indices))
    for indices in result_indices_y:
        indices += [-1] * (max_len_y - len(indices))

    result_x = torch.tensor(result_indices_x, dtype=torch.long, device=device)
    result_y = torch.tensor(result_indices_y, dtype=torch.long, device=device)
    result_x = torch.unique(result_x, dim=1)
    result_y = torch.unique(result_y, dim=1)

    len_y = result_y.size(0)
    half_y = len_y // 2
    len_x = result_x.size(0)
    half_x = len_x // 2

    y_upper = result_y[:half_y]
    y_lower = result_y[half_y:]
    y_lower = torch.flip(y_lower, dims=[0])
    x_upper = result_x[:half_x]
    x_lower = result_x[half_x:]
    x_lower = torch.flip(x_lower, dims=[0])

    idx_Y = torch.cat((y_upper, y_lower), dim=1)
    idx_X = torch.cat((x_upper, x_lower), dim=1)

    return idx_X, idx_Y