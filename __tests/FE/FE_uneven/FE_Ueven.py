#!/usr/bin/env python
# coding: utf-8
import math

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
import os
from torch.optim import Adam
import psutil
import gc

class BoundedAdam(Adam):
    def __init__(self, params, lr=1e-3, bounds=None, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.bounds = bounds if bounds is not None else {}
        
    def step(self, closure=None):
        super().step(closure)
        # 应用边界约束
        for group in self.param_groups:
            for p in group['params']:
                if p in self.bounds:
                    p.data = torch.clamp(p.data, *self.bounds[p])

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.autograd.set_detect_anomaly(True)

# 自定义参数
length = 24
width = 24
n1 = 9
n2 = 7
judge = 0

## 网格生成函数（修改为返回GPU张量）
def generate_rectangular_grid_sg(length, width, n1, n2=2, judge=0, z=0, height=0):
    x_points = [i * (length / n1) for i in range(n1 + 1)]
    y_points = [j * (width / n2) for j in range(n2, -1, -1)]

    grid_points = []
    for x in x_points:
        for y in y_points:
            if y == width / 2:
                grid_points.append([x, y, height])
            else:
                grid_points.append([x, y, z])

    if judge == 1:
        corners = [
            [x_points[0], y_points[0], z],
            [x_points[0], y_points[-1], z],
            [x_points[-1], y_points[0], z],
            [x_points[-1], y_points[-1], z]
        ]
        grid_points = [point for point in grid_points if point not in corners]

    return torch.tensor(grid_points, dtype=torch.float32, device=device)

def generate_connectivity_matrix(new_coords):
    # 先将坐标数据移到CPU并转换为元组
    new_coords_cpu = new_coords.cpu().detach().numpy()
    indexed_points = {tuple(map(float, point)): idx + 1 for idx, point in enumerate(new_coords_cpu)}
    connectivity = []

    x_values = sorted({point[0] for point in new_coords_cpu})
    for x in x_values:
        points_on_line = [point for point in new_coords_cpu if point[0] == x]
        points_on_line.sort(key=lambda p: p[1], reverse=True) 

        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append([node1, node2])

    y_values = sorted({point[1] for point in new_coords_cpu})
    for y in y_values:
        points_on_line = [point for point in new_coords_cpu if point[1] == y]
        points_on_line.sort(key=lambda p: p[0])  

        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append([node1, node2])

    return torch.tensor(connectivity, device=device)

# 生成网格和连接性矩阵
grid_points = generate_rectangular_grid_sg(length, width, n1, n2, judge)
connectivity = generate_connectivity_matrix(grid_points)

# 问题上下文
n_dof_per_node = 6
total_dof = n_dof_per_node * (n1 + 1) * (n2 + 1)

# 固定节点
x_max = grid_points[:, 0].max()
x_min = grid_points[:, 0].min()
y_max = grid_points[:, 1].max()
y_min = grid_points[:, 1].min()
Fixed_nodes = torch.where(
    (grid_points[:, 1] == y_max) | 
    (grid_points[:, 1] == y_min)
)[0] + 1 

Free_nodes = []
n_elements = len(connectivity)
n_nodes = len(grid_points)
for i in range(1, n_nodes + 1):
    if i not in Fixed_nodes:
        Free_nodes.append(i)

# 转换为GPU张量
Free_nodes = torch.tensor(Free_nodes, device=device)
Fixed_nodes = Fixed_nodes.to(device)

fixed_dof = []
for node in Fixed_nodes:
    fixed_dof.extend([(node - 1) * 6 + i for i in range(6)])
fixed_dof = torch.tensor(fixed_dof, device=device)


def Force_mat(F_value, F_type, total_dof=total_dof, Free_nodes=Free_nodes, judge=0):
    
    F = torch.zeros(total_dof, dtype=torch.float32, device=device)
    
    if judge == 0:
        F_value = torch.tensor([F_value] * len(Free_nodes), device=device) * 1000 # The force value/direction
        F_type = [F_type] * len(Free_nodes)  # The force type
    else:
        F_value = torch.tensor(F_value) * 1000
        F_value = torch.tensor(F_type)
    
    for idx, i in enumerate(Free_nodes):
        F[6 * (i - 1) + F_type[idx]] = F_value[idx]  # unit: KN / KN*m
        
    return F, F_value


# 对称性处理函数
def Indicer(grid_points, Fixed_nodes, length, width, n1, n2):
    x_threshold = length / 2 
    y_threshold = width / 2
    indices = [i for i, point in enumerate(grid_points) 
               if point[0] < x_threshold and point[1] < y_threshold]
    Fixed_idx = [node - 1 for node in Fixed_nodes] 
    Free_indices = [i for i in indices if i not in Fixed_idx]
    return torch.tensor(indices, device=device), torch.tensor(Free_indices, device=device)

def symmetry_shaper(lower_left_points, length = length, width = width):
    lower_left_points = lower_left_points.to(device)
    half_x = length / 2
    half_y = width / 2
    mirrored_x = length - lower_left_points[:, 0]
    mirrored_y = width - lower_left_points[:, 1]
    
    right = torch.stack([mirrored_x, lower_left_points[:, 1], lower_left_points[:, 2]], dim=1)
    top = torch.stack([lower_left_points[:, 0], mirrored_y, lower_left_points[:, 2]], dim=1)
    top_right = torch.stack([mirrored_x, mirrored_y, lower_left_points[:, 2]], dim=1)
    
    full_grid = torch.cat([
        lower_left_points,
        right[mirrored_x >= half_x],
        top[mirrored_y >= half_y],
        top_right[(mirrored_x >= half_x) & (mirrored_y >= half_y)]
    ], dim=0)

    x_sorted = full_grid[torch.argsort(full_grid[:, 0])]
    grouped = x_sorted.view(n1 + 1, n2 + 1, 3)
    sort_indices = torch.argsort(grouped[:, :, 1], dim=1, descending=True)
    sort_indices = sort_indices.unsqueeze(-1).expand(-1, -1, 3)
    y_sorted_groups = torch.gather(grouped, 1, sort_indices)
    
    return y_sorted_groups.view(-1, 3)

# 连接性索引
def Symmetry_shaper(grid_points, connectivity, free_nodes):
    connectivity = connectivity.to(device)
    free_nodes = free_nodes.to(device)
    
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
        candidate_indices = torch.where(mask)[0]
        
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

idx_X, idx_Y = Symmetry_shaper(grid_points, connectivity, Free_nodes)

# FDM部分
px = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
py = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
pz = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)

idx_CF = Fixed_nodes - 1  
idx_CN = Free_nodes - 1
C = torch.zeros(n_elements, n_nodes, dtype=torch.float32, device=device)
for n, (i, j) in enumerate(connectivity):
    C[n, i - 1] = 1
    C[n, j - 1] = -1

CF = C[:, idx_CF]
CN = C[:, idx_CN]

def FDM(Q, F_value, CN=CN, CF=CF, px=px, py=py, pz=pz, 
        Fixed_nodes=Fixed_nodes, Free_nodes=Free_nodes,
        node_coords=grid_points, h_max=24.0, max_retries=5):
    
    pz[:, 0] = F_value
    original_Q = Q.clone()  
    retry_count = 0
    
    while retry_count <= max_retries:
        Dn = torch.matmul(CN.t(), torch.matmul(Q, CN))
        DF = torch.matmul(CN.t(), torch.matmul(Q, CF))
        
        fixed_idces = Fixed_nodes - 1
        xF = node_coords[fixed_idces, 0].unsqueeze(1)
        yF = node_coords[fixed_idces, 1].unsqueeze(1)
        zF = node_coords[fixed_idces, 2].unsqueeze(1)
        
        xN = torch.linalg.solve(Dn, (px - torch.matmul(DF, xF)))  
        yN = torch.linalg.solve(Dn, (py - torch.matmul(DF, yF)))
        zN = torch.linalg.solve(Dn, (pz - torch.matmul(DF, zF)))
        
        z_max = torch.max(zN).item()
        if z_max <= h_max or retry_count == max_retries:
            break
            
        scale = h_max / z_max
        Q = original_Q * (scale ** 0.5)  
        retry_count += 1
        print(f"Retry {retry_count}: Scaling Q by {scale:.3f} (z_max={z_max:.2f} > {h_max})")
    
    new_node_coords = node_coords.clone()
    free_indices = Free_nodes - 1
    new_node_coords[free_indices, 0] = xN.squeeze()
    new_node_coords[free_indices, 1] = yN.squeeze()
    new_node_coords[free_indices, 2] = zN.squeeze()
    
    return new_node_coords



# FE部分（修改为GPU执行）
D_radius = 0.75
D_young_modulus = 10e9
D_shear_modulus = 0.7e9
D_poisson_ratio = 0.3
cross_section_angle_a = 0
cross_section_angle_b = 0
a_small_number = 1e-10


def rotation(v, k, theta):
    v = torch.tensor(v, dtype=torch.float32, device=device)
    k = torch.tensor(k, dtype=torch.float32, device=device)
    theta = torch.tensor(theta, dtype=torch.float32, device=device)
    k = k / torch.norm(k)
    cross_product = torch.cross(k, v)
    dot_product = torch.dot(k, v)
    return v * torch.cos(theta) + cross_product * torch.sin(theta) + k * dot_product * (1 - torch.cos(theta))


class Beam:
    def __init__(self, R, node_coordinates, young_modulus=D_young_modulus,
                 shear_modulus=D_shear_modulus, poisson_ratio=D_poisson_ratio,
                 Beta_a=cross_section_angle_a, Beta_b=cross_section_angle_b):
        self.node_coordinates = node_coordinates.to(device)
        self.radius = torch.tensor(R, dtype=torch.float32, device=device)
        self.young_modulus = torch.tensor(young_modulus, dtype=torch.float32, device=device)
        self.shear_modulus = torch.tensor(shear_modulus, dtype=torch.float32, device=device)
        self.poisson_ratio = torch.tensor(poisson_ratio, dtype=torch.float32, device=device)
        self.Beta_a = torch.tensor(Beta_a, dtype=torch.float32, device=device)
        self.Beta_b = torch.tensor(Beta_b, dtype=torch.float32, device=device)

        self.length = torch.norm(self.node_coordinates[1] - self.node_coordinates[0])
        self.Iy = (torch.pi * self.radius ** 4) / 4
        self.Iz = self.Iy
        self.A = torch.pi * self.radius ** 2
        self.J = (torch.pi * self.radius ** 4) / 2

        # 刚度组件
        self.S_u = self.young_modulus * self.A / self.length
        self.S_v1a = 12 * self.young_modulus * self.Iy / (self.length ** 3)
        self.S_v1b = 6 * self.young_modulus * self.Iy / (self.length ** 2)
        self.S_v2a = 12 * self.young_modulus * self.Iz / (self.length ** 3)
        self.S_v2b = 6 * self.young_modulus * self.Iz / (self.length ** 2)
        self.S_theta1a = 6 * self.young_modulus * self.Iy / (self.length ** 2)
        self.S_theta1b = 4 * self.young_modulus * self.Iy / self.length
        self.S_theta1c = 2 * self.young_modulus * self.Iy / self.length
        self.S_theta2a = 6 * self.young_modulus * self.Iz / (self.length ** 2)
        self.S_theta2b = 4 * self.young_modulus * self.Iz / self.length
        self.S_theta2c = 2 * self.young_modulus * self.Iz / self.length
        self.S_Tr = self.shear_modulus * self.J / self.length

    def get_element_stiffness_matrix(self):
        K_element = torch.tensor([
            [self.S_u, 0, 0, 0, 0, 0, -self.S_u, 0, 0, 0, 0, 0],
            [0, self.S_v1a, 0, 0, 0, self.S_theta1a, 0, -self.S_v1a, 0, 0, 0, self.S_theta1a],
            [0, 0, self.S_v2a, 0, -self.S_theta2a, 0, 0, 0, -self.S_v2a, 0, -self.S_theta2a, 0],
            [0, 0, 0, self.S_Tr, 0, 0, 0, 0, 0, -self.S_Tr, 0, 0],
            [0, 0, -self.S_v2b, 0, self.S_theta2b, 0, 0, 0, self.S_v2b, 0, self.S_theta2c, 0],
            [0, self.S_v1b, 0, 0, 0, self.S_theta1b, 0, -self.S_v1b, 0, 0, 0, self.S_theta1c],
            [-self.S_u, 0, 0, 0, 0, 0, self.S_u, 0, 0, 0, 0, 0],
            [0, -self.S_v1a, 0, 0, 0, -self.S_theta1a, 0, self.S_v1a, 0, 0, 0, -self.S_theta1a],
            [0, 0, -self.S_v2a, 0, self.S_theta2a, 0, 0, 0, self.S_v2a, 0, self.S_theta2a, 0],
            [0, 0, 0, -self.S_Tr, 0, 0, 0, 0, 0, self.S_Tr, 0, 0],
            [0, 0, -self.S_v2b, 0, self.S_theta2c, 0, 0, 0, self.S_v2b, 0, self.S_theta2b, 0],
            [0, self.S_v1b, 0, 0, 0, self.S_theta1c, 0, -self.S_v1b, 0, 0, 0, self.S_theta1b],
        ], dtype=torch.float32, device=device)
        return K_element

    def System_Transform(self):
        vector_x = self.node_coordinates[1, 0] - self.node_coordinates[0, 0]
        vector_y = self.node_coordinates[1, 1] - self.node_coordinates[0, 1]
        vector_z = self.node_coordinates[1, 2] - self.node_coordinates[0, 2]
        length = torch.norm(self.node_coordinates[1] - self.node_coordinates[0])

        z_value = torch.clamp(vector_z / length, min=-1 + 1e-6, max=1 - 1e-6)
        ceta = torch.acos(z_value)
        value = vector_x / torch.sqrt(vector_y ** 2 + vector_x ** 2 + a_small_number)
        value = torch.clamp(value, min=-1 + 1e-6, max=1 - 1e-6)
        alpha = torch.acos(value)

        Projection_Z_x = - vector_z / length * torch.sin(alpha)
        Projection_Z_y = - vector_z / length * torch.cos(alpha)
        Projection_Z_z = torch.cos(torch.pi / 2 - ceta)

        V_projection = torch.stack([Projection_Z_x, Projection_Z_y, Projection_Z_z])
        X_axis = torch.stack([vector_x / length, vector_y / length, vector_z / length])
        Z_axis_a = rotation(V_projection, X_axis, self.Beta_a)
        Y_axis_a = rotation(Z_axis_a, X_axis, -torch.pi / 2)
        Z_axis_a = Z_axis_a / torch.norm(Z_axis_a)
        Y_axis_a = Y_axis_a / torch.norm(Y_axis_a)

        lambda_matrix = torch.stack([X_axis, Y_axis_a, Z_axis_a], dim=0)
        matrix_T = torch.zeros((12, 12), dtype=torch.float32, device=device)
        for i in range(0, 12, 3):
            matrix_T[i:i + 3, i:i + 3] = lambda_matrix
        return matrix_T


def assemble_stiffness_matrix(beams, n_nodes, n_dof_per_node, connectivity):
    total_dof = n_nodes * n_dof_per_node
    K_global = torch.zeros((total_dof, total_dof), dtype=torch.float32, device=device)

    for idx, (i, j) in enumerate(connectivity):
        Matrix_T = beams[idx].System_Transform()
        K_element = torch.matmul(torch.transpose(Matrix_T, 0, 1),
                                 torch.matmul(beams[idx].get_element_stiffness_matrix(), Matrix_T))

        start_idx = (i - 1) * n_dof_per_node
        end_idx = (j - 1) * n_dof_per_node

        K_global[start_idx:start_idx + 6, start_idx:start_idx + 6] += K_element[0:6, 0:6]
        K_global[end_idx:end_idx + 6, end_idx:end_idx + 6] += K_element[6:12, 6:12]
        K_global[start_idx:start_idx + 6, end_idx:end_idx + 6] += K_element[0:6, 6:12]
        K_global[end_idx:end_idx + 6, start_idx:start_idx + 6] += K_element[6:12, 0:6]

    return K_global


def robust_solve(K_global, F, fixed_dof, max_attempts=3):
    """
    鲁棒的线性系统求解器，完整处理固定自由度和奇异问题。

    参数:
        K_global: 全局刚度矩阵（需已处理固定自由度）
        F: 载荷向量
        fixed_dof: 固定自由度索引列表
        max_attempts: 最大尝试次数
    """
    attempts = 0
    while attempts < max_attempts:
        # 1. 基础正则化（保持固定自由度的大对角元不变）
        reg = 1e-6 * torch.eye(K_global.shape[0], device=K_global.device)
        reg[fixed_dof, fixed_dof] = 0  # 不干扰固定自由度
        K_reg = K_global + reg

        try:
            # 尝试直接求解（双精度）
            displacements = torch.linalg.solve(
                K_reg.to(torch.float64),
                F.to(torch.float64)
            )
            return displacements.to(K_global.dtype)

        except RuntimeError:
            # 2. 识别并处理极端刚度（跳过固定自由度）
            diag = torch.diag(K_global)
            extreme_mask = (diag > 1e12) & (~torch.isin(torch.arange(len(diag)), torch.tensor(fixed_dof)))  # 排除固定自由度
            K_reg[extreme_mask] = 0
            K_reg[:, extreme_mask] = 0
            K_reg[extreme_mask, extreme_mask] = 1e12  # 设为合理上限

            # 3. 确保固定自由度约束不被破坏
            K_reg[fixed_dof, :] = 0
            K_reg[:, fixed_dof] = 0
            K_reg[fixed_dof, fixed_dof] = 1e10  # 保持原始大值

            try:
                # 尝试迭代法（共轭梯度）
                displacements, info = torch.linalg.cg(
                    K_reg.to(torch.float64),
                    F.to(torch.float64),
                    maxiter=5000,
                    atol=1e-6
                )
                if info > 0:
                    raise RuntimeError("CG未收敛")
                return displacements.to(K_global.dtype)

            except:
                # 4. 最终回退：伪逆（保持固定自由度约束）
                K_pinv = torch.linalg.pinv(K_reg)
                K_pinv[fixed_dof, :] = 0  # 固定自由度位移强制为0
                displacements = K_pinv @ F
                print("警告：使用伪逆求解，精度可能降低")
                return displacements

        attempts += 1

    raise RuntimeError("无法求解线性系统")


def Strain_E(node_coords, connectivity, fixed_dof, F, force, D_radius=D_radius):
    # Element Assembly
    Beam_lens = []
    beams = []
    for idx, connection in enumerate(connectivity):
        node_1_coords = node_coords[connection[0] - 1]
        node_2_coords = node_coords[connection[1] - 1]

        if force[idx] == 0:
            R = D_radius
        elif force[idx] > 0:
            sigma = 50
            R = math.sqrt(force[idx] / (math.pi * sigma))
        elif force[idx] < 0:
            sigma = 100
            R = math.sqrt(abs(force[idx]) / (math.pi * sigma))

        beam = Beam(R, node_coordinates=torch.stack([node_1_coords, node_2_coords]),
                    young_modulus=D_young_modulus,
                    shear_modulus=D_shear_modulus, poisson_ratio=D_poisson_ratio, Beta_a=cross_section_angle_a,
                    Beta_b=cross_section_angle_b)
        beams.append(beam)
        Beam_lens.append(beam.length)

    # Stiffness renewal
    K_global = assemble_stiffness_matrix(beams, n_nodes=len(node_coords), n_dof_per_node=6, connectivity=connectivity)
    K_global[fixed_dof, :] = 0
    K_global[:, fixed_dof] = 0
    K_global[fixed_dof, fixed_dof] = 1e10

    displacements = robust_solve(K_global, F, fixed_dof)

    # Compute strain energy
    strain_energy_list = []
    force_list = []
    ASE_list = []
    V_list = []
    Local_d = torch.zeros(len(connectivity), 12, dtype=torch.float32, device=device)
    for n, (i, j) in enumerate(connectivity):
        matrix_T = beams[n].System_Transform()
        Tep_displacements = torch.cat(
            [displacements[6 * (i - 1):6 * (i - 1) + 6], displacements[6 * (j - 1):6 * (j - 1) + 6]], dim=0)
        Local_d_n = torch.matmul(Tep_displacements, matrix_T.T)
        Local_d[n, :] = Local_d_n.clone()
        K_l = beams[n].get_element_stiffness_matrix()
        strain_energy_list.append(0.5 * torch.matmul(Local_d_n, torch.matmul(K_l, Local_d_n.reshape(-1, 1))))
        force_list.append(torch.matmul(K_l, Local_d_n.reshape(-1, 1)))
        ASE_list.append(0.5 * (Local_d_n[0] - Local_d_n[6]) * beams[n].S_u * (Local_d_n[0] - Local_d_n[6]))
        V_list.append(beams[n].A * beams[n].length)

    Strain_energy = torch.stack(strain_energy_list)
    forces = torch.stack(force_list)
    lens = torch.stack(Beam_lens)
    ASE = torch.stack(ASE_list)
    V = torch.stack(V_list)
    return Strain_energy, forces, displacements, ASE, V, lens






# 可视化函数（需要将数据移回CPU）
def save_OPT(state_idx, grid_points, new_node_coords,
             connectivity, Free_nodes, Fixed_nodes, force, SED,
             save_dir="results", max_states=6):
    """
    多状态组合图保存函数S

    参数:
        state_idx: 状态序号 (0=初始, 1=第cut次, 2=第2*cut次...)
        max_states: 组合图中最多显示的状态数
    """
    os.makedirs(save_dir, exist_ok=True)

    # 数据准备
    x_orig = grid_points[:, 0].cpu().detach().numpy()
    y_orig = grid_points[:, 1].cpu().detach().numpy()
    z_orig = grid_points[:, 2].cpu().detach().numpy()

    x_fdm = new_node_coords[:, 0].cpu().detach().numpy()
    y_fdm = new_node_coords[:, 1].cpu().detach().numpy()
    z_fdm = new_node_coords[:, 2].cpu().detach().numpy()

    # 计算当前高度
    current_height = max(z_fdm)

    # 初始化图形容器
    if not hasattr(save_OPT, 'fig'):
        save_OPT.fig = plt.figure(figsize=(24, 16))
        save_OPT.axes = [save_OPT.fig.add_subplot(2, 3, i + 1, projection='3d')
                         for i in range(max_states)]
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        save_OPT.saved_states = 0
        save_OPT.max_z = 0  # 用于统一z轴尺度

    # 检查是否已存满
    if save_OPT.saved_states >= max_states:
        filename = os.path.join(save_dir, f"FE_States_{max_states}.png")
        save_OPT.fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(save_OPT.fig)
        delattr(save_OPT, 'fig')
        delattr(save_OPT, 'axes')
        delattr(save_OPT, 'saved_states')
        delattr(save_OPT, 'max_z')
        print(f"Saved full states to {filename}")
        return

    # 更新最大z值（用于统一坐标尺度）
    if current_height > save_OPT.max_z:
        save_OPT.max_z = current_height

    # 获取当前子图并清除旧内容
    ax = save_OPT.axes[save_OPT.saved_states]
    ax.clear()

    # ========== 可视化绘制 ==========
    # 1. 绘制原始网格（浅灰色虚线）
    for i, j in connectivity:
        ax.plot([x_orig[i - 1], x_orig[j - 1]],
                [y_orig[i - 1], y_orig[j - 1]],
                [z_orig[i - 1], z_orig[j - 1]],
                ':', color='#CCCCCC', linewidth=0.8, alpha=0.7)

    # 2. 绘制当前状态网格
    color = '#1f77b4' if state_idx == 0 else '#ff7f0e'  # 初始蓝色，迭代橙色
    for i, j in connectivity:
        ax.plot([x_fdm[i - 1], x_fdm[j - 1]],
                [y_fdm[i - 1], y_fdm[j - 1]],
                [z_fdm[i - 1], z_fdm[j - 1]],
                '-', color=color, linewidth=1.8, alpha=0.9)

    # 3. 标记固定节点（黑色实心圆）
    for node in Fixed_nodes:
        ax.scatter(x_fdm[node - 1], y_fdm[node - 1], z_fdm[node - 1],
                   c='black', s=50, marker='o', alpha=0.8)

    # 4. 添加高度标注（替换原来的力值标注）
    ax.text(x=0.05, y=0.90, z=save_OPT.max_z * 1.05,
            s=f"Height: {current_height:.2f}m\nSE: {SED:.8f} ",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))

    # ========== 子图装饰 ==========
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_title(f"State {state_idx}" if state_idx > 0 else "Initial State",
                 fontsize=11, pad=12)
    ax.set_zlim(0, save_OPT.max_z * 1.1)  # 统一z轴尺度
    ax.view_init(elev=35, azim=45)
    ax.grid(True, linestyle=':', alpha=0.5)

    # 更新状态计数器
    save_OPT.saved_states += 1

    # 如果是最后一个状态，立即保存
    if save_OPT.saved_states == max_states:
        filename = os.path.join(save_dir, f"FE_States_{max_states}.png")
        save_OPT.fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(save_OPT.fig)
        delattr(save_OPT, 'fig')
        delattr(save_OPT, 'axes')
        delattr(save_OPT, 'saved_states')
        delattr(save_OPT, 'max_z')
        print(f"Saved full states to {filename}")


def finalize_OPT(save_dir="results", completed=False):

    if not hasattr(save_OPT, 'fig') or save_OPT.saved_states == 0:
        return

    # 如果已完成所有迭代，保存当前进度（不强制填满）
    if completed:
        filename = os.path.join(save_dir,
                                f"FE_States_completed_{save_OPT.saved_states}.png")
    # 如果是提前终止，保存上一次有效迭代
    else:
        # 回退一个状态，因为最后一次迭代可能不完整
        save_OPT.saved_states = max(0, save_OPT.saved_states - 1)
        filename = os.path.join(save_dir,
                                f"FE_States_partial_{save_OPT.saved_states + 1}.png")

    # 保存图像
    save_OPT.fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(save_OPT.fig)

    # 打印保存信息
    if completed:
        print(f"Saved completed states ({save_OPT.saved_states}/{len(save_OPT.axes)}) to {filename}")
    else:
        print(f"Saved last valid state ({save_OPT.saved_states + 1}) to {filename}")

    # 清理属性
    for attr in ['fig', 'axes', 'saved_states', 'max_z']:
        if hasattr(save_OPT, attr):
            delattr(save_OPT, attr)


# In[2]:


def optimizer(OPT_variables, gradients, step):
    
    OPT_variables.data -= gradients / torch.norm(gradients) * step
    
    with torch.no_grad():
        OPT_variables.data = torch.clamp(OPT_variables.data, min=0.0, max=12.0)

    return OPT_variables
    
def check_available_memory():
    """返回当前可用CPU内存（MB）"""
    return psutil.virtual_memory().available / (1024 ** 2)




def run_fe_test(N_coords, grid_points, connectivity, Fixed_nodes, Free_nodes,
                total_dof, fixed_dof, Beam_lens, Volume, Max_height,
                force_value=4, force_direction=0, title_suffix=""):

    # 设置力条件
    F_test = torch.zeros(total_dof, dtype=torch.float32, device=N_coords.device)
    f_test = [force_direction] * len(Free_nodes)
    F_vatest = torch.tensor([force_value] * len(Free_nodes), device=N_coords.device) * 1000

    for idx, i in enumerate(Free_nodes):
        F_test[6 * (i - 1) + f_test[idx]] = F_vatest[idx]

    # 运行有限元分析
    Strain_energy_test, forces_test, *_ = Strain_E(
        N_coords.clone(), connectivity, fixed_dof, F_test
    )

    # 计算结果指标
    force_t = abs(forces_test[:, 0, 0])
    load_path_t = torch.dot(force_t, Beam_lens)
    Total_ES_t = torch.sum(Strain_energy_test)
    SED = Total_ES_t / Volume

    print(f"\nFE Test Results ({title_suffix}):")
    print(f"Load path: {load_path_t.item():.2f}")
    print(f"Total strain energy: {Total_ES_t.item():.2f}")
    print(f"Strain energy density: {SED.item():.4f}")

    # 准备可视化数据
    x_orig = grid_points[:, 0].cpu().detach().numpy()
    y_orig = grid_points[:, 1].cpu().detach().numpy()
    z_orig = grid_points[:, 2].cpu().detach().numpy()

    x_fdm = N_coords[:, 0].cpu().detach().numpy()
    y_fdm = N_coords[:, 1].cpu().detach().numpy()
    z_fdm = N_coords[:, 2].cpu().detach().numpy()
    force_np = force_t.cpu().detach().numpy()

    Max_height = max(z_fdm)
    # 创建图表
    fig = go.Figure()

    # 添加原始网格线
    for i, j in connectivity:
        fig.add_trace(go.Scatter3d(
            x=[x_orig[i - 1], x_orig[j - 1]],
            y=[y_orig[i - 1], y_orig[j - 1]],
            z=[z_orig[i - 1], z_orig[j - 1]],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Grid',
            showlegend=False
        ))

    # 添加优化后网格线
    for i, j in connectivity:
        fig.add_trace(go.Scatter3d(
            x=[x_fdm[i - 1], x_fdm[j - 1]],
            y=[y_fdm[i - 1], y_fdm[j - 1]],
            z=[z_fdm[i - 1], z_fdm[j - 1]],
            mode='lines',
            line=dict(color='red', width=4),
            name='FDM solution',
            showlegend=False
        ))

    # 添加固定节点
    for node in Fixed_nodes:
        fig.add_trace(go.Scatter3d(
            x=[x_fdm[node - 1]],
            y=[y_fdm[node - 1]],
            z=[z_fdm[node - 1]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name=f'Fixed Node {node}',
            showlegend=False
        ))

    # 添加力显示
    force_traces = []
    for idx, (i, j) in enumerate(connectivity):
        mid_x = (x_fdm[i - 1] + x_fdm[j - 1]) / 2
        mid_y = (y_fdm[i - 1] + y_fdm[j - 1]) / 2
        mid_z = (z_fdm[i - 1] + z_fdm[j - 1]) / 2

        trace = go.Scatter3d(
            x=[mid_x],
            y=[mid_y],
            z=[mid_z],
            mode='markers+text',
            marker=dict(size=1, color='green'),
            text=[f"{force_np[idx]:.0f}"],
            textposition='top center',
            textfont=dict(size=8),
            name=f'Force {idx + 1}',
            visible=True
        )
        force_traces.append(trace)
        fig.add_trace(trace)

    # 确定标题
    direction_map = {2: "Gravity", 0: "Lateral X", 1: "Lateral Y"}
    direction_name = direction_map.get(force_direction, f"Direction {force_direction}")
    title = f"Test with {direction_name} Load {title_suffix}"

    # 更新布局
    fig.update_layout(
        updatemenus=[{
            'type': "buttons",
            'direction': "right",
            'x': 0.1,
            'y': 1.1,
            'buttons': [
                {
                    'label': "✅ Show forces",
                    'method': "update",
                    'args': [{"visible": [True] * len(fig.data)}],
                },
                {
                    'label': "❌ Hide forces",
                    'method': "update",
                    'args': [{"visible": [True] * (len(fig.data) - len(force_traces)) + [False] * len(force_traces)}],
                }
            ]
        }],
        scene={
            'xaxis': {'showbackground': False, 'showgrid': False, 'showline': False, 'showticklabels': False,
                      'title': ''},
            'yaxis': {'showbackground': False, 'showgrid': False, 'showline': False, 'showticklabels': False,
                      'title': ''},
            'zaxis': {'showbackground': False, 'showgrid': False, 'showline': False, 'showticklabels': False,
                      'title': ''},
            'aspectmode': 'data'
        },
        title=title,
        annotations=[{
            'x': 0.05,
            'y': 0.95,
            'xref': "paper",
            'yref': "paper",
            'text': f"Strain energy = {Total_ES_t.item():.4f}, Volume = {Volume:.4f}, Max_height = {Max_height:.4f}, SED = {SED.item():.4f}",
            'showarrow': False,
            'font': {'size': 14, 'color': "black"},
            'bgcolor': "white",
            'bordercolor': "black",
            'borderwidth': 1,
            'borderpad': 4
        }]
    )

    # 保存结果
    filename = f"E_strain_Test_{direction_name.replace(' ', '_')}{title_suffix}.html"
    fig.write_html(filename)
    print(f"Visualization saved to {filename}")

    return {
        'strain_energy': Total_ES_t.item(),
        'load_path': load_path_t.item(),
        'SED': SED.item(),
        'figure': fig
    }
# In[3]:



# 初始化
q = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1], device=device)
q_vec = torch.zeros(n_elements, device=device)
for i in range(len(idx_X)):
    q_vec[idx_X[i,:]] = q[i]
for j in range(len(idx_Y)):
    q_vec[idx_Y[j,:]] = q[j+len(idx_X)]
D_radius = 0.75

# 梯度下降参数
step = 0.005
epochs = 500
patience = 20
count = 0

_, F_value = Force_mat(-1, 2)
F_fe_g, _ = Force_mat(-1, 2)
F_fe_l, _ = Force_mat(0, 0)

F_fe = F_fe_g + F_fe_l

r = 1 / torch.max(F_value)
print('r', r)
Q = torch.diag(q_vec) * 1 / r 
Ini_G = FDM(Q, F_value)
cut = epochs / 5



import time

ES_his = []
R_his = []
Ratio_his = []
V_his = []
force = torch.zeros(len(connectivity), dtype=torch.float32, device=device)
RESULTS_FILE = os.path.join("data_records", "optimization_results.txt")
os.makedirs("data_records", exist_ok=True)
with open(RESULTS_FILE, 'a') as f:
    f.write(f"Context:XZ")
    f.write(f"Total epochs: {epochs}\n")
    f.write(f"Step length: {step}\n")

Id, Free_Id = Indicer(Ini_G, Fixed_nodes, length, width, n1, n2)
OPT_variables = Ini_G[Free_Id][:, 2].detach().clone().requires_grad_(True)
Crd = Ini_G[Id].detach().clone()
iddx = torch.nonzero(Id.unsqueeze(0) == Free_Id.unsqueeze(1), as_tuple=True)[1]

for iteration in range(epochs + 1):
    str_time = time.time()
    print(f'Iteration {iteration}')

    # 前向传播
    Crd[iddx, 2] = OPT_variables
    N_coords = symmetry_shaper(Crd).clone()
    ex_force = force
    Strain_energy, forces, displacements, ASE, V, Beam_lens = Strain_E(N_coords, connectivity, fixed_dof, F_fe, force)

    force = abs(forces[:, 0, 0])
    Total_ES = torch.sum(Strain_energy)

    ES_his.append(Total_ES.item())
    load_path = torch.dot(force, Beam_lens)

    Volume = torch.sum(V)
    # epsilon = (D1 + D2) / Beam_lens
    SED = Total_ES / V

    R = torch.var(SED)
    R_his.append(R.item())

    Rate = torch.sum(ASE) / Total_ES
    Ratio_his.append(Rate.item())
    print('Total_ES', Total_ES)
    print('load_path', load_path)
    print('R', R)

    # 早期停止检查
    if iteration > 0:
        Pre_Total_ES = ES_his[iteration - 1]
        change = abs(Total_ES - Pre_Total_ES) / Pre_Total_ES
        if change < 1 / 10000:
            count += 1
        else:
            count = 0
        if count >= patience:
            print(f"Early stopping at iteration {iteration}")
            break

            # 计算变形后坐标
    N_coords_flat = N_coords.reshape(-1)
    New_Coordinates = torch.zeros(n_nodes * 3, dtype=torch.float32, device=device)
    for n in range(n_nodes):
        New_Coordinates[3 * n: 3 * n + 3] = N_coords_flat[3 * n: 3 * n + 3] + displacements[6 * n: 6 * n + 3]
    New_Coordinates = New_Coordinates.view(n_nodes, 3).clone()

    # 反向传播
    Back_str = time.time()
    if OPT_variables.grad is not None:
        OPT_variables.grad.detach_()
        OPT_variables.grad.zero_()

    Total_ES.backward(retain_graph=True)
    Back_time = (time.time() - Back_str) / 60

    # 梯度信息
    gradients = OPT_variables.grad
    frob_norm = torch.norm(gradients)
    OPT_variables = optimizer(OPT_variables, gradients, step)

    end_time = (time.time() - str_time) / 60
    # 定期保存结果
    if iteration % cut == 0 or iteration == 0:
        with open(RESULTS_FILE, 'a') as f:
            f.write(f"\n=== Iteration {iteration} ===\n")
            f.write(f"Ite Time: {end_time}\n")
            f.write(f"Total Strain Energy: {Total_ES.item():.6f}\n")
            f.write(f"Load path: {load_path.item():.6f}\n")
            f.write(f"Var: {R.item():.6f}\n")
            f.write(f"Volume: {Volume.item():.6f}\n")
            f.write(f"Axial ratio: {Rate.item():.6f}\n")
            f.write(f"Back propagation time: {Back_time:.6f}\n")


        state_idx = iteration // cut
        save_OPT(state_idx, grid_points, N_coords,
                 connectivity, Free_nodes, Fixed_nodes,
                 force, Total_ES.detach().cpu().item())

        print(f"Saved data at iteration {iteration}")
    if iteration % 5 == 0:
        print(f"Iter {iteration}: Grad Norm = {frob_norm.item():.4f}, LR = {step}, Load_path = {load_path.item()}")

finalize_OPT()
with open(RESULTS_FILE, 'a') as f:
    f.write(f"Forces: {ex_force.detach().numpy():.2f}\n")
print("Optimization completed.")
print("Final Strain Energy:", Total_ES.item())

# In[9]:


fig, ax1 = plt.subplots(figsize=(10, 7))

color = 'red'
ax1.set_ylabel('E_strain', color=color)
ax1.plot(range(len(ES_his)), ES_his, label='Total_Es', color=color, linewidth=2, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

# Mark specific points
marker_points = [
    0,  # First point
    *range(50, len(ES_his), 50),  # Every 200th point
    len(ES_his)-1  # Last point
]

for point in marker_points:
    ax1.scatter(point, ES_his[point], color='blue', zorder=5)
    ax1.text(point, ES_his[point]*1.001, 
             f'({ES_his[point]:.4f})',
             ha='right' if point == len(ES_his)-1 else 'left',
             va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax1.legend(loc='upper right')
plt.title('Total_Es vs. Iterations')
plt.tight_layout()
plt.savefig('Loss_history.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


#############################################################################################################
## Visualization 1
x_orig = grid_points[:, 0].cpu().detach().numpy()
y_orig = grid_points[:, 1].cpu().detach().numpy()
z_orig = grid_points[:, 2].cpu().detach().numpy()

x_fdm = N_coords[:, 0].cpu().detach().numpy()
y_fdm = N_coords[:, 1].cpu().detach().numpy()
z_fdm = N_coords[:, 2].cpu().detach().numpy()

Max_height = max(z_fdm)

fig = go.Figure()

for connection in connectivity:
    i, j = connection
    fig.add_trace(go.Scatter3d(
        x=[x_orig[i-1], x_orig[j-1]],
        y=[y_orig[i-1], y_orig[j-1]],
        z=[z_orig[i-1], z_orig[j-1]],
        mode='lines',
        line=dict(color='blue', width=1),
        name='Grid',
        showlegend=False
    ))

for connection in connectivity:
    i, j = connection
    fig.add_trace(go.Scatter3d(
        x=[x_fdm[i-1], x_fdm[j-1]],
        y=[y_fdm[i-1], y_fdm[j-1]],
        z=[z_fdm[i-1], z_fdm[j-1]],
        mode='lines',
        line=dict(color='red', width=4),
        name='Found',
        showlegend=False
    ))


for node in Fixed_nodes:
    fig.add_trace(go.Scatter3d(
        x=[x_fdm[node-1]],
        y=[y_fdm[node-1]],
        z=[z_fdm[node-1]],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        name=f'Fixed Node {node}',
        showlegend=False
    ))
    

force_traces = []
force_np = force.cpu().detach().numpy() 
for idx, connection in enumerate(connectivity):
    i, j = connection
    mid_x = (x_fdm[i-1] + x_fdm[j-1]) / 2
    mid_y = (y_fdm[i-1] + y_fdm[j-1]) / 2
    mid_z = (z_fdm[i-1] + z_fdm[j-1]) / 2
    trace = go.Scatter3d(
        x=[mid_x],
        y=[mid_y],
        z=[mid_z],
        mode='markers+text',
        marker=dict(size=1, color='green'),
        text=[f"{force_np[idx]:.0f}"],
        textposition='top center',
        textfont=dict(size=8),
        name=f'Force {idx+1}',
        visible=True
    )
    force_traces.append(trace)
    fig.add_trace(trace)

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.1,
            y=1.1,
            buttons=[
                dict(
                    label="✅ Show forces",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)}],
                ),
                dict(
                    label="❌ Hide forces",
                    method="update",
                    args=[{"visible": [True] * (len(fig.data) - len(force_traces)) + [False] * len(force_traces)}],
                )
            ]
        )
    ],
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            title=''
        ),
        zaxis=dict(
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            title=''
        ),
        aspectmode='data'
    ),
    title='OPT',
    annotations=[
        dict(
            x=0.05,  # X position (0-1, left to right)
            y=0.95,  # Y position (0-1, bottom to top)
            xref="paper",
            yref="paper",
            text=f"Strain energy= {Total_ES:.4f}, Volume = {Volume:.4f}, Max_height = {Max_height:.4f} ",
            showarrow=False,
            font=dict(
                size=14,
                color="black"
            ),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    ]
)

fig.show()

fig.write_html("E_strain Opt(FE).html")


# In[ ]:

# 运行重力测试 (Z方向)
gravity_results = run_fe_test(
    N_coords, grid_points, connectivity, Fixed_nodes, Free_nodes,
    total_dof, fixed_dof, Beam_lens, Volume, Max_height,
    force_value=-1, force_direction=2, title_suffix="(Optimized)"
)

# 运行横向X测试
lateral_x_results = run_fe_test(
    N_coords, grid_points, connectivity, Fixed_nodes, Free_nodes,
    total_dof, fixed_dof, Beam_lens, Volume, Max_height,
    force_value=1, force_direction=0, title_suffix="(Optimized)"
)

# 运行横向Y测试
lateral_y_results = run_fe_test(
    N_coords, grid_points, connectivity, Fixed_nodes, Free_nodes,
    total_dof, fixed_dof, Beam_lens, Volume, Max_height,
    force_value=1, force_direction=1, title_suffix="(Optimized)"
)






