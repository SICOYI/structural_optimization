#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
import time
from datetime import timedelta
import json
import os
import psutil
import gc
import math

gc.collect()  
device = torch.device('cpu')
print(f"Using device: {device}")


# In[2]:


############ customization
length = 24
width = 24
n1 = 13
n2 = 13
judge = 0


# In[58]:


############### Problem context formulation

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

    return grid_points

def plot_grid(grid_points, length, width):
    x = [point[0] for point in grid_points]
    y = [point[1] for point in grid_points]
    z = [point[2] for point in grid_points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', s=50, label='Grid Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def generate_connectivity_matrix(new_coords):
    indexed_points = {tuple(point): idx + 1 for idx, point in enumerate(new_coords)}
    connectivity = []

    x_values = sorted(set(point[0] for point in new_coords))
    for x in x_values:
        points_on_line = [point for point in new_coords if point[0] == x]
        points_on_line.sort(key=lambda p: p[1], reverse=True) 

        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append([node1, node2])

    y_values = sorted(set(point[1] for point in new_coords))
    for y in y_values:
        points_on_line = [point for point in new_coords if point[1] == y]
        points_on_line.sort(key=lambda p: p[0])  

        for i in range(len(points_on_line) - 1):
            node1 = indexed_points[tuple(points_on_line[i])]
            node2 = indexed_points[tuple(points_on_line[i + 1])]
            connectivity.append([node1, node2])

    return connectivity

grid_points = generate_rectangular_grid_sg(length, width, n1, n2, judge)
connectivity = generate_connectivity_matrix(grid_points)
plot_grid(grid_points, length, width)


###########################################################################################################################################################
n_dof_per_node = 6  # Degrees of freedom per node
grid_points = torch.tensor(grid_points, device=device, dtype=torch.float32)
total_dof = n_dof_per_node * len(grid_points)

########## Surrounding fixed
x_max = grid_points[:, 0].max()
x_min = grid_points[:, 0].min()
y_max = grid_points[:, 1].max()
y_min = grid_points[:, 1].min()

Fixed_nodes = torch.where(
        (grid_points[:, 0] == x_max) | 
    (grid_points[:, 0] == x_min) |    
    (grid_points[:, 1] == y_max) |  # y = y_max
    (grid_points[:, 1] == y_min)    # y = y_min
)[0]

Fixed_nodes += 1
Free_nodes = []

n_elements = len(connectivity)
n_nodes = len(grid_points)

for i in range(1, n_nodes + 1):
    if i not in Fixed_nodes:
        Free_nodes.append(i)


####### BCs
fixed_dof = []
for node in Fixed_nodes:
    fixed_dof.extend([(node - 1) * 6 + i for i in range(6)])

##########################################################################################################################################################
##########################################################################################################################################################


# In[59]:


rows = n2 + 1
cols = n1 + 1

idx_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

for k in range(rows * cols):
    row = k % rows
    col = k // rows
    idx_matrix[row][col] = k
V_matrix = torch.arange(0, n2 * (n1 + 1)).reshape(n1 + 1, n2).T

start = n2 * (n1 + 1)
H_matrix = torch.arange(start, start + (n2 + 1) * n1)
H_matrix = H_matrix.reshape(n1 + 1 , n2)
H_matrix = H_matrix.flip(0)
print(V_matrix)
print(H_matrix)


# In[60]:


def Symmetry_shaper(q, matrix):

    rows, cols = matrix.shape

    row_odd = rows % 2 == 1
    col_odd = cols % 2 == 1
    
    if row_odd:

        q = torch.cat([q, q.flip(0)[1:,:]], dim=0)
    else:

        q = torch.cat([q, q.flip(0)], dim=0)
    

    if col_odd:

        q = torch.cat([q, q.flip(1)[:,1:]], dim=1)
    else:

        q = torch.cat([q, q.flip(1)], dim=1)
    
    return q
    


# In[61]:


############################################################# Force condition
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


# In[62]:


def save_fdm(state_idx, grid_points, new_node_coords, 
            connectivity, Free_nodes, Fixed_nodes, force, SED,
            save_dir="results", max_states=6):
    """
    多状态组合图保存函数
    
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
    if not hasattr(save_fdm, 'fig'):
        save_fdm.fig = plt.figure(figsize=(24, 16))
        save_fdm.axes = [save_fdm.fig.add_subplot(2, 3, i+1, projection='3d') 
                       for i in range(max_states)]
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        save_fdm.saved_states = 0
        save_fdm.max_z = 0  # 用于统一z轴尺度
    
    # 检查是否已存满
    if save_fdm.saved_states >= max_states:
        filename = os.path.join(save_dir, f"FDM_States_{max_states}.png")
        save_fdm.fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(save_fdm.fig)
        delattr(save_fdm, 'fig')
        delattr(save_fdm, 'axes')
        delattr(save_fdm, 'saved_states')
        delattr(save_fdm, 'max_z')
        print(f"Saved full states to {filename}")
        return
    
    # 更新最大z值（用于统一坐标尺度）
    if current_height > save_fdm.max_z:
        save_fdm.max_z = current_height
    
    # 获取当前子图并清除旧内容
    ax = save_fdm.axes[save_fdm.saved_states]
    ax.clear()
    
    # ========== 可视化绘制 ==========
    # 1. 绘制原始网格（浅灰色虚线）
    for i, j in connectivity:
        ax.plot([x_orig[i-1], x_orig[j-1]],
                [y_orig[i-1], y_orig[j-1]],
                [z_orig[i-1], z_orig[j-1]], 
                ':', color='#CCCCCC', linewidth=0.8, alpha=0.7)
    
    # 2. 绘制当前状态网格
    color = '#1f77b4' if state_idx == 0 else '#ff7f0e'  # 初始蓝色，迭代橙色
    for i, j in connectivity:
        ax.plot([x_fdm[i-1], x_fdm[j-1]],
                [y_fdm[i-1], y_fdm[j-1]],
                [z_fdm[i-1], z_fdm[j-1]], 
                '-', color=color, linewidth=1.8, alpha=0.9)
    
    # 3. 标记固定节点（黑色实心圆）
    for node in Fixed_nodes:
        ax.scatter(x_fdm[node-1], y_fdm[node-1], z_fdm[node-1],
                  c='black', s=50, marker='o', alpha=0.8)
    
    # 4. 添加高度标注（替换原来的力值标注）
    ax.text(x=0.05, y=0.90, z=save_fdm.max_z*1.05,
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
    ax.set_zlim(0, save_fdm.max_z * 1.1)  # 统一z轴尺度
    ax.view_init(elev=35, azim=45)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 更新状态计数器
    save_fdm.saved_states += 1
    
    # 如果是最后一个状态，立即保存
    if save_fdm.saved_states == max_states:
        filename = os.path.join(save_dir, f"FDM_States_{max_states}.png")
        save_fdm.fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(save_fdm.fig)
        delattr(save_fdm, 'fig')
        delattr(save_fdm, 'axes')
        delattr(save_fdm, 'saved_states')
        delattr(save_fdm, 'max_z')
        print(f"Saved full states to {filename}")


def finalize_fdm(save_dir="results", completed=False):
    """
    最终化处理函数
    
    参数:
        save_dir: 保存目录
        completed: 是否完成所有迭代 (True=已完成全部迭代，False=提前终止)
    """
    if not hasattr(save_fdm, 'fig') or save_fdm.saved_states == 0:
        return
    
    # 如果已完成所有迭代，保存当前进度（不强制填满）
    if completed:
        filename = os.path.join(save_dir, 
                              f"FDM_States_completed_{save_fdm.saved_states}.png")
    # 如果是提前终止，保存上一次有效迭代
    else:
        # 回退一个状态，因为最后一次迭代可能不完整
        save_fdm.saved_states = max(0, save_fdm.saved_states - 1)
        filename = os.path.join(save_dir,
                              f"FDM_States_partial_{save_fdm.saved_states+1}.png")
    
    # 保存图像
    save_fdm.fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(save_fdm.fig)
    
    # 打印保存信息
    if completed:
        print(f"Saved completed states ({save_fdm.saved_states}/{len(save_fdm.axes)}) to {filename}")
    else:
        print(f"Saved last valid state ({save_fdm.saved_states+1}) to {filename}")
    
    # 清理属性
    for attr in ['fig', 'axes', 'saved_states', 'max_z']:
        if hasattr(save_fdm, attr):
            delattr(save_fdm, attr)


# In[63]:


####### FDM part 
C = torch.zeros(n_elements, n_nodes, dtype=torch.float32, device=device)
for n, (i, j) in enumerate(connectivity):
    C[n, i - 1] = 1
    C[n, j - 1] = -1
    
px= torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
py = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
pz = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)


fixed_idces = torch.tensor([node - 1 for node in Fixed_nodes], device=device)
free_node_indices = torch.tensor([node - 1 for node in Free_nodes], device=device)

CF = C[:, fixed_idces]
CN = C[:, free_node_indices]

def FDM(Q, F_value, CN=CN, CF=CF, px=px, py=py, pz=pz, 
        fixed_idces=fixed_idces, free_node_indices=free_node_indices,
        node_coords=grid_points):
    
    pz[:, 0] = F_value
        
    Dn = torch.matmul(torch.transpose(CN, 0, 1), torch.matmul(Q, CN))
    DF = torch.matmul(torch.transpose(CN, 0, 1), torch.matmul(Q, CF))
    
    
    xF = node_coords[fixed_idces, 0].unsqueeze(1)
    yF = node_coords[fixed_idces, 1].unsqueeze(1)
    zF = node_coords[fixed_idces, 2].unsqueeze(1)
    
    xN = torch.matmul(torch.inverse(Dn), (px - torch.matmul(DF, xF)))
    yN = torch.matmul(torch.inverse(Dn), (py - torch.matmul(DF, yF)))
    zN = torch.matmul(torch.inverse(Dn), (pz - torch.matmul(DF, zF)))
        
    
    new_node_coords = node_coords.clone()
    new_node_coords[free_node_indices, 0] = xN.squeeze()
    new_node_coords[free_node_indices, 1] = yN.squeeze()
    new_node_coords[free_node_indices, 2] = zN.squeeze()
    
    return new_node_coords


# In[64]:


###### FE part
D_radius = 0.75
D_young_modulus = 10e9 
D_shear_modulus = 0.7e9 
D_poisson_ratio = 0.3
cross_section_angle_a = 0  
cross_section_angle_b = 0  
a_small_number = 1e-10

def rotation(v, k, theta):
    """Rotation of vector v around axis k by angle theta."""
    k = k / torch.norm(k)  # Normalize k
    cross_product = torch.cross(k, v)
    dot_product = torch.dot(k, v)

    # Ensure theta is a tensor
    theta = torch.tensor(theta, dtype=torch.float32, device=device) if not isinstance(theta, torch.Tensor) else theta

    v_rotated = v * torch.cos(theta) + cross_product * torch.sin(theta) + k * dot_product * (1 - torch.cos(theta))
    return v_rotated

class Beam:
    def __init__(self, node_coordinates, R=D_radius, young_modulus=D_young_modulus,
                 shear_modulus=D_shear_modulus, poisson_ratio=D_poisson_ratio, Beta_a=cross_section_angle_a,
                 Beta_b=cross_section_angle_b):
        self.node_coordinates = node_coordinates  # (2, 3) tensor for node coordinates

        # Material and geometry
        self.radius = R
        self.young_modulus = young_modulus
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio

        # Cross-sectional properties
        self.length = torch.norm(self.node_coordinates[1] - self.node_coordinates[0])  # Length of the beam
        self.Iy = (torch.pi * self.radius ** 4) / 4 
        self.Iz = self.Iy
        self.A = torch.pi * self.radius ** 2
        self.J = (torch.pi * self.radius ** 4) / 2

        # Stiffness components
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

        # Section rotations at the two ends
        self.Beta_a = Beta_a
        self.Beta_b = Beta_b

    def get_element_stiffness_matrix(self):
        """Element stiffness matrix."""
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
        """Coordinate transformation matrix."""
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

    def nodal_transform(self):
        """Coordinate transformation matrix."""
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
        return lambda_matrix


def assemble_stiffness_matrix(beams, n_nodes, n_dof_per_node, connectivity):
    """Global stiffness matrix assembly."""
    total_dof = n_nodes * n_dof_per_node  # Total degrees of freedom
    K_global = torch.zeros((total_dof, total_dof), dtype=torch.float32, device=device)
    
    for idx, (i, j) in enumerate(connectivity):
        Matrix_T = beams[idx].System_Transform()  # Get transformation matrix
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
    
    attempts = 0
    while attempts < max_attempts:
        reg = 1e-6 * torch.eye(K_global.shape[0], device=K_global.device)
        reg[fixed_dof, fixed_dof] = 0  
        K_reg = K_global + reg
        
        try:
            displacements = torch.linalg.solve(
                K_reg.to(torch.float64), 
                F.to(torch.float64)
            )
            return displacements.to(K_global.dtype)
            
        except RuntimeError:
            diag = torch.diag(K_global)
            extreme_mask = (diag > 1e12) & (~torch.isin(torch.arange(len(diag)), torch.tensor(fixed_dof)))  
            K_reg[extreme_mask] = 0
            K_reg[:, extreme_mask] = 0
            K_reg[extreme_mask, extreme_mask] = 1e12  
            
            K_reg[fixed_dof, :] = 0
            K_reg[:, fixed_dof] = 0
            K_reg[fixed_dof, fixed_dof] = 1e10  
            
            try:
                displacements, info = torch.linalg.cg(
                    K_reg.to(torch.float64),
                    F.to(torch.float64),
                    maxiter=5000,
                    atol=1e-6
                )
                if info > 0:
                    raise RuntimeError("CG nah nah")
                return displacements.to(K_global.dtype)
                
            except:
                K_pinv = torch.linalg.pinv(K_reg)
                K_pinv[fixed_dof, :] = 0  
                displacements = K_pinv @ F
                print("警告：使用伪逆求解，精度可能降低")
                return displacements
                
        attempts += 1
    
    raise RuntimeError("无法求解线性系统")


def Strain_E(node_coords, connectivity, fixed_dof, F):
    # Element Assembly
    Beam_lens = []
    beams = []
    for connection in connectivity:
        node_1_coords = node_coords[connection[0] - 1]
        node_2_coords = node_coords[connection[1] - 1]
        beam = Beam(node_coordinates=torch.stack([node_1_coords, node_2_coords]),
                    R=D_radius, young_modulus=D_young_modulus,
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
        ASE_list.append(0.5 * (Local_d_n[0]-Local_d_n[6]) * beams[n].S_u * (Local_d_n[0]-Local_d_n[6]))  
        V_list.append(beams[n].A * beams[n].length)
    
     
    
    Strain_energy = torch.stack(strain_energy_list)
    forces = torch.stack(force_list)
    ASE = torch.stack(ASE_list)
    lens = torch.stack(Beam_lens)
    # epsilon = Local_d[:, 0] / lens
    # Axial_d = Local_d[:, 0]
    V = torch.stack(V_list)
    SED = Strain_energy / lens 
    R = torch.var(SED)
    
    return Strain_energy, forces, displacements, ASE, lens, R, V


# In[65]:


def optimizer(q, gradients, step):
    
    q.data -= gradients / torch.norm(gradients ) * step
    
    with torch.no_grad():
        q[0] = torch.clamp(q[0], min=2.0, max=3.5)
        q[1] = torch.clamp(q[1], min=0.0, max=2.0)

    return q
    
def check_available_memory():
    """返回当前可用CPU内存（MB）"""
    return psutil.virtual_memory().available / (1024 ** 2)


# In[66]:


##### INItializaing the Q:
rows, cols = V_matrix.shape

q_rows = math.ceil(rows / 2)
q_cols = math.ceil(cols / 2)
q_v = torch.ones((q_rows, q_cols)) * 2 
rows_, cols_ = H_matrix.shape
q_rows_ = math.ceil(rows_ / 2)
q_cols_ = math.ceil(cols_ / 2)
q_h = torch.zeros((q_rows_, q_cols_))

q_cat = torch.cat([q_v.unsqueeze(0), q_h.unsqueeze(0)], dim=0)  # 形状 (2, q_rows, q_cols)
q = q_cat.clone().requires_grad_(True) 


# In[68]:


############### Formulating :::::
####### Gradient descent
step = 0.5
epochs = 600
# Initilizing
patience = 20
####### Force Condition

_, F_value = Force_mat(- 1, 2)
F_fe_g, _ = Force_mat(1, 1)

F_fe = F_fe_g 

r = 1 / torch.max(F_value)


# In[ ]:

###############################  FDM + FE part
############## Optimization loop

#### Initializing Data storage
os.makedirs("data_records", exist_ok=True)
optimization_data = {
    "metadata": {
        "project": "Structural Optimization",
        "Context": "FDM + FE",
        "device": str(device),
        "parameters": {
            "length": length,
            "width": width,
            "grid_size": f"{n1}x{n2}",
            "epochs": epochs,
            "step_size": step
        }
    },
    "iterations": []
}
####### Loop start
count = 0
n_elem = len(connectivity)
start_time = time.time()
cut = epochs / 5
ES_his = []
LP_his = []

# Loop start
for iteration in range(epochs + 1):
    print('ite', iteration)
    
    iter_start = time.time() 
    
    avail_mem = check_available_memory()
    print(f"Iter {iteration} - Available Memory: {avail_mem:.2f} MB")
    if avail_mem < 1000: 
        print(f"⚠️  Low memory warning: {avail_mem:.2f} MB left!")
    
    # Forwards
    q_v, q_h = q[0], q[1] 
    q_V = Symmetry_shaper(q_v, V_matrix)
    q_H = Symmetry_shaper(q_h, H_matrix)
    
    q_vec = torch.zeros(n_elements, device=device)
    
    
    for i in range(V_matrix.shape[0]): 
        for j in range(V_matrix.shape[1]): 
            index = V_matrix[i, j].item()  
            q_vec[index] = q_V[i, j] 
    for i in range(H_matrix.shape[0]):  
        for j in range(H_matrix.shape[1]):  
            index = H_matrix[i, j].item() 
            q_vec[index] = q_H[i, j] 
            
    q_vec = q_vec * 1 / r
    
    Q = torch.diag(q_vec) 
    new_node_coords = FDM(Q, F_value)
    
    height = max(new_node_coords[:,2])
     
    ####### FDM time
    FDM_time = (time.time() - iter_start) / 60
        
    N_coords = new_node_coords.clone()
    FE_str = time.time()
    Strain_energy, forces, displacements, ASE, Beam_lens, R, V = Strain_E(N_coords, connectivity, fixed_dof, F_fe)
    
    ######## FE time
    FE_time = (time.time() - FE_str) / 60       
    force = abs(forces[:, 0, 0])
    load_path = torch.dot(force , Beam_lens)


    Total_ES = torch.sum(Strain_energy) 
    Axial_rate = torch.sum(ASE) / Total_ES

    Volume = torch.sum(V)
    Loss = torch.dot(abs(q_vec * Beam_lens), Beam_lens)

    
    # Loss_his.append(loss.item())
    LP_his.append(load_path.item()) 
    ES_his.append(Total_ES.item())


    print("Estrain:", Total_ES)
    print('load_path', load_path)
    print('ite height:', height)

    
    Back_str = time.time()
    if q.grad is not None:
        q.grad.detach_()
        q.grad.zero_()
       
    Total_ES.backward(retain_graph=True)
    Back_time = (time.time() - Back_str) / 60
    
    # Grad
    gradients = q.grad
    frob_norm = torch.norm(gradients)
    q = optimizer(q, gradients, step)
    
    Wins_q = q.cpu().clone().detach()
    
    print('Wins_grad', gradients) 
    
            
    print('Wins_q', Wins_q)    
    ####### Data storage:
    iteration_record = {
    "iteration": iteration,
    "variables": q.detach().cpu().numpy().tolist(),
    "strain_energy": Total_ES.item(),
    "Load_path": load_path.item(),
    "Axial SE ratio": Axial_rate.item(),
    "Volume": Volume.item(),
    "R": R.item(),
    "gradient_norm": torch.norm(gradients).item() if q.grad is not None else 0.0,
               "timing": {
            "FDM_time": FDM_time,
            "FE_time": FE_time,
            "Back_propagation time": Back_time,
        },
    }  
    optimization_data["iterations"].append(iteration_record)

    if iteration % cut == 0 or iteration == 0:
        state_idx = iteration // cut  
        save_fdm(state_idx, grid_points, new_node_coords, 
                connectivity, Free_nodes, Fixed_nodes, 
                force, Total_ES.detach().cpu().item())
    
    # print iteration
    if iteration % 5 == 0:
        print(f"Iteration {iteration}: Normalized Gradient = {frob_norm}, Adaptive learning rate = {step}")

    if iteration % 10 == 0:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
            
    ####### Early stopping
    if iteration > 0:  
        Pre_Total_ES = ES_his[iteration - 1]  
        change = abs(Total_ES - Pre_Total_ES) / Pre_Total_ES 
        if change < 1/100000:
            count += 1
        else:
            count = 0 
        if count >= patience:
            print(f"Early stopping at iteration {iteration}: Total_ES change < 1%% for {patience} consecutive iterations.")
            break

finalize_fdm()
total_time = (time.time() - iter_start) / 60
optimization_data["metadata"].update({
    "Ite_time": total_time,
})

with open(os.path.join("data_records", "FULL_data.json"), 'w') as f:
    json.dump(optimization_data, f, indent=2)
    
print("Optimization completed.")
print(q)
print("Estrain:", Total_ES)

################################################################################################################## FE part

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
        F_value = torch.tensor([F_value] * len(Free_nodes), device=device) * 1000  # The force value/direction
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


def symmetry_shaper(lower_left_points, length=length, width=width):
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
    def __init__(self, node_coordinates, R=D_radius, young_modulus=D_young_modulus,
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


def Strain_E(node_coords, connectivity, fixed_dof, F):
    # Element Assembly
    Beam_lens = []
    beams = []
    for connection in connectivity:
        node_1_coords = node_coords[connection[0] - 1]
        node_2_coords = node_coords[connection[1] - 1]
        beam = Beam(node_coordinates=torch.stack([node_1_coords, node_2_coords]),
                    R=D_radius, young_modulus=D_young_modulus,
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
    ASE = torch.stack(ASE_list)
    lens = torch.stack(Beam_lens)
    V = torch.stack(V_list)
    # D = Local_d[:, 0]
    return Strain_energy, forces, displacements, ASE, V, lens


# 可视化函数（需要将数据移回CPU）
def save_OPT(iteration, grid_points, N_coords, force,
             connectivity, Free_nodes, Fixed_nodes, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    # 将数据移回CPU并转为numpy
    x_orig = grid_points[:, 0].cpu().detach().numpy()
    y_orig = grid_points[:, 1].cpu().detach().numpy()
    z_orig = grid_points[:, 2].cpu().detach().numpy()

    x_fdm = N_coords[:, 0].cpu().detach().numpy()
    y_fdm = N_coords[:, 1].cpu().detach().numpy()
    z_fdm = N_coords[:, 2].cpu().detach().numpy()
    force_np = force.cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始网格 (蓝色)
    for i, j in connectivity.cpu().numpy():
        ax.plot([x_orig[i - 1], x_orig[j - 1]],
                [y_orig[i - 1], y_orig[j - 1]],
                [z_orig[i - 1], z_orig[j - 1]], 'b-', linewidth=2, alpha=0.5, label='Grid' if i == 1 else "")

    # 绘制OPT解 (红色)
    for i, j in connectivity.cpu().numpy():
        ax.plot([x_fdm[i - 1], x_fdm[j - 1]],
                [y_fdm[i - 1], y_fdm[j - 1]],
                [z_fdm[i - 1], z_fdm[j - 1]], 'r-', linewidth=2, label='OPT solution' if i == 1 else "")

    # 绘制力值 (绿色标记+文本)
    for idx, (i, j) in enumerate(connectivity):
        mid_x = (x_fdm[i - 1] + x_fdm[j - 1]) / 2
        mid_y = (y_fdm[i - 1] + y_fdm[j - 1]) / 2
        mid_z = (z_fdm[i - 1] + z_fdm[j - 1]) / 2
        ax.scatter(mid_x, mid_y, mid_z, c='g', s=10)
        ax.text(mid_x, mid_y, mid_z, f"{force_np[idx]:.0f}", color='green', fontsize=8)

    # 绘制固定节点 (黑色)
    for node in Fixed_nodes.cpu().numpy():
        ax.scatter(x_fdm[node - 1], y_fdm[node - 1], z_fdm[node - 1],
                   c='k', s=50, marker='s', label='Fixed Node' if node == Fixed_nodes[0] else "")

    # 设置图例和标题
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'OPT Solution - Iteration {iteration}')

    # 保存图像
    filename = os.path.join(save_dir, f"E_strain_Opt(Y)_iter_{iteration}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filename}")


# In[2]:


def optimizer(OPT_variables, gradients, step):
    OPT_variables.data -= gradients / torch.norm(gradients) * step

    with torch.no_grad():
        OPT_variables.data = torch.clamp(OPT_variables.data, lower_bound, upper_bound)

    return OPT_variables


def check_available_memory():
    """返回当前可用CPU内存（MB）"""
    return psutil.virtual_memory().available / (1024 ** 2)


# 初始化
q = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
lower_bound, upper_bound = 0.0, 12.0
q_vec = torch.zeros(n_elements, device=device)
for i in range(len(idx_X)):
    q_vec[idx_X[i, :]] = q[i]
for j in range(len(idx_Y)):
    q_vec[idx_Y[j, :]] = q[j + len(idx_X)]

# 梯度下降参数
step = 0.1
epochs = 500
patience = 30
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

# In[4]:

FEES_his = []
FELP_his = []

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

    avail_mem = check_available_memory()
    print(f"Iter {iteration} - Available Memory: {avail_mem:.2f} MB")
    if avail_mem < 1000:
        print(f"⚠️  Low memory warning: {avail_mem:.2f} MB left!")

    # 前向传播
    Crd[iddx, 2] = OPT_variables
    N_coords = symmetry_shaper(Crd).clone()
    Strain_energy, forces, displacements, ASE, V, Beam_lens = Strain_E(N_coords, connectivity, fixed_dof, F_fe)

    Volume = torch.sum(V)

    force = abs(forces[:, 0, 0])
    Total_ES = torch.sum(Strain_energy)

    FEES_his.append(Total_ES.item())
    load_path = torch.dot(force, Beam_lens)
    FELP_his.append(load_path.item())

    SED = Total_ES / Beam_lens
    R = torch.var(SED)

    print('Total_ES', Total_ES)


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
    print(end_time)
    # 定期保存结果
    if iteration % 5 == 0:
        print(f"Iter {iteration}: Grad Norm = {frob_norm.item():.4f}, LR = {step}, Load_path = {load_path.item()}")

    if iteration % 10 == 0:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

print("Optimization completed.")
print("Final Strain Energy:", Total_ES.item())
#####################################################################################
# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot both curves
color1 = 'tab:blue'
color2 = 'tab:red'

# Plot ES_his (FDM + FE)
line1 = ax1.plot(ES_his,
                 label='FDM + FE (ES_his)',
                 color=color1,
                 linewidth=2,
                 linestyle='-')

# Plot FEES_his (FE only)
line2 = ax1.plot(FEES_his,
                 label='FE Only (FEES_his)',
                 color=color2,
                 linewidth=2,
                 linestyle='--')

# Set labels and title
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Strain Energy', color='black')
ax1.grid(True, linestyle=':', alpha=0.6)
plt.title('Strain Energy Comparison: FDM+FE vs FE Only')

# Add markers and annotations at key points
marker_points = [0, *range(50, max(len(ES_his), 50), len(ES_his) - 1]

for point in marker_points:
    if
point < len(ES_his):
ax1.scatter(point, ES_his[point], color=color1, zorder=5, s=60)
ax1.text(point, ES_his[point],
         f'FDM+FE: {ES_his[point]:.2e}',
         ha='right' if point == len(ES_his) - 1 else 'left',
         va='bottom',
         fontsize=8,
         bbox=dict(facecolor='white', alpha=0.7))

if point < len(FEES_his):
    ax1.scatter(point, FEES_his[point], color=color2, zorder=5, s=60, marker='s')
ax1.text(point, FEES_his[point],
         f'FE: {FEES_his[point]:.2e}',
         ha='right' if point == len(FEES_his) - 1 else 'left',
         va='top',
         fontsize=8,
         bbox=dict(facecolor='white', alpha=0.7))

# Combine y-axis ranges for proper scaling
combined_values = ES_his + FEES_his
ax1.set_ylim(min(combined_values) * 0.95, max(combined_values) * 1.05)

# Add legend
ax1.legend(loc='upper right')

# Save and show plot
plt.tight_layout()
plt.savefig('StrainEnergy_Comparison_FDM_FE_vs_FE.png', dpi=300, bbox_inches='tight')
plt.show()
