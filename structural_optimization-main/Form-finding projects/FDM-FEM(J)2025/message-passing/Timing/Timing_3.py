#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch_geometric.nn import MessagePassing
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
import torch.nn as nn

gc.collect()
device = torch.device('cpu')
print(f"Using device: {device}")

# In[3]:


############ customization
length = 48
width = 48
judge = 0


# In[4]:


def generate_rectangular_grid_sg(length, width, n1, n2, judge=0, z=0.0, height=0):
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


# In[5]:


def Symmetry_shaper(q, matrix):
    rows, cols = matrix.shape

    row_odd = rows % 2 == 1
    col_odd = cols % 2 == 1

    if row_odd:

        q = torch.cat([q, q.flip(0)[1:, :]], dim=0)
    else:

        q = torch.cat([q, q.flip(0)], dim=0)

    if col_odd:

        q = torch.cat([q, q.flip(1)[:, 1:]], dim=1)
    else:

        q = torch.cat([q, q.flip(1)], dim=1)

    return q


############################################################# Force condition
def Force_mat(F_value, F_type, total_dof, Free_nodes, judge=0):
    F = torch.zeros(total_dof, dtype=torch.float32, device=device)

    if judge == 0:
        F_value = torch.tensor([F_value] * len(Free_nodes), device=device) * 1000  # The force value/direction
        F_type = [F_type] * len(Free_nodes)  # The force type
    else:
        F_value = torch.tensor(F_value) * 1000

    for idx, i in enumerate(Free_nodes):
        F[6 * (i - 1) + F_type[idx]] = F_value[idx]  # unit: KN / KN*m

    F = F.view(-1, 6)

    return F, F_value


def FDM(Q, F_value, CN, CF, px, py, pz,
        fixed_idces, free_node_indices,
        node_coords):
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


# In[10]:


###### FE
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


# In[11]:


def branch_node_to_directed_edge_index(branch_node_mat):
    """
    Transform branch-node matrix into edge_index (directed graph, one direction per branch)
    Input:
        branch_node_mat: [num_branches, num_nodes]
    Return:
        edge_index: [2, num_edges] where num_edges = num_branches
    """
    edge_list = []

    for branch in branch_node_mat:
        nodes = torch.where(branch != 0)[0].tolist()
        if len(nodes) != 2:
            raise ValueError("Error: Branch must connect exactly 2 nodes")

        node1, node2 = nodes[0], nodes[1]
        edge_list.append([min(node1, node2), max(node1, node2)])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


# In[12]:


class FixedNodeProcessor_CG:
    def __init__(self, fixed_nodes, num_nodes):
        self.fixed_nodes = fixed_nodes
        self.num_nodes = num_nodes

    def create_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[self.fixed_nodes] = 0
        return mask

    def apply_mask(self, tensor):
        mask = self.create_mask(tensor)
        return tensor * mask


# In[13]:


def optimizer(tensor, gradients, step, max_step=0.1, eps=1e-8):
    grad_norm = gradients.norm()
    normalized_grad = gradients / (grad_norm + eps)

    effective_step = min(step, max_step)

    tensor.data -= effective_step * normalized_grad

    return tensor


def check_available_memory():
    return psutil.virtual_memory().available / (1024 ** 2)


# In[14]:


import torch
from torch_geometric.nn import MessagePassing


class ParallelBeamCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_i, pos_j, radius, young_modulus, shear_modulus, poisson_ratio, angle_a, angle_b):
        # Beam instance
        beams = [Beam(
            node_coordinates=torch.stack([pi, pj]),
            R=r, young_modulus=ym, shear_modulus=sm,
            poisson_ratio=pr, Beta_a=aa, Beta_b=ab
        ) for pi, pj, r, ym, sm, pr, aa, ab in zip(
            pos_i, pos_j, radius, young_modulus,
            shear_modulus, poisson_ratio, angle_a, angle_b
        )]

        Ts = torch.stack([b.System_Transform() for b in beams])
        Ks = torch.stack([b.get_element_stiffness_matrix() for b in beams])

        return Ts, Ks


class CG1(MessagePassing):
    def __init__(self,
                 radius=0.1,
                 young_modulus=2.1e11,
                 shear_modulus=8.1e10,
                 poisson_ratio=0.3,
                 angle_a=0.0,
                 angle_b=0.0):
        super().__init__(aggr=None, flow='source_to_target')

        # Parametric choices
        self.radius = torch.nn.Parameter(torch.tensor(float(radius)))
        self.young_modulus = torch.nn.Parameter(torch.tensor(float(young_modulus)))
        self.shear_modulus = torch.nn.Parameter(torch.tensor(float(shear_modulus)))
        self.poisson_ratio = torch.nn.Parameter(torch.tensor(float(poisson_ratio)))
        self.angle_a = torch.nn.Parameter(torch.tensor(float(angle_a)))
        self.angle_b = torch.nn.Parameter(torch.tensor(float(angle_b)))

        self.calculator = ParallelBeamCalculator()

    def forward(self, pos, edge_index, displacement):
        return self.propagate(edge_index, pos=pos, displacement=displacement)

    def message(self, pos_i, pos_j, displacement_i, displacement_j):
        num_edges = pos_i.shape[0]
        radius = self.radius.expand(num_edges)
        ym = self.young_modulus.expand(num_edges)
        sm = self.shear_modulus.expand(num_edges)
        pr = self.poisson_ratio.expand(num_edges)
        aa = self.angle_a.expand(num_edges)
        ab = self.angle_b.expand(num_edges)

        T, K_local = self.calculator(pos_i, pos_j, radius, ym, sm, pr, aa, ab)

        # Ku:
        K_global = torch.bmm(T.transpose(1, 2), torch.bmm(K_local, T))
        u = torch.cat([displacement_i, displacement_j], dim=-1).unsqueeze(-1)
        force = torch.bmm(K_global, u).squeeze(-1)

        return force

    def aggregate(self, inputs, index, dim_size=None):
        return inputs


# In[15]:


class CG2(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_attr, fixed_processor=None):
        """
        input:
            x: nodal forces [num_nodes, 6] (fx, fy, fz, mx, my, mz)
            edge_index: [2, num_edges]
            edge_attr: from Ku: [num_edges, 12]
            fixed_processor: mask the fixed nodes
        """
        force_ij = edge_attr[:, :6]  # i→j [num_edges,6]
        force_ji = edge_attr[:, 6:]  # j→i [num_edges,6]
        reverse_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        full_index = torch.cat([edge_index, reverse_index], dim=1)
        full_forces = torch.cat([force_ij, force_ji], dim=0)
        return self.propagate(full_index, x=x, edge_attr=full_forces,
                              fixed_processor=fixed_processor)

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x, fixed_processor):
        """
            residual = external_forces - sum(internal_forces)
        """
        # aggr_out.shape: [num_nodes,6]
        residual = x - aggr_out

        if fixed_processor is not None:
            residual = fixed_processor.apply_mask(residual)

        return residual


# In[16]:


def compute_system_force(pos, edge_index, field, fixed_processor=None):
    """General internal action calculator for Ad and for Ku"""
    # Elemental wise
    edge_forces = CG1()(pos, edge_index, field)

    # Vertices wise
    nodal_forces = CG2()(
        x=torch.zeros_like(field),
        edge_index=edge_index,
        edge_attr=edge_forces,
        fixed_processor=fixed_processor
    )
    return nodal_forces


# In[25]:


def local_conjugate_gradient(pos, edge_index, F_ext, fixed_nodes, max_iter=300, tol=0.05):
    num_nodes = pos.shape[0]
    u = torch.zeros(num_nodes, 6, device=pos.device)
    processor = FixedNodeProcessor_CG(fixed_nodes, num_nodes)
    r = F_ext - compute_system_force(pos, edge_index, u, processor)
    u = u.reshape(1, -1).clone()
    d = r.clone()
    r_logged = r * 1e6
    r = r.reshape(1, -1).clone()
    r_logged = r_logged.reshape(1, -1).clone()
    for k in range(1, max_iter + 1):
        Ad = compute_system_force(pos, edge_index, d, processor)
        d = d.reshape(1, -1).clone()
        Ad = Ad.reshape(1, -1).clone()
        alpha = (r @ r.T) / (d @ Ad.T + 1e-10)
        u = u + alpha * d
        r = r - alpha * Ad
        r_norm = torch.norm(r)

        if r_norm < tol * torch.norm(F_ext):
            print(f"Converged at iteration {k}")
            break

        Beta = -(r @ r.T) / (r_logged @ r_logged.T + 1e-10)
        d = r - Beta * d.clone()
        d = d.reshape(num_nodes, 6)
        r_logged = r.clone()

    u = u.reshape(num_nodes, 6)
    return processor.apply_mask(u)


# In[19]:


class FixedNodeProcessor:
    def __init__(self, fixed_nodes, num_nodes):
        self.fixed_nodes = torch.as_tensor(fixed_nodes)
        self.num_nodes = num_nodes
        self.mask = self._create_fixed_mask()
        self.fixed_values = None

    def _create_fixed_mask(self):
        mask = torch.ones(6 * self.num_nodes, dtype=torch.bool)
        for node in self.fixed_nodes:
            start = 6 * node
            mask[start:start + 6] = False
        return mask

    def set_fixed_values(self, fixed_values):
        if isinstance(fixed_values, torch.Tensor):
            self.fixed_values = fixed_values.flatten()
        else:
            self.fixed_values = torch.tensor(fixed_values).flatten()

    def apply_mask(self, u):
        return u * self.mask

    def apply_fixed_conditions(self, u):
        u = self.apply_mask(u)

        if self.fixed_values is not None:
            fixed_vec = torch.zeros_like(u)
            for i, node in enumerate(self.fixed_nodes):
                start = 6 * node
                fixed_vec[start:start + 6] = self.fixed_values[i * 6: (i + 1) * 6]

            u += fixed_vec

        return u

    def apply_residual_mask(self, r):
        return r * self.mask


class ParallelBeamCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_i, pos_j, radius, young_modulus, shear_modulus, poisson_ratio, angle_a, angle_b):
        beams = [Beam(
            node_coordinates=torch.stack([pi, pj]),
            R=r, young_modulus=ym, shear_modulus=sm,
            poisson_ratio=pr, Beta_a=aa, Beta_b=ab
        ) for pi, pj, r, ym, sm, pr, aa, ab in zip(
            pos_i, pos_j, radius, young_modulus,
            shear_modulus, poisson_ratio, angle_a, angle_b
        )]

        Ts = torch.stack([b.System_Transform() for b in beams])
        Ks = torch.stack([b.get_element_stiffness_matrix() for b in beams])

        return Ts, Ks


class From_edges_to_global(MessagePassing):
    def __init__(self,
                 radius=0.1,
                 young_modulus=2.1e11,
                 shear_modulus=8.1e10,
                 poisson_ratio=0.3,
                 angle_a=0.0,
                 angle_b=0.0):
        super().__init__(aggr=None, flow='source_to_target')

        # Parametric choices
        self.radius = torch.nn.Parameter(torch.tensor(float(radius)))
        self.young_modulus = torch.nn.Parameter(torch.tensor(float(young_modulus)))
        self.shear_modulus = torch.nn.Parameter(torch.tensor(float(shear_modulus)))
        self.poisson_ratio = torch.nn.Parameter(torch.tensor(float(poisson_ratio)))
        self.angle_a = torch.nn.Parameter(torch.tensor(float(angle_a)))
        self.angle_b = torch.nn.Parameter(torch.tensor(float(angle_b)))

        self.calculator = ParallelBeamCalculator()

    def forward(self, pos, edge_index, num_nodes):
        # Initialize global stiffness matrix
        self.global_K = torch.zeros((6 * num_nodes, 6 * num_nodes),
                                    dtype=torch.float32, device=pos.device)

        # Propagate to compute and assemble stiffness matrices
        self.propagate(edge_index, pos=pos, num_nodes=num_nodes)

        return self.global_K

    def message(self, pos_i, pos_j):
        num_edges = pos_i.shape[0]
        radius = self.radius.expand(num_edges)
        ym = self.young_modulus.expand(num_edges)
        sm = self.shear_modulus.expand(num_edges)
        pr = self.poisson_ratio.expand(num_edges)
        aa = self.angle_a.expand(num_edges)
        ab = self.angle_b.expand(num_edges)

        T, K_local = self.calculator(pos_i, pos_j, radius, ym, sm, pr, aa, ab)

        # Transform to global coordinates
        K_global = torch.bmm(T.transpose(1, 2), torch.bmm(K_local, T))

        # Split into submatrices
        Kii = K_global[:, :6, :6]  # (num_edges, 6, 6)
        Kij = K_global[:, :6, 6:]  # (num_edges, 6, 6)
        Kji = K_global[:, 6:, :6]  # (num_edges, 6, 6)
        Kjj = K_global[:, 6:, 6:]  # (num_edges, 6, 6)

        return {'Kii': Kii, 'Kij': Kij, 'Kji': Kji, 'Kjj': Kjj}

    def aggregate(self, inputs, edge_index, ptr=None, dim_size=None):
        # Get source and target indices from edge_index
        row, col = edge_index  # row is source nodes, col is target nodes

        # Add contributions to global matrix
        for i in range(row.shape[0]):
            r_node = row[i].item()
            c_node = col[i].item()

            # Get the submatrices for this edge
            Kii = inputs['Kii'][i]
            Kij = inputs['Kij'][i]
            Kji = inputs['Kji'][i]
            Kjj = inputs['Kjj'][i]

            # Add to global matrix (using in-place operations)
            self.global_K[6 * r_node:6 * r_node + 6, 6 * r_node:6 * r_node + 6] += Kii
            self.global_K[6 * r_node:6 * r_node + 6, 6 * c_node:6 * c_node + 6] += Kij
            self.global_K[6 * c_node:6 * c_node + 6, 6 * r_node:6 * r_node + 6] += Kji
            self.global_K[6 * c_node:6 * c_node + 6, 6 * c_node:6 * c_node + 6] += Kjj

        return None

    # In[26]:


def Global_Conjugate_gradient(displacement_ini, new_node_coords, edge_index, F, fixed_nodes, max_iters=100):
    u = displacement_ini.clone().requires_grad_(True)
    pos = new_node_coords
    num_nodes = pos.shape[0]
    k = 1
    stiffness_calculator = From_edges_to_global()
    processor = FixedNodeProcessor(fixed_nodes, num_nodes)
    K = stiffness_calculator(pos, edge_index, num_nodes)
    while k <= max_iters:
        u = processor.apply_fixed_conditions(u)
        residual = F - K @ u.T
        residual = processor.apply_residual_mask(residual)
        if torch.norm(residual) <= 0.05 * torch.norm(F):
            print(f"Solved with early stop at {k} iteration")
            break
        elif k == 1:
            g = residual.clone()
            d = residual.clone()
            g_saved = g
            Alpha = (g.T @ g) / (d.T @ K @ d)
            u += Alpha * d
            k += 1
        elif k > 1:
            g = residual.clone()
            Beta = -(g.T @ g) / (g_saved.T @ g_saved)
            g_saved = g
            d = g - Beta * d
            Alpha = (g.T @ g) / (d.T @ K @ d)
            u += Alpha * d
            k += 1

    return u.detach()


import time
import json
import matplotlib.pyplot as plt


def run_experiments():
    # Parameters to test
    results = {
        'grid_sizes': list(range(13, 77, 4)),
        'local_cg': {'fe_times': [], 'back_times': []},
        'global_cg': {'fe_times': [], 'back_times': []},
        'global_ln': {'fe_times': [], 'back_times': []}
    }

    for i, size in enumerate(results['grid_sizes']):
        print(f"\nRunning experiment {i + 1}/{len(results['grid_sizes'])} with grid size: {size}x{size}")

        # Update grid size
        n1 = n2 = size

        # Regenerate geometry with new size
        grid_points = generate_rectangular_grid_sg(length, width, n1, n2, judge)
        connectivity = generate_connectivity_matrix(grid_points)

        # Update dependent variables
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

        Free_nodes = torch.tensor(Free_nodes, device=device)
        Fixed_nodes = Fixed_nodes.to(device)
        grid_points[Free_nodes - 1, 2] = 0.0
        #######################################
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
        H_matrix = H_matrix.reshape(n1 + 1, n2)
        H_matrix = H_matrix.flip(0)

        rows, cols = V_matrix.shape

        q_rows = math.ceil(rows / 2)
        q_cols = math.ceil(cols / 2)
        q_v = torch.ones((q_rows, q_cols)) * 5.44

        rows_, cols_ = H_matrix.shape
        q_rows_ = math.ceil(rows_ / 2)
        q_cols_ = math.ceil(cols_ / 2)
        q_h = torch.ones((q_rows_, q_cols_)) * 0.0

        q_cat = torch.cat([q_v.unsqueeze(0), q_h.unsqueeze(0)], dim=0)
        q = q_cat.clone()

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
                ####### FDM part
        C = torch.zeros(n_elements, n_nodes, dtype=torch.float32, device=device)
        for n, (i, j) in enumerate(connectivity):
            C[n, i - 1] = 1
            C[n, j - 1] = -1

        px = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
        py = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)
        pz = torch.zeros(len(Free_nodes), 1, dtype=torch.float32, device=device)

        fixed_idces = torch.tensor([node - 1 for node in Fixed_nodes], device=device)
        free_node_indices = torch.tensor([node - 1 for node in Free_nodes], device=device)

        CF = C[:, fixed_idces]
        CN = C[:, free_node_indices]

        # Update matrices
        edge_index = branch_node_to_directed_edge_index(C)
        F_vert, F_value = Force_mat(-1, 2, total_dof=6 * n_nodes, Free_nodes=Free_nodes, judge=0)
        F_ext, F_value = Force_mat(-1, 2, total_dof=6 * n_nodes, Free_nodes=Free_nodes, judge=0)
        r = 1 / torch.max(F_value)
        q_vec_Q = q_vec * 1 / r
        Q = torch.diag(q_vec_Q)
        new_node_coords = FDM(Q, F_value, CN=CN, CF=CF, px=px, py=py, pz=pz,
        fixed_idces=fixed_idces, free_node_indices=free_node_indices,
        node_coords=grid_points)

        pos = new_node_coords.clone().requires_grad_(True)
        num_nodes = len(new_node_coords)
        F = F_vert.flatten()
        disp_ini = torch.ones(num_nodes * 6) * 0.00001
        fixed_nodes = Fixed_nodes - 1

        # Run Local Conjugate Gradient
        # ===== 1. Local Conjugate Gradient =====
        print("Running Local CG...")
        u1, FE_localCG = None, 10e8  # 默认失败值

        try:
            cg_str = time.time()
            u1 = local_conjugate_gradient(pos=pos.clone(), edge_index=edge_index,
                                          F_ext=F_ext, fixed_nodes=fixed_nodes)
            FE_localCG = time.time() - cg_str
            results['local_cg']['fe_times'].append(FE_localCG)
        except Exception as e:
            print(f"Local CG forward failed: {str(e)}")
            results['local_cg']['fe_times'].append(10e8)
            results['local_cg']['back_times'].append(10e8)
            u1 = None

        # --- 反向传播（仅当前向成功时执行）---
        if u1 is not None:
            try:
                f = compute_system_force(new_node_coords.clone(), edge_index, u1)
                Strain_energy1 = 0.5 * f.reshape(1, -1) @ u1.reshape(1, -1).T
                back_str = time.time()
                Strain_energy1.backward()
                back_localCG = time.time() - back_str
                results['local_cg']['back_times'].append(back_localCG)
            except Exception as e:
                print(f"Local CG backward failed: {str(e)}")
                results['local_cg']['back_times'].append(10e8)

        # ===== 2. Global Conjugate Gradient =====
        print("Running Global CG...")
        displacement_output, fe_globalCG = None, 10e8

        try:
            fe_str = time.time()
            displacement_output = Global_Conjugate_gradient(
                displacement_ini=disp_ini,
                new_node_coords=pos,
                edge_index=edge_index,
                F=F,
                fixed_nodes=fixed_nodes,
                max_iters=300
            )
            fe_globalCG = time.time() - fe_str
            results['global_cg']['fe_times'].append(fe_globalCG)
        except Exception as e:
            print(f"Global CG forward failed: {str(e)}")
            results['global_cg']['fe_times'].append(10e8)
            results['global_cg']['back_times'].append(10e8)
            displacement_output = None

        if displacement_output is not None:
            try:
                stiffness_calculator = From_edges_to_global()
                K = stiffness_calculator(pos, edge_index, num_nodes)
                Strain_energy2 = 0.5 * displacement_output.T @ K @ displacement_output
                back_str = time.time()
                Strain_energy2.backward()
                back_globalCG = time.time() - back_str
                results['global_cg']['back_times'].append(back_globalCG)
            except Exception as e:
                print(f"Global CG backward failed: {str(e)}")
                results['global_cg']['back_times'].append(10e8)

        # ===== 3. Global Linear Solver =====
        print("Running Global LN...")
        u, fe_LN = None, 10e8

        try:
            fixed_dof = []
            for node in fixed_nodes:
                fixed_dof.extend([(node - 1) * 6 + i for i in range(6)])

            fe_str = time.time()
            K = stiffness_calculator(pos, edge_index, num_nodes)
            K[fixed_dof, :] = 0
            K[:, fixed_dof] = 0
            K[fixed_dof, fixed_dof] = 1e10
            reg = 1e-6 * torch.eye(K.shape[0], device=K.device)
            reg[fixed_dof, fixed_dof] = 0
            K_reg = K + reg
            u = torch.linalg.solve(K_reg.to(torch.float32), F.to(torch.float32))
            fe_LN = time.time() - fe_str
            results['global_ln']['fe_times'].append(fe_LN)
        except Exception as e:
            print(f"Global LN forward failed: {str(e)}")
            results['global_ln']['fe_times'].append(10e8)
            results['global_ln']['back_times'].append(10e8)
            u = None

        # --- 反向传播 ---
        if u is not None:
            try:
                Strain_energy3 = 0.5 * u.T.to(torch.float32) @ K @ u.to(torch.float32)
                back_str = time.time()
                Strain_energy3.backward()
                back_LN = time.time() - back_str
                results['global_ln']['back_times'].append(back_LN)
            except Exception as e:
                print(f"Global LN backward failed: {str(e)}")
                results['global_ln']['back_times'].append(10e8)

        if (i + 1) % 2 == 0:
            try:
                with open('checkpoint.json', 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Checkpoint saved after {i + 1} iterations")
            except Exception as e:
                print(f"Failed to save checkpoint: {str(e)}")


        try:
            with open('final_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            print("Final results saved to final_results.json")
        except Exception as e:
            print(f"Failed to save final results: {str(e)}")

        return results



if __name__ == "__main__":
    results = run_experiments()

    with open('timing_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # plot_results(results)
