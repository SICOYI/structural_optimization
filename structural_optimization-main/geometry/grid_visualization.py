import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import List, Tuple


def matplotlib_grid(
        grid_points: List[List[float]],
        connectivity: List[Tuple[int, int]],
        fixed_nodes: List[int] = None,
        title: str = "Grid Visualization"
):
    """
    使用Matplotlib绘制3D网格
    Args:
        grid_points: 网格点坐标
        connectivity: 连接关系
        fixed_nodes: 固定节点索引列表
        title: 图表标题
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标
    x = [p[0] for p in grid_points]
    y = [p[1] for p in grid_points]
    z = [p[2] for p in grid_points]

    # 绘制连接线
    for i, j in connectivity:
        ax.plot(
            [x[i - 1], x[j - 1]],
            [y[i - 1], y[j - 1]],
            [z[i - 1], z[j - 1]],
            'b-', alpha=0.6, linewidth=1
        )

    # 绘制网格点
    ax.scatter(x, y, z, c='r', marker='o', s=30, label='Nodes')

    # 标记固定节点
    if fixed_nodes:
        fixed_x = [x[i - 1] for i in fixed_nodes]
        fixed_y = [y[i - 1] for i in fixed_nodes]
        fixed_z = [z[i - 1] for i in fixed_nodes]
        ax.scatter(fixed_x, fixed_y, fixed_z, c='k', marker='s', s=50, label='Fixed Nodes')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotly_grid(
        grid_points: List[List[float]],
        connectivity: List[Tuple[int, int]],
        fixed_nodes: List[int] = None,
        title: str = "Interactive Grid"
) -> go.Figure:
    """
    使用Plotly生成交互式3D可视化
    Args:
        grid_points: 网格点坐标
        connectivity: 连接关系
        fixed_nodes: 固定节点索引列表
        title: 图表标题
    Returns:
        plotly.graph_objects.Figure 对象
    """
    fig = go.Figure()

    # 提取坐标
    x = [p[0] for p in grid_points]
    y = [p[1] for p in grid_points]
    z = [p[2] for p in grid_points]

    # 添加连接线
    for i, j in connectivity:
        fig.add_trace(go.Scatter3d(
            x=[x[i - 1], x[j - 1]],
            y=[y[i - 1], y[j - 1]],
            z=[z[i - 1], z[j - 1]],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False
        ))

    # 添加普通节点
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Nodes'
    ))

    # 添加固定节点
    if fixed_nodes:
        fixed_x = [x[i - 1] for i in fixed_nodes]
        fixed_y = [y[i - 1] for i in fixed_nodes]
        fixed_z = [z[i - 1] for i in fixed_nodes]
        fig.add_trace(go.Scatter3d(
            x=fixed_x, y=fixed_y, z=fixed_z,
            mode='markers',
            marker=dict(size=6, color='black'),
            name='Fixed Nodes'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig