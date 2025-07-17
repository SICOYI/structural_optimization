import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

# 创建图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D箭头（从 (0,0,0.5) 指向 (0,0,-1.5)，绿色）
arrow = Arrow3D(
    xs=[0, 0], ys=[0, 0], zs=[0.5, -1.5],
    mutation_scale=20, lw=3, arrowstyle="-|>", color="green"
)
ax.add_artist(arrow)

# 设置视角和标题
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-2, 1)
ax.set_title("3D Arrow (Force Direction)", pad=20)
ax.set_axis_off()  # 隐藏坐标轴

plt.tight_layout()
plt.show()