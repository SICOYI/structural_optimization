import json
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取JSON文件
with open('checkpoint.json', 'r') as f:
    data = json.load(f)

# 2. 提取数据
grid_sizes = data['grid_sizes'][:len(data['local_cg']['fe_times'])]

methods = ['local_cg', 'global_cg', 'global_ln']
line_styles = ['-', '--', ':']
colors = {'fe_times': 'red', 'back_times': 'blue'}

# 3. 创建画布
plt.figure(figsize=(12, 6))
plt.grid(True, alpha=0.3)
plt.title('Computation Time Comparison', fontsize=14)
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)

# 4. 绘制曲线
for i, method in enumerate(methods):
    # 前向计算时间 (fe_times)
    plt.plot(grid_sizes, data[method]['fe_times'],
             label=f'{method} (FE)',
             linestyle=line_styles[i],
             color=colors['fe_times'],
             marker='o')

    # 反向传播时间 (back_times)
    plt.plot(grid_sizes, data[method]['back_times'],
             label=f'{method} (Back)',
             linestyle=line_styles[i],
             color=colors['back_times'],
             marker='s')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.yscale('log')  # 对数坐标（因时间跨度大）
plt.xticks(grid_sizes, rotation=45)
plt.tight_layout()

plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')
plt.show()